# resolveiq/tests/test_cases.py
"""Hermetic unit tests for similar-case retrieval, privacy safeguards, and temporal masking.

100% offline: relies on zero network sockets, no Gemini API calls, and no running Qdrant daemon.
"""

from datetime import timezone
from unittest.mock import MagicMock

import pytest
from google.genai.errors import APIError
from pydantic import ValidationError

from resolveiq.cases import (
    RetrievalRequestError,
    RetrievalUnavailableError,
    RetrievedCaseItem,
    SimilarCasesRequest,
    SimilarCasesResponse,
    build_embedding_text,
    format_retrieved_case,
    parse_and_normalize_timestamp,
    retrieve_similar_cases,
)

BASE_PAYLOAD = {
    "unique_key": "60428192",
    "shape_id": 1,
    "shape_slug": "APT_NO_HEAT",
    "complaint_type": "HEAT/HOT WATER",
    "location_type": "RESIDENTIAL BUILDING",
    "descriptor": "APARTMENT ONLY",
    "descriptor_2": "NO HEAT",
    "borough": "MANHATTAN",
    "created_date": "2024-01-27T10:00:00",
    "status": "Closed",
    "resolution_description": "The Department of HPD inspected the complaint...",
}


def test_transient_dependency_failure_is_classified():
    """Verify an embedding connection outage becomes temporary unavailability."""
    request = SimilarCasesRequest(
        complaint_type="HEAT/HOT WATER",
        location_type="RESIDENTIAL BUILDING",
        descriptor="APARTMENT ONLY",
        intake_time="2024-01-27T18:00:00",
    )

    def unavailable_embedder(_text):
        raise ConnectionError("embedding service unavailable")

    with pytest.raises(RetrievalUnavailableError):
        retrieve_similar_cases(request, embedder=unavailable_embedder)


def test_non_retryable_provider_failure_is_not_classified_as_temporary():
    """Verify authentication failures become a separate safe retrieval error."""
    request = SimilarCasesRequest(
        complaint_type="HEAT/HOT WATER",
        location_type="RESIDENTIAL BUILDING",
        descriptor="APARTMENT ONLY",
        intake_time="2024-01-27T18:00:00",
    )

    def unauthorized_embedder(_text):
        raise APIError(401, {"status": "UNAUTHENTICATED", "message": "secret details"})

    with pytest.raises(RetrievalRequestError) as error:
        retrieve_similar_cases(request, embedder=unauthorized_embedder)

    assert "temporarily unavailable" not in str(error.value)
    assert "secret details" not in str(error.value)


def test_timezone_normalization_equivalent_offsets_not_earlier():
    """Prove that equivalent timestamps with different offsets compare as the same instant.

    A naive NYC collection timestamp '2024-01-27T18:00:00' (EST, UTC-5)
    and an aware UTC timestamp '2024-01-27T23:00:00+00:00' represent the exact same instant.
    Under the strict '<' rule, closed_date is NOT strictly earlier, so outcomes must be masked.
    """
    naive_nyc = "2024-01-27T18:00:00"
    aware_utc = "2024-01-27T23:00:00+00:00"
    aware_est = "2024-01-27T18:00:00-05:00"

    dt_naive, _ = parse_and_normalize_timestamp(naive_nyc)
    dt_utc, _ = parse_and_normalize_timestamp(aware_utc)
    dt_est, _ = parse_and_normalize_timestamp(aware_est)

    assert dt_naive == dt_utc == dt_est
    assert dt_naive.tzinfo == timezone.utc

    payload = {**BASE_PAYLOAD, "closed_date": naive_nyc}
    result = format_retrieved_case(payload, similarity_score=0.9, intake_time_str=aware_utc)
    assert result is not None
    assert result["is_historical_evidence"] is False
    assert result["status"] is None
    assert result["closed_date"] is None
    assert result["resolution_description"] is None


def test_case_created_after_intake_excluded():
    """Admission Gate: Case created after intake_time is excluded entirely (returns None)."""
    intake = "2024-01-27T18:00:00"
    payload = {**BASE_PAYLOAD, "created_date": "2024-01-27T19:00:00"}

    result = format_retrieved_case(payload, similarity_score=0.95, intake_time_str=intake)
    assert result is None


def test_case_created_exactly_at_intake_excluded():
    """Admission Gate: Case created exactly at intake_time (created_date == intake_time) is excluded entirely."""
    intake = "2024-01-27T18:00:00"
    payload = {**BASE_PAYLOAD, "created_date": "2024-01-27T18:00:00"}

    result = format_retrieved_case(payload, similarity_score=0.95, intake_time_str=intake)
    assert result is None


def test_temporal_boundary_1_eligible_closed_before():
    """Boundary 1: Closed strictly before intake -> outcomes exposed."""
    intake = "2024-01-27T18:00:00"
    payload = {**BASE_PAYLOAD, "closed_date": "2024-01-27T16:30:00"}

    res = format_retrieved_case(payload, similarity_score=0.95, intake_time_str=intake)
    assert res is not None
    assert res["is_historical_evidence"] is True
    assert res["status"] == "Closed"
    assert res["closed_date"] == "2024-01-27T16:30:00"
    assert res["resolution_description"] is not None


def test_temporal_boundary_2_ineligible_closed_after():
    """Boundary 2: Closed strictly after intake -> outcomes masked."""
    intake = "2024-01-27T18:00:00"
    payload = {**BASE_PAYLOAD, "closed_date": "2024-01-27T19:00:00"}

    res = format_retrieved_case(payload, similarity_score=0.95, intake_time_str=intake)
    assert res is not None
    assert res["is_historical_evidence"] is False
    assert res["status"] is None
    assert res["closed_date"] is None
    assert res["resolution_description"] is None


def test_temporal_boundary_3_ineligible_closed_exactly_at():
    """Boundary 3: Closed exactly at intake (strict <, not <=) -> outcomes masked."""
    intake = "2024-01-27T18:00:00"
    payload = {**BASE_PAYLOAD, "closed_date": "2024-01-27T18:00:00"}

    res = format_retrieved_case(payload, similarity_score=0.95, intake_time_str=intake)
    assert res is not None
    assert res["is_historical_evidence"] is False
    assert res["status"] is None
    assert res["closed_date"] is None
    assert res["resolution_description"] is None


def test_temporal_boundary_4_ineligible_open_ticket():
    """Boundary 4: No closure date (open ticket) -> outcomes masked."""
    intake = "2024-01-27T18:00:00"
    payload = {**BASE_PAYLOAD, "closed_date": None, "status": "Open"}

    res = format_retrieved_case(payload, similarity_score=0.95, intake_time_str=intake)
    assert res is not None
    assert res["is_historical_evidence"] is False
    assert res["status"] is None
    assert res["closed_date"] is None
    assert res["resolution_description"] is None


def test_nonexistent_local_intake_time_rejected():
    """Spring forward transition gap in America/New_York (e.g. 2024-03-10 02:30:00) is rejected."""
    with pytest.raises(ValidationError):
        SimilarCasesRequest(
            complaint_type="HEAT/HOT WATER",
            location_type="RESIDENTIAL BUILDING",
            descriptor="APARTMENT ONLY",
            intake_time="2024-03-10T02:30:00",
        )


def test_ambiguous_naive_timestamp_conservatively_masked():
    """Fall back transition fold in America/New_York is ambiguous; outcomes are conservatively suppressed."""
    ambiguous_time = "2024-11-03T01:30:00"
    _, is_ambig = parse_and_normalize_timestamp(ambiguous_time)
    assert is_ambig is True

    # Closed date is ambiguous -> masked
    payload = {**BASE_PAYLOAD, "closed_date": ambiguous_time}
    res = format_retrieved_case(payload, similarity_score=0.9, intake_time_str="2024-11-03T05:00:00")
    assert res is not None
    assert res["is_historical_evidence"] is False
    assert res["status"] is None

    # Intake time is ambiguous -> masked
    payload_clean = {**BASE_PAYLOAD, "closed_date": "2024-11-03T00:00:00"}
    res2 = format_retrieved_case(payload_clean, similarity_score=0.9, intake_time_str=ambiguous_time)
    assert res2 is not None
    assert res2["is_historical_evidence"] is False
    assert res2["status"] is None


def test_allowlist_enforcement_rejects_free_text_or_addresses():
    """Privacy safeguard: free text or street addresses in complaint fields must fail validation."""
    # Free text street address in descriptor
    with pytest.raises(ValidationError):
        SimilarCasesRequest(
            complaint_type="HEAT/HOT WATER",
            location_type="RESIDENTIAL BUILDING",
            descriptor="123 Main Street, Apt 4B",  # Invalid
            intake_time="2024-01-27T18:00:00",
        )

    # Free text coordinates in location_type
    with pytest.raises(ValidationError):
        SimilarCasesRequest(
            complaint_type="HEAT/HOT WATER",
            location_type="Lat: 40.7128, Long: -74.0060",  # Invalid
            descriptor="APARTMENT ONLY",
            intake_time="2024-01-27T18:00:00",
        )

    # Non-allowlisted borough
    with pytest.raises(ValidationError):
        SimilarCasesRequest(
            complaint_type="HEAT/HOT WATER",
            location_type="RESIDENTIAL BUILDING",
            descriptor="APARTMENT ONLY",
            borough="BOSTON",  # Invalid
            intake_time="2024-01-27T18:00:00",
        )


def test_structured_embedding_text_builder():
    """Verify deterministic embedding text construction matches Condition B format."""
    text1 = build_embedding_text("HEAT/HOT WATER", "RESIDENTIAL BUILDING", "APARTMENT ONLY", "NO HEAT")
    assert text1 == "HEAT/HOT WATER, RESIDENTIAL BUILDING, APARTMENT ONLY, NO HEAT"

    text2 = build_embedding_text("HEAT/HOT WATER", "RESIDENTIAL BUILDING", "ENTIRE BUILDING", None)
    assert text2 == "HEAT/HOT WATER, RESIDENTIAL BUILDING, ENTIRE BUILDING"


def test_retrieve_similar_cases_hermetic_mocked_qdrant():
    """Hermetic retrieval test using mocked QdrantClient and mocked embedder (zero network)."""
    mock_client = MagicMock()
    mock_point = MagicMock()
    mock_point.score = 0.9123
    mock_point.payload = {
        **BASE_PAYLOAD,
        "closed_date": "2024-01-27T16:00:00",
    }
    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]
    mock_client.query_points.return_value = mock_search_result

    req = SimilarCasesRequest(
        complaint_type="HEAT/HOT WATER",
        location_type="RESIDENTIAL BUILDING",
        descriptor="APARTMENT ONLY",
        descriptor_2="NO HEAT",
        borough="MANHATTAN",
        intake_time="2024-01-27T18:00:00",
        limit=5,
    )

    fake_vector = [0.1] * 3072
    mock_embedder = MagicMock(return_value=fake_vector)

    resp: SimilarCasesResponse = retrieve_similar_cases(
        request=req,
        query_vector=None,
        qdrant_client=mock_client,
        embedder=mock_embedder,
    )

    # Verify embedder was invoked with safe text
    mock_embedder.assert_called_once_with("HEAT/HOT WATER, RESIDENTIAL BUILDING, APARTMENT ONLY, NO HEAT")

    # Verify Qdrant was queried with borough filter
    mock_client.query_points.assert_called_once()
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "resolveiq-cases-collection-01"
    assert call_kwargs["limit"] == 5
    assert call_kwargs["query_filter"] is not None

    # Verify response wrapping and temporal outcome exposure
    assert len(resp.cases) == 1
    case = resp.cases[0]
    assert case.ticket == "60428192"
    assert case.is_historical_evidence is True
    assert case.status == "Closed"
    assert case.closed_date == "2024-01-27T16:00:00"
    assert case.resolution_description is not None
