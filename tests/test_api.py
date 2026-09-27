# resolveiq/tests/test_api.py
"""Tests for resolveiq.api.main routes (/health, /api/heat-weeks, /api/score-intake, /api/cases/similar)."""

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from resolveiq.cases import RetrievalUnavailableError
from resolveiq.api.main import app
from resolveiq.config import REPO_ROOT

client = TestClient(app)
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "val_reference_sample.json"


def test_health_route():
    """Verify that GET /health returns 200 and expected service status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "resolveiq-api"
    assert data["version"] == "0.1.0"


def test_heat_weeks_route():
    """Verify that GET /api/heat-weeks returns 200 and matches the verified historical report."""
    response = client.get("/api/heat-weeks")
    assert response.status_code == 200
    data = response.json()

    assert data["report"] == "heat_jan2024_event_week_vs_quiet_week"
    assert data["windows"]["event_week"]["arrived"] == 22811
    assert data["windows"]["event_week"]["closed"] == 22770
    assert data["windows"]["event_week"]["share_of_closed_left_too_long_pct"] == 28.7

    assert data["windows"]["quiet_week"]["arrived"] == 7315
    assert data["windows"]["quiet_week"]["closed"] == 7309
    assert data["windows"]["quiet_week"]["share_of_closed_left_too_long_pct"] == 10.0

    assert data["pooled"]["arrived"] == 30126
    assert data["pooled"]["closed"] == 30079
    assert data["pooled"]["share_of_closed_left_too_long_pct"] == 24.1

    assert data["project_threshold"]["displayed"] == "62.3 hours"


def test_score_intake_route_success():
    """Verify POST /api/score-intake returns 200 with valid synthetic schema."""
    payload = {
        "descriptor": "APARTMENT ONLY",
        "descriptor_2": "NO HEAT",
        "location_type": "RESIDENTIAL BUILDING",
        "open_data_channel_type": "ONLINE",
        "agency": "HPD",
        "created_hour": 14,
        "created_day_of_week": 2,
        "created_is_weekend": 0,
    }
    response = client.post("/api/score-intake", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "frozen_model_1_no_borough"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["operating_cutoff"] == 0.28
    assert isinstance(data["above_frozen_cutoff"], bool)
    assert data["calibrated"] is False
    assert "0.28" in data["disclaimer"]
    assert "41.4%" in data["disclaimer"]


def test_score_intake_matches_validation_fixture():
    """Verify POST /api/score-intake returns exact parity score for reference record."""
    fixture_records = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    first_record = fixture_records[0]

    response = client.post("/api/score-intake", json=first_record["features"])
    assert response.status_code == 200
    data = response.json()

    assert abs(data["risk_score"] - first_record["expected_score"]) < 1e-6
    assert data["above_frozen_cutoff"] == (data["risk_score"] >= 0.28)


def test_score_intake_temporal_inconsistency_rejected():
    """Verify POST /api/score-intake returns 422 for inconsistent weekend/weekday."""
    payload = {
        "created_hour": 10,
        "created_day_of_week": 0,
        "created_is_weekend": 1,
    }
    response = client.post("/api/score-intake", json=payload)
    assert response.status_code == 422


def test_score_intake_out_of_range_rejected():
    """Verify POST /api/score-intake returns 422 for invalid hour or day."""
    payload = {
        "created_hour": 25,
        "created_day_of_week": 2,
        "created_is_weekend": 0,
    }
    response = client.post("/api/score-intake", json=payload)
    assert response.status_code == 422


def test_similar_cases_route_success(monkeypatch):
    """Verify POST /api/cases/similar returns 200 using mocked Qdrant and embedder."""
    mock_client = MagicMock()
    mock_point = MagicMock()
    mock_point.score = 0.8842
    mock_point.payload = {
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
        "closed_date": "2024-01-27T16:00:00",
        "resolution_description": "Inspected and resolved.",
    }
    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]
    mock_client.query_points.return_value = mock_search_result

    monkeypatch.setattr("resolveiq.cases.QdrantClient", lambda **kwargs: mock_client)
    monkeypatch.setattr("resolveiq.cases.embed_text_gemini", lambda text: [0.1] * 3072)

    payload = {
        "complaint_type": "HEAT/HOT WATER",
        "location_type": "RESIDENTIAL BUILDING",
        "descriptor": "APARTMENT ONLY",
        "descriptor_2": "NO HEAT",
        "borough": "MANHATTAN",
        "intake_time": "2024-01-27T18:00:00",
        "limit": 5,
    }
    response = client.post("/api/cases/similar", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["collection"] == "resolveiq-cases-collection-01"
    assert data["borough_filter"] == "MANHATTAN"
    assert len(data["cases"]) == 1
    c = data["cases"][0]
    assert c["ticket"] == "60428192"
    assert c["is_historical_evidence"] is True
    assert c["status"] == "Closed"
    assert c["closed_date"] == "2024-01-27T16:00:00"


def test_similar_cases_route_rejects_free_text_or_address():
    """Verify POST /api/cases/similar returns 422 for arbitrary address or free text in complaint fields."""
    payload = {
        "complaint_type": "HEAT/HOT WATER",
        "location_type": "RESIDENTIAL BUILDING",
        "descriptor": "456 Washington St, Apt 2F",  # Not allowlisted
        "intake_time": "2024-01-27T18:00:00",
    }
    response = client.post("/api/cases/similar", json=payload)
    assert response.status_code == 422


def test_similar_cases_route_rejects_nonexistent_local_timestamp():
    """Verify POST /api/cases/similar returns 422 for nonexistent local intake time."""
    payload = {
        "complaint_type": "HEAT/HOT WATER",
        "location_type": "RESIDENTIAL BUILDING",
        "descriptor": "APARTMENT ONLY",
        "intake_time": "2024-03-10T02:30:00",  # Nonexistent local time in NYC
    }
    response = client.post("/api/cases/similar", json=payload)
    assert response.status_code == 422


# --- Tests for POST /api/triage (LangGraph Workflow Endpoint) ---


def test_triage_route_scoring_only_branch():
    """Verify POST /api/triage with include_similar_cases=False returns scored_only and similar_cases=null."""
    payload = {
        "descriptor": "APARTMENT ONLY",
        "descriptor_2": "NO HEAT",
        "location_type": "RESIDENTIAL BUILDING",
        "open_data_channel_type": "ONLINE",
        "agency": "HPD",
        "created_hour": 14,
        "created_day_of_week": 0,
        "created_is_weekend": 0,
        "include_similar_cases": False,
    }
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["routing_decision"] == "scored_only"
    assert isinstance(data["risk_score"], float)
    assert data["operating_cutoff"] == 0.28
    assert data["above_frozen_cutoff"] is False
    assert data["similar_cases"] is None
    assert data["retrieval_status"] == "not_requested"
    assert data["retrieval_message"] is None
    assert "Score is an uncalibrated ranking score" in data["disclaimer"]


def test_triage_route_retrieval_unavailable_preserves_score(monkeypatch):
    """Verify optional retrieval failure returns the successful risk score."""
    def fail_retrieval(_request):
        raise RetrievalUnavailableError("dependency unavailable")

    monkeypatch.setattr("resolveiq.workflow.retrieve_similar_cases", fail_retrieval)

    payload = {
        "descriptor": "APARTMENT ONLY",
        "descriptor_2": "NO HEAT",
        "location_type": "RESIDENTIAL BUILDING",
        "open_data_channel_type": "ONLINE",
        "agency": "HPD",
        "created_hour": 14,
        "created_day_of_week": 0,
        "created_is_weekend": 0,
        "intake_time": "2024-01-27T18:00:00",
        "include_similar_cases": True,
    }
    response = client.post("/api/triage", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["risk_score"], float)
    assert data["routing_decision"] == "scored_retrieval_unavailable"
    assert data["retrieval_status"] == "unavailable"
    assert data["similar_cases"] is None
    assert "risk score is still available" in data["retrieval_message"]


def test_triage_route_precedents_branch_success(monkeypatch):
    """Verify POST /api/triage with include_similar_cases=True runs retrieval with outcome masking."""
    mock_client = MagicMock()
    mock_point = MagicMock()
    mock_point.id = "60159743"
    mock_point.score = 0.99
    mock_point.payload = {
        "unique_key": "60159743",
        "shape_id": 1,
        "shape_slug": "APT_NO_HEAT",
        "complaint_type": "HEAT/HOT WATER",
        "location_type": "RESIDENTIAL BUILDING",
        "descriptor": "APARTMENT ONLY",
        "descriptor_2": "NO HEAT",
        "borough": "BROOKLYN",
        "created_date": "2024-01-27T10:00:00",
        "status": "Closed",
        "closed_date": "2024-01-27T16:00:00",
        "resolution_description": "Inspected and resolved.",
    }
    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]
    mock_client.query_points.return_value = mock_search_result

    monkeypatch.setattr("resolveiq.cases.QdrantClient", lambda **kwargs: mock_client)
    monkeypatch.setattr("resolveiq.cases.embed_text_gemini", lambda text: [0.1] * 3072)

    payload = {
        "descriptor": "APARTMENT ONLY",
        "descriptor_2": "NO HEAT",
        "location_type": "RESIDENTIAL BUILDING",
        "open_data_channel_type": "ONLINE",
        "agency": "HPD",
        "created_hour": 14,
        "created_day_of_week": 0,
        "created_is_weekend": 0,
        "complaint_type": "HEAT/HOT WATER",
        "borough": "BROOKLYN",
        "intake_time": "2024-01-27T18:00:00",
        "limit": 2,
        "include_similar_cases": True,
    }
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["routing_decision"] == "scored_and_retrieved_precedents"
    assert isinstance(data["risk_score"], float)
    assert isinstance(data["similar_cases"], list)
    assert len(data["similar_cases"]) == 1

    case = data["similar_cases"][0]
    assert case["ticket"] == "60159743"
    assert case["is_historical_evidence"] is True
    assert case["status"] == "Closed"
    assert case["closed_date"] == "2024-01-27T16:00:00"


def test_triage_route_precedents_branch_empty_candidates(monkeypatch):
    """Verify executed retrieval with 0 matches returns [] (empty list), distinct from skipped (null)."""
    mock_client = MagicMock()
    mock_search_result = MagicMock()
    mock_search_result.points = []
    mock_client.query_points.return_value = mock_search_result

    monkeypatch.setattr("resolveiq.cases.QdrantClient", lambda **kwargs: mock_client)
    monkeypatch.setattr("resolveiq.cases.embed_text_gemini", lambda text: [0.1] * 3072)

    payload = {
        "descriptor": "APARTMENT ONLY",
        "descriptor_2": "NO HEAT",
        "location_type": "RESIDENTIAL BUILDING",
        "open_data_channel_type": "ONLINE",
        "agency": "HPD",
        "created_hour": 14,
        "created_day_of_week": 0,
        "created_is_weekend": 0,
        "complaint_type": "HEAT/HOT WATER",
        "borough": "BROOKLYN",
        "intake_time": "2024-01-27T18:00:00",
        "limit": 2,
        "include_similar_cases": True,
    }
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Crucial distinction: executed with 0 candidates yields []
    assert data["routing_decision"] == "scored_and_retrieved_precedents"
    assert data["similar_cases"] == []


def test_triage_route_requires_intake_time_when_retrieval_requested():
    """Verify POST /api/triage returns 422 if include_similar_cases=True but intake_time is missing."""
    payload = {
        "descriptor": "APARTMENT ONLY",
        "created_hour": 14,
        "created_day_of_week": 0,
        "created_is_weekend": 0,
        "include_similar_cases": True,
        # intake_time missing
    }
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 422


def test_triage_route_rejects_non_allowlisted_fields():
    """Verify POST /api/triage returns 422 if free text address is passed in descriptor."""
    payload = {
        "descriptor": "123 Main Street, Apt 4B",
        "created_hour": 14,
        "created_day_of_week": 0,
        "created_is_weekend": 0,
        "include_similar_cases": False,
    }
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 422
