# resolveiq/tests/test_workflow.py
"""Hermetic unit tests for the LangGraph triage workflow.

100% offline: tests graph state transitions, parity against frozen Model 1,
bypassed retrieval on the score-only path, and temporal masking preservation.
"""

from unittest.mock import MagicMock, patch

import pytest

from resolveiq.cases import RetrievalUnavailableError, RetrievedCaseItem, SimilarCasesResponse
from resolveiq.risk_model import IntakeFeaturesRequest, IntakeRiskModel
from resolveiq.workflow import TriageGraphState, run_triage_workflow

SAMPLE_INTAKE: TriageGraphState = {
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
    "intake_time": "2024-01-31T12:00:00",
    "limit": 2,
}


def test_workflow_scoring_only_parity_with_direct_model():
    """Verify exact parity between the workflow's score-only branch and direct model scoring."""
    input_state: TriageGraphState = {
        **SAMPLE_INTAKE,
        "include_similar_cases": False,
    }

    # 1. Run through LangGraph workflow
    graph_res = run_triage_workflow(input_state)

    # 2. Run directly through standalone IntakeRiskModel
    model = IntakeRiskModel.load()
    features_req = IntakeFeaturesRequest(
        descriptor=input_state["descriptor"],
        descriptor_2=input_state["descriptor_2"],
        location_type=input_state["location_type"],
        open_data_channel_type=input_state["open_data_channel_type"],
        agency=input_state["agency"],
        created_hour=input_state["created_hour"],
        created_day_of_week=input_state["created_day_of_week"],
        created_is_weekend=input_state["created_is_weekend"],
    )
    direct_res = model.score_single(features_req)

    # 3. Assert exact parity
    assert graph_res["risk_score"] == pytest.approx(direct_res.risk_score, rel=1e-6)
    assert graph_res["operating_cutoff"] == direct_res.operating_cutoff
    assert graph_res["above_frozen_cutoff"] == direct_res.above_frozen_cutoff
    assert graph_res["disclaimer"] == direct_res.disclaimer
    assert graph_res["routing_decision"] == "scored_only"
    assert "similar_cases" not in graph_res or graph_res.get("similar_cases") is None
    assert graph_res["retrieval_status"] == "not_requested"


@patch("resolveiq.workflow.retrieve_similar_cases")
def test_workflow_retrieval_unavailable_preserves_score(mock_retrieve: MagicMock):
    """Verify optional retrieval failure preserves the successful risk score."""
    mock_retrieve.side_effect = RetrievalUnavailableError("dependency unavailable")

    graph_res = run_triage_workflow({**SAMPLE_INTAKE, "include_similar_cases": True})

    assert isinstance(graph_res["risk_score"], float)
    assert graph_res["routing_decision"] == "scored_retrieval_unavailable"
    assert graph_res["retrieval_status"] == "unavailable"
    assert graph_res["similar_cases"] is None
    assert "risk score is still available" in graph_res["retrieval_message"]


@patch("resolveiq.workflow.retrieve_similar_cases")
def test_workflow_retrieval_skipped_does_not_invoke_cases_module(mock_retrieve: MagicMock):
    """Prove that when include_similar_cases=False, the retrieval module is completely bypassed."""
    input_state: TriageGraphState = {
        **SAMPLE_INTAKE,
        "include_similar_cases": False,
    }

    graph_res = run_triage_workflow(input_state)

    # The router must bypass retrieve_precedents node entirely
    mock_retrieve.assert_not_called()
    assert graph_res["routing_decision"] == "scored_only"
    assert "similar_cases" not in graph_res or graph_res.get("similar_cases") is None


@patch("resolveiq.workflow.retrieve_similar_cases")
def test_workflow_precedents_branch_with_temporal_masking(mock_retrieve: MagicMock):
    """Prove that when include_similar_cases=True, precedents are retrieved and masking is preserved."""
    # Mock return from retrieve_similar_cases
    mock_case_1 = RetrievedCaseItem(
        ticket="60159743",
        shape_id=1,
        shape_slug="APT_NO_HEAT",
        complaint_type="HEAT/HOT WATER",
        location_type="RESIDENTIAL BUILDING",
        descriptor="APARTMENT ONLY",
        descriptor_2="NO HEAT",
        borough="BROOKLYN",
        created_date="2024-01-27T18:22:35",
        similarity_score=0.98,
        is_historical_evidence=True,
        status="Closed",
        closed_date="2024-01-29T20:09:09",
        resolution_description="The Department of HPD inspected the complaint...",
    )
    mock_case_2 = RetrievedCaseItem(
        ticket="60158706",
        shape_id=1,
        shape_slug="APT_NO_HEAT",
        complaint_type="HEAT/HOT WATER",
        location_type="RESIDENTIAL BUILDING",
        descriptor="APARTMENT ONLY",
        descriptor_2="NO HEAT",
        borough="BROOKLYN",
        created_date="2024-01-27T14:48:34",
        similarity_score=0.95,
        is_historical_evidence=False,
        status=None,  # Masked because closed_date was not strictly before intake_time
        closed_date=None,
        resolution_description=None,
    )
    mock_retrieve.return_value = SimilarCasesResponse(
        collection="resolveiq-cases-collection-01",
        borough_filter="BROOKLYN",
        intake_time="2024-01-31T12:00:00",
        cases=[mock_case_1, mock_case_2],
    )

    input_state: TriageGraphState = {
        **SAMPLE_INTAKE,
        "include_similar_cases": True,
    }

    graph_res = run_triage_workflow(input_state)

    # 1. Assert retrieval was invoked once with correct arguments
    mock_retrieve.assert_called_once()
    called_arg = mock_retrieve.call_args[0][0]
    assert called_arg.borough == "BROOKLYN"
    assert called_arg.complaint_type == "HEAT/HOT WATER"
    assert called_arg.intake_time == "2024-01-31T12:00:00"
    assert called_arg.limit == 2

    # 2. Assert routing decision and scoring outputs
    assert graph_res["routing_decision"] == "scored_and_retrieved_precedents"
    assert "risk_score" in graph_res
    assert graph_res["risk_score"] is not None

    # 3. Assert similar cases structure and outcome masking
    similar = graph_res.get("similar_cases")
    assert isinstance(similar, list)
    assert len(similar) == 2

    # Case 1: unmasked historical evidence
    assert similar[0]["ticket"] == "60159743"
    assert similar[0]["is_historical_evidence"] is True
    assert similar[0]["status"] == "Closed"
    assert similar[0]["closed_date"] == "2024-01-29T20:09:09"

    # Case 2: strictly masked outcomes
    assert similar[1]["ticket"] == "60158706"
    assert similar[1]["is_historical_evidence"] is False
    assert similar[1]["status"] is None
    assert similar[1]["closed_date"] is None
    assert similar[1]["resolution_description"] is None


def test_workflow_fairness_boundary_borough_not_in_features():
    """Verify that providing borough in state does not contaminate IntakeFeaturesRequest."""
    with patch("resolveiq.workflow.IntakeRiskModel.load") as mock_load:
        mock_model_instance = MagicMock()
        mock_load.return_value = mock_model_instance
        mock_model_instance.score_single.return_value = MagicMock(
            risk_score=0.15,
            operating_cutoff=0.28,
            above_frozen_cutoff=False,
            disclaimer="Disclaimer text",
        )

        input_state: TriageGraphState = {
            **SAMPLE_INTAKE,
            "borough": "QUEENS",
            "include_similar_cases": False,
        }

        run_triage_workflow(input_state)

        # Inspect the features passed into score_single
        mock_model_instance.score_single.assert_called_once()
        features_passed = mock_model_instance.score_single.call_args[0][0]

        # IntakeFeaturesRequest does not have borough field
        assert not hasattr(features_passed, "borough")
        assert "borough" not in features_passed.model_dump()
