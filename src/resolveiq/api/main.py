# resolveiq/src/resolveiq/api/main.py
"""The ResolveIQ HTTP service.

Provides health checks, historical heat complaint analysis, real-time intake
scoring via frozen Model 1, and privacy-safe similar case retrieval.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from resolveiq.cases import (
    SimilarCasesRequest,
    SimilarCasesResponse,
    retrieve_similar_cases,
)
from resolveiq.heat_weeks import compare_event_and_quiet
from resolveiq.risk_model import (
    IntakeFeaturesRequest,
    IntakeRiskModel,
    IntakeRiskScoreResponse,
)
from resolveiq.workflow import (
    TriageGraphState,
    TriageWorkflowRequest,
    TriageWorkflowResponse,
    run_triage_workflow,
)


def _get_model(app: FastAPI) -> IntakeRiskModel:
    """Retrieve or initialize the cached frozen risk model on application state."""
    if not hasattr(app.state, "risk_model"):
        app.state.risk_model = IntakeRiskModel.load()
    return app.state.risk_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle manager: load and verify the frozen risk model once at startup."""
    _get_model(app)
    yield


app = FastAPI(
    title="ResolveIQ API",
    version="0.1.0",
    summary="NYC 311 Heat & Hot Water Triage and Risk Scoring Service",
    description=(
        "ResolveIQ API provides historical complaint triage analytics, real-time "
        "slow-resolution risk scoring for incoming 311 heat/hot-water service requests "
        "using frozen Model 1, and privacy-safe similar case retrieval from Qdrant."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["service"])
def health() -> dict:
    """Is the process alive and can it answer at all?"""
    return {"status": "ok", "service": "resolveiq-api", "version": app.version}


@app.get("/api/heat-weeks", tags=["reports"])
def heat_weeks_report() -> dict:
    """Event week versus quiet week: arrivals, closures, still-open, and slow shares."""
    return compare_event_and_quiet()


@app.post(
    "/api/score-intake",
    response_model=IntakeRiskScoreResponse,
    tags=["scoring"],
    summary="Score an intake request for slow-resolution risk",
    description=(
        "Scores an intake request using the frozen tabular baseline Model 1 "
        "(Logistic Regression without borough). "
        "Returns an uncalibrated ranking score and an indicator if score >= 0.28. "
        "Probability calibration was not established; the 0.28 cutoff flagged 41.4% "
        "of intake during the final test period."
    ),
)
def score_intake(
    payload: IntakeFeaturesRequest,
    request: Request,
) -> IntakeRiskScoreResponse:
    """Score an intake request using the pre-loaded frozen model from application state."""
    model = _get_model(request.app)
    return model.score_single(payload)


@app.post(
    "/api/cases/similar",
    response_model=SimilarCasesResponse,
    tags=["cases"],
    summary="Retrieve similar historical heat/hot-water complaints",
    description=(
        "Retrieves similar cases from resolveiq-cases-collection-01 using structured "
        "complaint category fields. Admissibility requires created_date < intake_time. "
        "Outcome details are exposed only when closed_date < intake_time."
    ),
)
def find_similar_cases(payload: SimilarCasesRequest) -> SimilarCasesResponse:
    """Retrieve similar cases matching the validated structured intake payload."""
    return retrieve_similar_cases(payload)


@app.post(
    "/api/triage",
    response_model=TriageWorkflowResponse,
    tags=["triage"],
    summary="Execute LangGraph triage workflow (scoring + optional precedent retrieval)",
    description=(
        "Executes the deterministic LangGraph triage workflow. Always scores slow-resolution risk "
        "via frozen Model 1. If include_similar_cases is True, conditionally routes to Qdrant "
        "collection-01 for historical precedents under strict temporal admissibility and outcome masking. "
        "If False, retrieval is bypassed (similar_cases returns null)."
    ),
)
def triage_complaint(payload: TriageWorkflowRequest) -> TriageWorkflowResponse:
    """Triage an incoming complaint via the deterministic LangGraph workflow."""
    state_input: TriageGraphState = {
        "descriptor": payload.descriptor,
        "descriptor_2": payload.descriptor_2,
        "location_type": payload.location_type,
        "open_data_channel_type": payload.open_data_channel_type,
        "agency": payload.agency,
        "created_hour": payload.created_hour,
        "created_day_of_week": payload.created_day_of_week,
        "created_is_weekend": payload.created_is_weekend,
        "complaint_type": payload.complaint_type,
        "borough": payload.borough,
        "intake_time": payload.intake_time,
        "limit": payload.limit,
        "include_similar_cases": payload.include_similar_cases,
    }
    result_state = run_triage_workflow(state_input)
    return TriageWorkflowResponse(
        risk_score=result_state["risk_score"],
        operating_cutoff=result_state["operating_cutoff"],
        above_frozen_cutoff=result_state["above_frozen_cutoff"],
        disclaimer=result_state["disclaimer"],
        routing_decision=result_state["routing_decision"],
        similar_cases=result_state.get("similar_cases"),
        retrieval_status=result_state["retrieval_status"],
        retrieval_message=result_state.get("retrieval_message"),
    )