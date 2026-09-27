# resolveiq/src/resolveiq/workflow.py
"""LangGraph Triage Workflow.

Orchestrates intake risk scoring and conditional precedent retrieval
into a deterministic, observable computational graph without LLM calls.
Exposes Pydantic request/response schemas for the HTTP service layer.
"""

from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from resolveiq.cases import (
    AllowedBorough,
    AllowedComplaintType,
    AllowedDescriptor,
    AllowedDescriptor2,
    AllowedLocationType,
    RetrievalRequestError,
    RetrievalUnavailableError,
    RetrievedCaseItem,
    SimilarCasesRequest,
    parse_and_normalize_timestamp,
    retrieve_similar_cases,
)
from resolveiq.risk_model import (
    FROZEN_OPERATING_CUTOFF,
    MODEL_DISCLAIMER,
    IntakeFeaturesRequest,
    IntakeRiskModel,
)
from resolveiq.tracing import triage_trace


class TriageGraphState(TypedDict, total=False):
    """The central data contract carried through the triage workflow."""

    # --- Input Intake Features (Used for Tabular Risk Scoring) ---
    descriptor: str | None
    descriptor_2: str | None
    location_type: str | None
    open_data_channel_type: str | None
    agency: str | None
    created_hour: int
    created_day_of_week: int
    created_is_weekend: int

    # --- Precedent Retrieval Fields (Strictly Isolated from Risk Scoring) ---
    complaint_type: str
    borough: str | None  # Fairness boundary: sent only to Qdrant, never to Model 1
    intake_time: str | None  # ISO timestamp for temporal admissibility (created_date < intake_time)
    limit: int

    # --- Caller Control Switch for Deterministic Conditional Routing ---
    include_similar_cases: bool  # If True -> route to Qdrant retrieval; if False -> route to END

    # --- Graph Outputs (Populated progressively by graph nodes) ---
    risk_score: float | None
    operating_cutoff: float | None
    above_frozen_cutoff: bool | None
    disclaimer: str | None
    similar_cases: list[dict[str, Any]] | None
    routing_decision: str | None  # Observable audit record of the conditional edge path taken
    retrieval_status: str
    retrieval_message: str | None


def score_intake_node(state: TriageGraphState) -> dict[str, Any]:
    """Node 1: Score the intake request using frozen Model 1 baseline."""
    # Enforce fairness boundary: borough is strictly omitted from risk features
    features_req = IntakeFeaturesRequest(
        descriptor=state.get("descriptor"),
        descriptor_2=state.get("descriptor_2"),
        location_type=state.get("location_type"),
        open_data_channel_type=state.get("open_data_channel_type"),
        agency=state.get("agency"),
        created_hour=state["created_hour"],
        created_day_of_week=state["created_day_of_week"],
        created_is_weekend=state["created_is_weekend"],
    )

    model = IntakeRiskModel.load()
    score_resp = model.score_single(features_req)

    return {
        "risk_score": score_resp.risk_score,
        "operating_cutoff": score_resp.operating_cutoff,
        "above_frozen_cutoff": score_resp.above_frozen_cutoff,
        "disclaimer": score_resp.disclaimer,
        "routing_decision": "scored_only",  # Default if graph ends here
        "retrieval_status": "not_requested",
        "retrieval_message": None,
    }


def route_after_scoring(state: TriageGraphState) -> str:
    """Conditional Router: Route based on caller's explicit retrieval request."""
    if state.get("include_similar_cases", False):
        return "retrieve_precedents"
    return "end"


def retrieve_precedents_node(state: TriageGraphState) -> dict[str, Any]:
    """Node 2: Retrieve similar historical complaints from Qdrant with temporal guards."""
    cases_req = SimilarCasesRequest(
        complaint_type=state.get("complaint_type", "HEAT/HOT WATER"),
        location_type=state.get("location_type"),
        descriptor=state.get("descriptor"),
        descriptor_2=state.get("descriptor_2"),
        borough=state.get("borough"),  # Borough filter is allowed for precedent search
        intake_time=state.get("intake_time"),
        limit=state.get("limit", 5),
    )

    try:
        cases_resp = retrieve_similar_cases(cases_req)
    except RetrievalUnavailableError:
        return {
            "routing_decision": "scored_retrieval_unavailable",
            "similar_cases": None,
            "retrieval_status": "unavailable",
            "retrieval_message": (
                "Similar-case retrieval is temporarily unavailable; risk score is still available."
            ),
        }
    except RetrievalRequestError:
        return {
            "routing_decision": "scored_retrieval_failed",
            "similar_cases": None,
            "retrieval_status": "failed",
            "retrieval_message": (
                "Similar-case retrieval could not be completed for this request; risk score is still available."
            ),
        }
    cases_dicts = [case.model_dump() for case in cases_resp.cases]

    return {
        "similar_cases": cases_dicts,
        "routing_decision": "scored_and_retrieved_precedents",
        "retrieval_status": "succeeded",
        "retrieval_message": None,
    }


def build_triage_graph() -> CompiledStateGraph:
    """Assemble and compile the deterministic triage workflow graph."""
    workflow = StateGraph(TriageGraphState)

    # Register computation nodes
    workflow.add_node("score_intake", score_intake_node)
    workflow.add_node("retrieve_precedents", retrieve_precedents_node)

    # Wire control flow edges
    workflow.add_edge(START, "score_intake")
    workflow.add_conditional_edges(
        "score_intake",
        route_after_scoring,
        {
            "retrieve_precedents": "retrieve_precedents",
            "end": END,
        },
    )
    workflow.add_edge("retrieve_precedents", END)

    return workflow.compile()


# Module-level compiled instance for reuse
triage_graph = build_triage_graph()


def run_triage_workflow(state: TriageGraphState) -> TriageGraphState:
    """Execute the triage graph workflow on an input state dictionary."""
    with triage_trace(state) as trace:
        result = triage_graph.invoke(state, config=trace.run_config)
        trace.record_result(result)
        return result


# --- HTTP API Request and Response Schemas ---


class TriageWorkflowRequest(BaseModel):
    """Validated structured request schema for the POST /api/triage endpoint."""

    model_config = ConfigDict(extra="forbid")

    # Complaint category fields (allowlisted for privacy & embeddings)
    complaint_type: AllowedComplaintType = Field(
        default="HEAT/HOT WATER",
        description="Controlled complaint type allowlist.",
    )
    location_type: Optional[AllowedLocationType] = Field(
        default="RESIDENTIAL BUILDING",
        description="Controlled location type allowlist.",
    )
    descriptor: AllowedDescriptor = Field(
        ...,
        description="Primary complaint descriptor allowlist.",
    )
    descriptor_2: Optional[AllowedDescriptor2] = Field(
        default=None,
        description="Secondary complaint descriptor allowlist.",
    )
    open_data_channel_type: Optional[str] = Field(
        default=None,
        description="Submission channel (e.g. 'ONLINE', 'PHONE'). Missing values map to 'MISSING'.",
    )
    agency: Optional[str] = Field(
        default="HPD",
        description="Agency code (e.g. 'HPD'). Missing values map to 'MISSING'.",
    )

    # Temporal feature fields (derived from intake wall clock)
    created_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of intake timestamp in NYC local time (0–23).",
    )
    created_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week (0=Monday, 1=Tuesday, ..., 6=Sunday).",
    )
    created_is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
        description="Weekend flag: 1 if Saturday or Sunday, 0 if Monday through Friday.",
    )

    # Precedent retrieval parameters
    borough: Optional[AllowedBorough] = Field(
        default=None,
        description="Optional borough filter matching collection payload. Strictly isolated from Model 1 scoring.",
    )
    intake_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 intake timestamp. Required if include_similar_cases is True.",
        examples=["2024-01-27T18:00:00", "2024-01-27T23:00:00Z"],
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of similar cases to retrieve if requested.",
    )

    # Caller-controlled routing switch
    include_similar_cases: bool = Field(
        default=False,
        description="Deterministic router flag. If True, executes Qdrant precedent retrieval; if False, bypasses retrieval.",
    )

    @model_validator(mode="after")
    def validate_workflow_constraints(self) -> "TriageWorkflowRequest":
        """Enforce day/weekend consistency and require valid intake_time if retrieval requested."""
        expected_weekend = 1 if self.created_day_of_week in (5, 6) else 0
        if self.created_is_weekend != expected_weekend:
            raise ValueError(
                f"Inconsistent weekend flag: created_day_of_week={self.created_day_of_week} "
                f"requires created_is_weekend={expected_weekend}, but got {self.created_is_weekend}."
            )
        if self.include_similar_cases:
            if not self.intake_time:
                raise ValueError("`intake_time` is required when `include_similar_cases` is True.")
            parse_and_normalize_timestamp(self.intake_time)
        return self


class TriageWorkflowResponse(BaseModel):
    """Structured HTTP response schema returned by the POST /api/triage endpoint."""

    model_config = ConfigDict(extra="forbid")

    # Risk scoring outputs (frozen tabular Model 1)
    risk_score: float = Field(..., description="Uncalibrated ranking score from frozen Model 1.")
    operating_cutoff: float = Field(
        default=FROZEN_OPERATING_CUTOFF,
        description="Frozen validation operating cutoff (0.28). Not an operational dispatch decision.",
    )
    above_frozen_cutoff: bool = Field(..., description="True if risk_score >= operating_cutoff (0.28).")
    disclaimer: str = Field(
        default=MODEL_DISCLAIMER,
        description="Uncalibrated ranking score and validation capacity disclaimer.",
    )

    # Observable routing outputs
    routing_decision: str = Field(
        ...,
        description="'scored_only' if retrieval was bypassed, 'scored_and_retrieved_precedents' if executed.",
    )
    similar_cases: Optional[list[RetrievedCaseItem]] = Field(
        default=None,
        description=(
            "Historical precedent cases with outcome masking. "
            "null (None) indicates retrieval was skipped (include_similar_cases=False); "
            "[] indicates retrieval was executed but 0 admissible candidates matched."
        ),
    )
    retrieval_status: Literal["not_requested", "succeeded", "unavailable", "failed"] = Field(
        ..., description="Status of optional similar-case retrieval."
    )
    retrieval_message: Optional[str] = Field(
        default=None,
        description="Safe user-facing message when optional retrieval is unavailable.",
    )
