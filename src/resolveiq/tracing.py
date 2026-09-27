"""Opt-in, privacy-redacted LangSmith tracing for the triage workflow."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from langsmith import Client, tracing_context
from langsmith.run_trees import RunTree

from resolveiq.config import get_settings

LOGGER = logging.getLogger(__name__)

_ROUTING_DECISIONS = {
    "scored_only",
    "scored_and_retrieved_precedents",
    "scored_retrieval_unavailable",
    "scored_retrieval_failed",
}
_RETRIEVAL_STATUSES = {"not_requested", "succeeded", "unavailable", "failed"}
_CANONICAL_SHAPES = {
    "APT_NO_HEAT",
    "APT_BOTH",
    "APT_NO_HOT",
    "BLDG_NO_HEAT",
    "BLDG_BOTH",
    "BLDG_NO_HOT",
}


def safe_trace_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """Build trace metadata without copying request or workflow state fields."""
    return {
        "workflow": "resolveiq_triage",
        "workflow_version": "0.1.0",
        "retrieval_requested": bool(state.get("include_similar_cases", False)),
        "trace_schema": "safe_operational_metadata_v1",
    }


def safe_trace_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return only allowlisted operational outcomes, never the raw graph result."""
    cases = result.get("similar_cases")
    if not isinstance(cases, list):
        cases = []

    shape_counts = Counter()
    for case in cases:
        shape = case.get("shape_slug") if isinstance(case, dict) else None
        if isinstance(shape, str) and shape in _CANONICAL_SHAPES:
            shape_counts[shape] += 1
    routing = result.get("routing_decision")
    retrieval = result.get("retrieval_status")
    above_cutoff = result.get("above_frozen_cutoff")

    return {
        "above_frozen_cutoff": above_cutoff if isinstance(above_cutoff, bool) else None,
        "routing_decision": routing if isinstance(routing, str) and routing in _ROUTING_DECISIONS else "other",
        "retrieval_status": retrieval if isinstance(retrieval, str) and retrieval in _RETRIEVAL_STATUSES else "other",
        "similar_case_count": len(cases),
        "retrieved_shape_counts": dict(sorted(shape_counts.items())),
    }


@dataclass
class TraceSession:
    """Holds trace identifiers and writes only a sanitized post-run summary."""

    run_id: UUID | None = None
    trace_client: Client | None = None
    summary_client: Client | None = None
    project_name: str | None = None
    pending_summary: dict[str, Any] | None = None

    @property
    def run_config(self) -> dict[str, UUID] | None:
        """Config that gives LangGraph a known root run ID for the safe update."""
        return {"run_id": self.run_id} if self.run_id is not None else None

    def record_result(self, result: dict[str, Any]) -> None:
        """Keep only an allowlisted summary for persistence after the trace closes."""
        self.pending_summary = safe_trace_summary(result)

    def persist_summary(self) -> None:
        """Post an allowlisted child span after the graph root closes."""
        if (
            self.run_id is None
            or self.trace_client is None
            or self.summary_client is None
            or self.pending_summary is None
        ):
            return
        try:
            # Drain graph spans and read the completed root before attaching the
            # independently sanitized child run.
            self.trace_client.flush(timeout=3.0)
        except Exception as exc:  # noqa: BLE001 - a queue conflict must not skip the summary
            LOGGER.debug("LangSmith trace flush reported %s; continuing with the safe summary", type(exc).__name__)
        try:
            parent_run = self.trace_client.read_run(self.run_id)
        except Exception as exc:  # noqa: BLE001 - observability must fail open
            LOGGER.debug("Safe LangSmith summary parent lookup failed: %s", type(exc).__name__)
            return
        if not parent_run.dotted_order:
            LOGGER.debug("Safe LangSmith summary skipped: root dotted order unavailable")
            return
        try:
            summary_run = RunTree(
                name="resolveiq_safe_summary",
                run_type="chain",
                inputs={},
                outputs=self.pending_summary,
                parent_run_id=self.run_id,
                parent_dotted_order=parent_run.dotted_order,
                trace_id=self.run_id,
                project_name=self.project_name,
                tags=["resolveiq", "triage", "safe-summary"],
                ls_client=self.summary_client,
            )
            summary_run.end(outputs=self.pending_summary)
            summary_run.post()
            self.summary_client.flush(timeout=3.0)
        except Exception as exc:  # noqa: BLE001 - observability must fail open
            LOGGER.debug("Safe LangSmith summary update failed: %s", type(exc).__name__)


@contextmanager
def triage_trace(state: dict[str, Any]) -> Iterator[TraceSession]:
    """Trace one workflow invocation only when explicitly configured."""
    try:
        settings = get_settings()
        api_key = settings.langsmith_api_key
    except Exception as exc:  # noqa: BLE001 - optional tracing must fail open
        LOGGER.debug("LangSmith tracing configuration unavailable: %s", type(exc).__name__)
        yield TraceSession()
        return

    if not settings.langsmith_tracing or api_key is None:
        yield TraceSession()
        return

    try:
        client = Client(
            api_url=settings.langsmith_endpoint,
            api_key=api_key.get_secret_value(),
            hide_inputs=True,
            hide_outputs=True,
        )
        # This client is used only to patch the allowlisted summary after the graph
        # run has been captured with outputs hidden.
        summary_client = Client(
            api_url=settings.langsmith_endpoint,
            api_key=api_key.get_secret_value(),
            hide_inputs=True,
            hide_outputs=False,
        )
        trace_context = tracing_context(
            client=client,
            enabled=True,
            project_name=settings.langsmith_project,
            tags=["resolveiq", "triage"],
            metadata=safe_trace_metadata(state),
        )
        trace_context.__enter__()
    except Exception as exc:  # noqa: BLE001 - tracing must never break triage
        LOGGER.warning("LangSmith tracing setup failed; continuing without tracing: %s", type(exc).__name__)
        yield TraceSession()
        return

    session = TraceSession(
        run_id=uuid4(),
        trace_client=client,
        summary_client=summary_client,
        project_name=settings.langsmith_project,
    )
    try:
        yield session
    except BaseException:
        try:
            # Do not send application exception details, which may contain input-derived data.
            trace_context.__exit__(None, None, None)
        except Exception as trace_exc:  # noqa: BLE001 - tracing finalization is non-critical
            LOGGER.warning("LangSmith trace finalization failed: %s", type(trace_exc).__name__)
        raise
    else:
        try:
            trace_context.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - tracing finalization is non-critical
            LOGGER.warning("LangSmith trace finalization failed: %s", type(exc).__name__)
        session.persist_summary()
