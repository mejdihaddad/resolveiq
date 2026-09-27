"""Hermetic tests for opt-in, privacy-redacted LangSmith tracing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langsmith.utils import LangSmithConflictError

from resolveiq.tracing import safe_trace_metadata, safe_trace_summary, triage_trace


def test_trace_metadata_excludes_request_and_result_fields():
    """Trace metadata contains only operational fields, never raw state values."""
    state = {
        "descriptor": "APARTMENT ONLY",
        "borough": "BROOKLYN",
        "intake_time": "2024-01-27T18:00:00",
        "ticket": "60159743",
        "resolution_description": "Sensitive historical detail",
        "include_similar_cases": True,
    }

    metadata = safe_trace_metadata(state)

    assert metadata == {
        "workflow": "resolveiq_triage",
        "workflow_version": "0.1.0",
        "retrieval_requested": True,
        "trace_schema": "safe_operational_metadata_v1",
    }
    assert "BROOKLYN" not in repr(metadata)
    assert "60159743" not in repr(metadata)
    assert "Sensitive historical detail" not in repr(metadata)


def test_safe_trace_summary_contains_only_allowlisted_outcomes():
    """Summary is useful but excludes scores, IDs, timestamps, boroughs, and case text."""
    summary = safe_trace_summary(
        {
            "risk_score": 0.734,
            "above_frozen_cutoff": True,
            "routing_decision": "scored_and_retrieved_precedents",
            "retrieval_status": "succeeded",
            "similar_cases": [
                {
                    "ticket": "PRIVATE-TICKET-ID",
                    "borough": "PRIVATE-BOROUGH",
                    "created_date": "PRIVATE-TIMESTAMP",
                    "shape_slug": "APT_NO_HEAT",
                    "resolution_description": "PRIVATE-RESOLUTION-TEXT",
                },
                {"shape_slug": "APT_NO_HEAT"},
                {"shape_slug": "UNRECOGNIZED-SHAPE"},
            ],
        }
    )

    assert summary == {
        "above_frozen_cutoff": True,
        "routing_decision": "scored_and_retrieved_precedents",
        "retrieval_status": "succeeded",
        "similar_case_count": 3,
        "retrieved_shape_counts": {"APT_NO_HEAT": 2},
    }
    rendered = repr(summary)
    for private_value in (
        "0.734",
        "PRIVATE-TICKET-ID",
        "PRIVATE-BOROUGH",
        "PRIVATE-TIMESTAMP",
        "PRIVATE-RESOLUTION-TEXT",
    ):
        assert private_value not in rendered


def test_tracing_disabled_leaves_client_unused(monkeypatch):
    """Tracing is a no-op unless the explicit ResolveIQ switch and key are set."""
    monkeypatch.setattr(
        "resolveiq.tracing.get_settings",
        lambda: SimpleNamespace(langsmith_tracing=False, langsmith_api_key=None),
    )

    with patch("resolveiq.tracing.Client") as client, triage_trace(
        {"include_similar_cases": True}
    ):
        pass

    client.assert_not_called()


def test_missing_gemini_key_does_not_block_tracing_disabled_workflow(monkeypatch):
    """Tracing configuration failure must not affect score-only workflow execution."""
    def missing_gemini_key():
        raise ValueError("GEMINI_API_KEY is required")

    monkeypatch.setattr("resolveiq.tracing.get_settings", missing_gemini_key)

    with triage_trace({"include_similar_cases": False}):
        workflow_continued = True

    assert workflow_continued is True


def test_enabled_client_redacts_inputs_and_outputs(monkeypatch):
    """Enabled tracing configures SDK redaction before entering a trace."""
    api_key = MagicMock()
    api_key.get_secret_value.return_value = "local-test-key"
    monkeypatch.setattr(
        "resolveiq.tracing.get_settings",
        lambda: SimpleNamespace(
            langsmith_tracing=True,
            langsmith_api_key=api_key,
            langsmith_endpoint="https://api.smith.langchain.com",
            langsmith_project="resolveiq-test",
        ),
    )

    trace_context = MagicMock()
    with patch("resolveiq.tracing.Client") as client, patch(
        "resolveiq.tracing.tracing_context", return_value=trace_context
    ) as context, triage_trace({"include_similar_cases": False}):
        pass

    assert client.call_count == 2
    assert client.call_args_list[0].kwargs == {
        "api_url": "https://api.smith.langchain.com",
        "api_key": "local-test-key",
        "hide_inputs": True,
        "hide_outputs": True,
    }
    assert client.call_args_list[1].kwargs == {
        "api_url": "https://api.smith.langchain.com",
        "api_key": "local-test-key",
        "hide_inputs": True,
        "hide_outputs": False,
    }
    context.assert_called_once()
    assert context.call_args.kwargs["metadata"] == {
        "workflow": "resolveiq_triage",
        "workflow_version": "0.1.0",
        "retrieval_requested": False,
        "trace_schema": "safe_operational_metadata_v1",
    }


def test_safe_summary_is_posted_as_child_of_known_root(monkeypatch):
    """Raw graph outputs stay hidden; the visible child contains only safe summary fields."""
    api_key = MagicMock()
    api_key.get_secret_value.return_value = "local-test-key"
    monkeypatch.setattr(
        "resolveiq.tracing.get_settings",
        lambda: SimpleNamespace(
            langsmith_tracing=True,
            langsmith_api_key=api_key,
            langsmith_endpoint="https://api.smith.langchain.com",
            langsmith_project="resolveiq-test",
        ),
    )

    trace_client = MagicMock()
    trace_client.read_run.return_value = SimpleNamespace(dotted_order="root-order")
    summary_client = MagicMock()
    trace_context = MagicMock()
    summary_run = MagicMock()
    events = []
    summary_run.post.side_effect = lambda: events.append("summary")
    trace_context.__exit__.side_effect = lambda *args: events.append("context_exit")
    with patch("resolveiq.tracing.Client", side_effect=[trace_client, summary_client]), patch(
        "resolveiq.tracing.tracing_context", return_value=trace_context
    ), patch("resolveiq.tracing.RunTree", return_value=summary_run) as run_tree, triage_trace(
        {"include_similar_cases": True}
    ) as trace:
        assert trace.run_config == {"run_id": trace.run_id}
        result = {
            "risk_score": 0.734,
            "above_frozen_cutoff": True,
            "routing_decision": "scored_and_retrieved_precedents",
            "retrieval_status": "succeeded",
            "similar_cases": [
                {
                    "ticket": "PRIVATE-TICKET-ID",
                    "shape_slug": "APT_NO_HEAT",
                    "resolution_description": "PRIVATE-RESOLUTION-TEXT",
                }
            ],
        }
        trace.record_result(result)
        run_tree.assert_not_called()

    trace_client.flush.assert_called_once_with(timeout=3.0)
    trace_client.read_run.assert_called_once_with(trace.run_id)
    safe_output = {
        "above_frozen_cutoff": True,
        "routing_decision": "scored_and_retrieved_precedents",
        "retrieval_status": "succeeded",
        "similar_case_count": 1,
        "retrieved_shape_counts": {"APT_NO_HEAT": 1},
    }
    run_tree.assert_called_once_with(
        name="resolveiq_safe_summary",
        run_type="chain",
        inputs={},
        outputs=safe_output,
        parent_run_id=trace.run_id,
        parent_dotted_order="root-order",
        trace_id=trace.run_id,
        project_name="resolveiq-test",
        tags=["resolveiq", "triage", "safe-summary"],
        ls_client=summary_client,
    )
    summary_run.end.assert_called_once_with(outputs=safe_output)
    summary_run.post.assert_called_once_with()
    summary_client.flush.assert_called_once_with(timeout=3.0)
    assert events == ["context_exit", "summary"]


def test_trace_flush_conflict_does_not_skip_safe_summary(monkeypatch):
    """An optional trace queue conflict must not suppress the sanitized child span."""
    api_key = MagicMock()
    api_key.get_secret_value.return_value = "local-test-key"
    monkeypatch.setattr(
        "resolveiq.tracing.get_settings",
        lambda: SimpleNamespace(
            langsmith_tracing=True,
            langsmith_api_key=api_key,
            langsmith_endpoint="https://api.smith.langchain.com",
            langsmith_project="resolveiq-test",
        ),
    )

    trace_client = MagicMock()
    trace_client.flush.side_effect = LangSmithConflictError("do not expose this message")
    trace_client.read_run.return_value = SimpleNamespace(dotted_order="root-order")
    summary_client = MagicMock()
    trace_context = MagicMock()
    summary_run = MagicMock()
    with patch("resolveiq.tracing.Client", side_effect=[trace_client, summary_client]), patch(
        "resolveiq.tracing.tracing_context", return_value=trace_context
    ), patch("resolveiq.tracing.RunTree", return_value=summary_run), triage_trace(
        {"include_similar_cases": False}
    ) as trace:
        trace.record_result(
            {
                "above_frozen_cutoff": False,
                "routing_decision": "scored_only",
                "retrieval_status": "not_requested",
                "similar_cases": None,
            }
        )

    trace_client.read_run.assert_called_once_with(trace.run_id)
    summary_run.post.assert_called_once_with()


def test_workflow_errors_are_not_forwarded_to_trace(monkeypatch):
    """Application exception details are suppressed before trace finalization."""
    api_key = MagicMock()
    api_key.get_secret_value.return_value = "local-test-key"
    monkeypatch.setattr(
        "resolveiq.tracing.get_settings",
        lambda: SimpleNamespace(
            langsmith_tracing=True,
            langsmith_api_key=api_key,
            langsmith_endpoint="https://api.smith.langchain.com",
            langsmith_project="resolveiq-test",
        ),
    )

    trace_context = MagicMock()
    with patch("resolveiq.tracing.Client"), patch(
        "resolveiq.tracing.tracing_context", return_value=trace_context
    ):
        try:
            with triage_trace({"descriptor": "PRIVATE INPUT"}):
                raise RuntimeError("private case detail")
        except RuntimeError:
            pass

    trace_context.__exit__.assert_called_once_with(None, None, None)
    assert "PRIVATE INPUT" not in repr(trace_context.__exit__.call_args)
    assert "private case detail" not in repr(trace_context.__exit__.call_args)
