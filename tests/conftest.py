"""Shared test isolation for external tracing integrations."""

from contextlib import contextmanager

import pytest
from langsmith import tracing_context

from resolveiq.tracing import TraceSession


@pytest.fixture(autouse=True)
def disable_langsmith_for_application_tests(monkeypatch):
    """Keep workflow/API tests offline even when the developer .env enables tracing."""

    @contextmanager
    def no_external_trace(_state):
        with tracing_context(enabled=False):
            yield TraceSession()

    monkeypatch.setattr("resolveiq.workflow.triage_trace", no_external_trace)
