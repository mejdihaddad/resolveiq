# resolveiq/tests/test_heat_weeks.py
"""Tests for resolveiq.heat_weeks calculation logic and sidecar validation."""

import hashlib
import json
from pathlib import Path

import pytest

from resolveiq.heat_weeks import _read_verified, _window_block, compare_event_and_quiet


def test_compare_event_and_quiet_report_structure_and_values():
    """Verify that compare_event_and_quiet returns verified numbers matching the project baseline."""
    report = compare_event_and_quiet()

    # Top-level keys
    expected_keys = {
        "report",
        "kind",
        "population",
        "project_threshold",
        "windows",
        "pooled",
        "open_tickets",
        "cautions",
        "sidecar_known_limits",
        "sources",
    }
    assert expected_keys.issubset(report.keys())
    assert report["report"] == "heat_jan2024_event_week_vs_quiet_week"

    # Event week numbers
    event = report["windows"]["event_week"]
    assert event["arrived"] == 22811
    assert event["closed"] == 22770
    assert event["still_open_no_label"] == 41
    assert event["closed_later_than_project_threshold"] == 6526
    assert event["share_of_closed_left_too_long_pct"] == 28.7

    # Quiet week numbers
    quiet = report["windows"]["quiet_week"]
    assert quiet["arrived"] == 7315
    assert quiet["closed"] == 7309
    assert quiet["still_open_no_label"] == 6
    assert quiet["closed_later_than_project_threshold"] == 731
    assert quiet["share_of_closed_left_too_long_pct"] == 10.0

    # Pooled numbers
    pooled = report["pooled"]
    assert pooled["arrived"] == 30126
    assert pooled["closed"] == 30079
    assert pooled["still_open_no_label"] == 47
    assert pooled["closed_later_than_project_threshold"] == 7257
    assert pooled["share_of_closed_left_too_long_pct"] == 24.1

    # Project threshold
    threshold = report["project_threshold"]
    assert abs(threshold["value_hours"] - 62.34933333333333) < 1e-9
    assert threshold["displayed"] == "62.3 hours"

    # Cautions and sources
    assert len(report["cautions"]) == 4
    assert "features_at_intake" in report["sources"]
    assert "slow_resolution_labels" in report["sources"]


def test_read_verified_tampered_hash_fails(tmp_path: Path):
    """Verify that _read_verified rejects a file if its SHA-256 hash does not match its sidecar."""
    dummy_data = [{"unique_key": 1, "borough": "MANHATTAN"}]
    raw_bytes = json.dumps(dummy_data).encode("utf-8")

    data_file = tmp_path / "test_data.json"
    data_file.write_bytes(raw_bytes)

    # Sidecar with a bogus SHA-256 hash
    sidecar_data = {
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "artifact_bytes": len(raw_bytes),
        "rows": 1,
    }
    sidecar_file = tmp_path / "test_data.meta.json"
    sidecar_file.write_text(json.dumps(sidecar_data), encoding="utf-8")

    with pytest.raises(AssertionError, match="bytes are not what the sidecar hashed"):
        _read_verified(data_file)


def test_read_verified_tampered_size_fails(tmp_path: Path):
    """Verify that _read_verified rejects a file if its byte size does not match its sidecar."""
    dummy_data = [{"unique_key": 1, "borough": "MANHATTAN"}]
    raw_bytes = json.dumps(dummy_data).encode("utf-8")

    data_file = tmp_path / "test_data.json"
    data_file.write_bytes(raw_bytes)

    # Sidecar with matching hash but wrong byte count
    real_digest = hashlib.sha256(raw_bytes).hexdigest()
    sidecar_data = {
        "artifact_sha256": real_digest,
        "artifact_bytes": len(raw_bytes) + 100,  # Incorrect size
        "rows": 1,
    }
    sidecar_file = tmp_path / "test_data.meta.json"
    sidecar_file.write_text(json.dumps(sidecar_data), encoding="utf-8")

    with pytest.raises(AssertionError, match="the size moved"):
        _read_verified(data_file)


def test_window_block_nesting_assertions():
    """Verify that _window_block enforces count nesting (closed <= arrived, slow <= closed, etc.)."""
    bounds = {
        "created_from": "2024-01-15",
        "created_before": "2024-01-23",
        "rows_pulled": 100,
    }

    # Case 1: closed > arrived
    with pytest.raises(AssertionError, match="more closed tickets than arrived tickets"):
        _window_block(
            window_id="test",
            readable="Test Window",
            bounds=bounds,
            arrived=100,
            closed=105,  # Invalid: closed > arrived
            slow=10,
            still_open=-5,
        )

    # Case 2: slow > closed
    with pytest.raises(AssertionError, match="more slow tickets than closed tickets"):
        _window_block(
            window_id="test",
            readable="Test Window",
            bounds=bounds,
            arrived=100,
            closed=90,
            slow=95,  # Invalid: slow > closed
            still_open=10,
        )

    # Case 3: still_open does not match arrived - closed
    with pytest.raises(AssertionError, match="open count disagrees with the totals"):
        _window_block(
            window_id="test",
            readable="Test Window",
            bounds=bounds,
            arrived=100,
            closed=90,
            slow=20,
            still_open=5,  # Invalid: 100 - 90 != 5
        )
