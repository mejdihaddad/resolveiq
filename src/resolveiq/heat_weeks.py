# resolveiq/src/resolveiq/heat_weeks.py
"""The event-week versus quiet-week comparison, computed from the two January artefacts.

This module is the only home for that arithmetic. It reads two files from the
data/labelled folder, refuses to trust them unless their bytes still match the
SHA-256 recorded in their own sidecars, and returns one JSON-ready dict.

Scope, stated because a manager reads the result: this is a historical comparison of
CLOSED tickets. It is not a model, it predicts nothing about new tickets, and it
claims no cause for the difference between the two windows.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from resolveiq.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "labelled"
FEATURES_PATH = DATA_DIR / "heat_jan2024_event_quiet_features_at_intake.json"
LABELS_PATH = DATA_DIR / "heat_jan2024_event_quiet_slow_resolution_labels.json"

EVENT = "event_2024-01-15_22"
QUIET = "quiet_2024-01-24_30"

OUTCOME_COLUMNS = {"status", "closed_date", "resolution_description", "resolution_action_updated_date", "due_date"}


def _read_verified(path: Path):
    """Return (frame, sidecar, provenance), but only if the bytes match the sidecar hash."""
    sidecar = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == sidecar["artifact_sha256"], path.name + ": bytes are not what the sidecar hashed"
    assert path.stat().st_size == int(sidecar["artifact_bytes"]), path.name + ": the size moved"
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    frame["unique_key"] = frame["unique_key"].astype("int64")
    assert frame["unique_key"].is_unique, path.name + ": duplicate unique_key"
    assert len(frame) == int(sidecar["rows"]), path.name + ": row count disagrees with the sidecar"
    provenance = {}
    provenance["file"] = path.relative_to(REPO_ROOT).as_posix()
    provenance["sha256"] = digest
    provenance["bytes"] = int(path.stat().st_size)
    provenance["rows"] = int(sidecar["rows"])
    return frame, sidecar, provenance


def _window_block(window_id: str, readable: str, bounds: dict, arrived: int, closed: int,
                  slow: int, still_open: int) -> dict:
    """Build one window's block, checking that the counts nest before shipping them."""
    first = bounds["created_from"]
    before = bounds["created_before"]
    days = (date.fromisoformat(before) - date.fromisoformat(first)).days
    assert int(bounds["rows_pulled"]) == arrived, window_id + ": sidecar rows_pulled disagrees"
    assert closed <= arrived, window_id + ": more closed tickets than arrived tickets"
    assert slow <= closed, window_id + ": more slow tickets than closed tickets"
    assert still_open == arrived - closed, window_id + ": open count disagrees with the totals"
    assert still_open >= 0, window_id + ": negative open count"
    share_pct = round(100.0 * slow / closed, 1)
    block = {}
    block["window_id"] = window_id
    block["label"] = readable
    block["created_from"] = first
    block["created_before"] = before
    block["days"] = days
    block["arrived"] = arrived
    block["closed"] = closed
    block["still_open_no_label"] = still_open
    block["closed_later_than_project_threshold"] = slow
    block["share_of_closed_left_too_long_pct"] = share_pct
    block["denominator"] = "closed tickets in " + readable.lower() + " (" + format(closed, ",") + ")"
    block["note"] = "the share is of CLOSED tickets; still-open ones are reported separately and counted neither fast nor slow"
    return block


def compare_event_and_quiet() -> dict:
    """Return the January event-week versus quiet-week comparison as a JSON-ready dict."""
    features, features_sidecar, features_src = _read_verified(FEATURES_PATH)
    labels, labels_sidecar, labels_src = _read_verified(LABELS_PATH)
    merged = labels.merge(features, on="unique_key", how="inner", validate="one_to_one")
    assert len(merged) == len(labels), "a labelled ticket is missing from the features file"

    thresholds = merged["label_threshold_hours"].unique()
    assert len(thresholds) == 1, "more than one project threshold in this file"
    project_threshold_hours = float(thresholds[0])
    sidecar_threshold_hours = float(labels_sidecar["label_threshold_hours"])
    assert abs(project_threshold_hours - sidecar_threshold_hours) < 1e-9, "threshold disagrees with the sidecar"

    leaked = OUTCOME_COLUMNS.intersection(features.columns)
    assert not leaked, "the intake file carries outcome columns: " + str(sorted(leaked))

    arrived = features.groupby("window")["unique_key"].count()
    closed = merged.groupby("window")["unique_key"].count()
    slow = merged.groupby("window")["slow_resolution"].sum()
    assert set(arrived.index) == {EVENT, QUIET}, "unexpected window labels in the features file"

    fw = features_sidecar["windows"]
    event_arrived = int(arrived[EVENT])
    event_closed = int(closed[EVENT])
    event_slow = int(slow[EVENT])
    event_open = event_arrived - event_closed
    quiet_arrived = int(arrived[QUIET])
    quiet_closed = int(closed[QUIET])
    quiet_slow = int(slow[QUIET])
    quiet_open = quiet_arrived - quiet_closed
    event_block = _window_block(EVENT, "Event week", fw[EVENT], event_arrived, event_closed, event_slow, event_open)
    quiet_block = _window_block(QUIET, "Quiet week", fw[QUIET], quiet_arrived, quiet_closed, quiet_slow, quiet_open)

    arrived_total = event_arrived + quiet_arrived
    closed_total = event_closed + quiet_closed
    slow_total = event_slow + quiet_slow
    open_total = arrived_total - closed_total
    pooled_share_pct = round(100.0 * slow_total / closed_total, 1)

    assert arrived_total == len(features), "arrived total disagrees with the features row count"
    assert closed_total == len(labels), "closed total disagrees with the labels row count"
    assert arrived_total == int(features_sidecar["rows"]), "arrived total disagrees with the sidecar"
    assert closed_total == int(labels_sidecar["rows"]), "closed total disagrees with the sidecar"
    assert open_total == int(labels_sidecar["open_tickets"]["count"]), "open count disagrees with the sidecar"

    recorded = labels_sidecar["slow_by_window_pct"]
    assert abs(event_block["share_of_closed_left_too_long_pct"] - float(recorded["event"])) < 0.05, "event-week share vs sidecar"
    assert abs(quiet_block["share_of_closed_left_too_long_pct"] - float(recorded["quiet"])) < 0.05, "quiet-week share vs sidecar"
    assert abs(pooled_share_pct - float(labels_sidecar["slow_overall_pct"])) < 0.05, "pooled share vs sidecar"

    pooled = {}
    pooled["arrived"] = arrived_total
    pooled["closed"] = closed_total
    pooled["still_open_no_label"] = open_total
    pooled["closed_later_than_project_threshold"] = slow_total
    pooled["share_of_closed_left_too_long_pct"] = pooled_share_pct
    pooled["method"] = "sum(slow) / sum(closed) over both windows - not the mean of the two window shares"
    pooled["denominator"] = "closed tickets in both windows (" + format(closed_total, ",") + ")"

    by_window = {"event_week": event_block["still_open_no_label"], "quiet_week": quiet_block["still_open_no_label"]}
    open_tickets = {"count": open_total, "by_window": by_window, "rule": labels_sidecar["open_tickets"]["label"]}

    cautions = []
    cautions.append("the event week is defined by heat intake volume in these dates; no weather field exists in these artefacts, so nothing here attributes the change to weather")
    cautions.append("the quiet-week share is not independent evidence: the project threshold was fitted on that window's own closed tickets")
    cautions.append("these are shares of CLOSED tickets; the still-open ones are absent from every denominator")
    cautions.append("this report describes two January windows and answers no question about future tickets")
    assert len(cautions) == 4, "a caution line is missing"

    threshold = {}
    threshold["value_hours"] = project_threshold_hours
    threshold["displayed"] = str(round(project_threshold_hours, 1)) + " hours"
    threshold["status"] = "project threshold - not an SLA and not a published service standard"
    threshold["source"] = labels_sidecar["threshold_source"]
    threshold["fitted_on"] = labels_sidecar["threshold_fitted_on"]
    threshold["label_rule"] = labels_sidecar["label_rule"]

    sources = {"features_at_intake": features_src, "slow_resolution_labels": labels_src}
    windows = {"event_week": event_block, "quiet_week": quiet_block}

    report = {}
    report["report"] = "heat_jan2024_event_week_vs_quiet_week"
    report["kind"] = "historical comparison of closed tickets - not a model and not a prediction"
    report["population"] = features_sidecar["complaint_filter"]
    report["project_threshold"] = threshold
    report["windows"] = windows
    report["pooled"] = pooled
    report["open_tickets"] = open_tickets
    report["cautions"] = cautions
    report["sidecar_known_limits"] = labels_sidecar["known_limits"]
    report["sources"] = sources
    return report
