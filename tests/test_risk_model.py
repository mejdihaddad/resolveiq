# resolveiq/tests/test_risk_model.py
"""Offline tests for intake risk model loading, hash verification, schemas, and parity."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from resolveiq.config import REPO_ROOT
from resolveiq.risk_model import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    FROZEN_OPERATING_CUTOFF,
    IntakeFeaturesRequest,
    IntakeRiskModel,
)

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "val_reference_sample.json"
EXPECTED_SHA256 = "9daffe1971b15fd95c45f9b2fabe5c685bad909d84bfd31459cb811d7ae3b6e2"


def test_model_artifact_hash_integrity():
    """Verify that intake_risk_model_v1.joblib exists and matches expected SHA-256."""
    assert DEFAULT_MODEL_PATH.exists(), f"Model artifact missing: {DEFAULT_MODEL_PATH}"
    assert DEFAULT_META_PATH.exists(), f"Metadata sidecar missing: {DEFAULT_META_PATH}"

    sidecar = json.loads(DEFAULT_META_PATH.read_text(encoding="utf-8"))
    assert sidecar["sha256"] == EXPECTED_SHA256

    actual_hash = hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
    assert actual_hash == EXPECTED_SHA256, (
        f"Model artifact SHA-256 hash changed! Expected {EXPECTED_SHA256}, got {actual_hash}"
    )


def test_startup_fails_on_missing_or_corrupt_artifact(tmp_path: Path):
    """Verify IntakeRiskModel.load raises FileNotFoundError or RuntimeError on missing/corrupted files."""
    missing_joblib = tmp_path / "nonexistent.joblib"
    missing_meta = tmp_path / "nonexistent.meta.json"

    # Missing file
    with pytest.raises(FileNotFoundError):
        IntakeRiskModel.load(model_path=missing_joblib, meta_path=DEFAULT_META_PATH)

    with pytest.raises(FileNotFoundError):
        IntakeRiskModel.load(model_path=DEFAULT_MODEL_PATH, meta_path=missing_meta)

    # Hash mismatch / tampering
    corrupt_meta = tmp_path / "corrupt.meta.json"
    corrupt_meta.write_text(
        json.dumps({"sha256": "0000000000000000000000000000000000000000000000000000000000000000"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Model artifact SHA-256 checksum mismatch"):
        IntakeRiskModel.load(model_path=DEFAULT_MODEL_PATH, meta_path=corrupt_meta)


def test_validation_parity_against_reference_fixture():
    """Required Gate: verify artifact-to-fixture score parity against original fitted model.

    This test loads the 5-record validation fixture generated directly from the live
    fitted notebook objects (without touching test rows). All predicted probabilities
    must match expected scores within floating-point tolerance (1e-6).
    """
    assert FIXTURE_PATH.exists(), f"Required validation fixture missing: {FIXTURE_PATH}"

    fixture_records = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(fixture_records) == 5, f"Expected 5 records, got {len(fixture_records)}"

    model = IntakeRiskModel.load()

    for idx, item in enumerate(fixture_records):
        features = item["features"]
        expected_score = item["expected_score"]

        # 1. Test batch predict_proba
        probas = model.predict_proba([features])
        actual_score = float(probas[0])

        diff = abs(actual_score - expected_score)
        assert diff < 1e-6, (
            f"Parity gate failed for record {idx}!\n"
            f"  Features      : {features}\n"
            f"  Expected score: {expected_score:.12f}\n"
            f"  Actual score  : {actual_score:.12f}\n"
            f"  Difference    : {diff:.2e} (exceeds tolerance 1e-6)"
        )

        # 2. Test single score through Pydantic request
        req = IntakeFeaturesRequest(**features)
        resp = model.score_single(req)
        assert abs(resp.risk_score - expected_score) < 1e-6
        assert resp.above_frozen_cutoff == (resp.risk_score >= FROZEN_OPERATING_CUTOFF)


def test_categorical_null_normalization():
    """Verify that null/None is mapped to 'MISSING' while blank strings are preserved as-is."""
    model = IntakeRiskModel.load()

    # Null categoricals
    df_null = model.prepare_dataframe([{
        "descriptor": None,
        "descriptor_2": None,
        "location_type": None,
        "open_data_channel_type": None,
        "agency": None,
        "created_hour": 10,
        "created_day_of_week": 1,
        "created_is_weekend": 0,
    }])
    for col in ["descriptor", "descriptor_2", "location_type", "open_data_channel_type", "agency"]:
        assert df_null[col].iloc[0] == "MISSING"

    # Blank strings must NOT be transformed to 'MISSING' (preserves training preprocessing)
    df_blank = model.prepare_dataframe([{
        "descriptor": "",
        "descriptor_2": "   ",
        "location_type": "",
        "open_data_channel_type": "",
        "agency": "",
        "created_hour": 10,
        "created_day_of_week": 1,
        "created_is_weekend": 0,
    }])
    assert df_blank["descriptor"].iloc[0] == ""
    assert df_blank["descriptor_2"].iloc[0] == "   "

    # Ensure scoring completes without exceptions on both
    proba_null = model.predict_proba([{"created_hour": 12, "created_day_of_week": 2, "created_is_weekend": 0}])
    assert 0.0 <= float(proba_null[0]) <= 1.0

    proba_blank = model.predict_proba([{
        "descriptor": "",
        "created_hour": 12,
        "created_day_of_week": 2,
        "created_is_weekend": 0,
    }])
    assert 0.0 <= float(proba_blank[0]) <= 1.0


def test_temporal_validation_and_consistency():
    """Verify range limits for hour/day and consistency between weekday and weekend flag."""
    # Valid weekday (Wednesday)
    req_valid_wd = IntakeFeaturesRequest(created_hour=14, created_day_of_week=2, created_is_weekend=0)
    assert req_valid_wd.created_is_weekend == 0

    # Valid weekend (Saturday, Sunday)
    req_sat = IntakeFeaturesRequest(created_hour=8, created_day_of_week=5, created_is_weekend=1)
    assert req_sat.created_is_weekend == 1
    req_sun = IntakeFeaturesRequest(created_hour=23, created_day_of_week=6, created_is_weekend=1)
    assert req_sun.created_is_weekend == 1

    # Inconsistent: Monday marked as weekend
    with pytest.raises(ValidationError, match="Inconsistent weekend flag"):
        IntakeFeaturesRequest(created_hour=12, created_day_of_week=0, created_is_weekend=1)

    # Inconsistent: Saturday marked as weekday
    with pytest.raises(ValidationError, match="Inconsistent weekend flag"):
        IntakeFeaturesRequest(created_hour=12, created_day_of_week=5, created_is_weekend=0)

    # Out of range hour
    with pytest.raises(ValidationError):
        IntakeFeaturesRequest(created_hour=-1, created_day_of_week=2, created_is_weekend=0)
    with pytest.raises(ValidationError):
        IntakeFeaturesRequest(created_hour=24, created_day_of_week=2, created_is_weekend=0)

    # Out of range day of week
    with pytest.raises(ValidationError):
        IntakeFeaturesRequest(created_hour=12, created_day_of_week=-1, created_is_weekend=0)
    with pytest.raises(ValidationError):
        IntakeFeaturesRequest(created_hour=12, created_day_of_week=7, created_is_weekend=0)


def test_cutoff_boundary_logic():
    """Verify above_frozen_cutoff correctly reflects the 0.28 operating threshold."""
    model = IntakeRiskModel.load()
    req = IntakeFeaturesRequest(
        descriptor="APARTMENT ONLY",
        descriptor_2="NO HEAT",
        location_type="RESIDENTIAL BUILDING",
        open_data_channel_type="ONLINE",
        agency="HPD",
        created_hour=18,
        created_day_of_week=0,
        created_is_weekend=0,
    )
    resp = model.score_single(req)
    assert resp.operating_cutoff == 0.28
    assert resp.above_frozen_cutoff is (resp.risk_score >= 0.28)
    assert resp.calibrated is False
    assert "0.28" in resp.disclaimer
    assert "41.4%" in resp.disclaimer
