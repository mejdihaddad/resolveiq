# resolveiq/src/resolveiq/risk_model.py
"""Intake risk model inference, artifact verification, and scoring schemas.

Implements the frozen tabular baseline Model 1 (Logistic Regression without borough).
Loads pre-trained pipeline artifact, validates cryptographic SHA-256 integrity,
enforces request validation, and returns uncalibrated risk scores and cutoff flags.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from resolveiq.config import REPO_ROOT

CATEGORICAL_FEATURES: list[str] = [
    "descriptor",
    "descriptor_2",
    "location_type",
    "open_data_channel_type",
    "agency",
]

NUMERIC_FEATURES: list[str] = [
    "created_hour",
    "created_day_of_week",
    "created_is_weekend",
]

MODEL_FEATURE_COLS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DEFAULT_MODEL_PATH: Path = REPO_ROOT / "data" / "models" / "intake_risk_model_v1.joblib"
DEFAULT_META_PATH: Path = REPO_ROOT / "data" / "models" / "intake_risk_model_v1.meta.json"

FROZEN_MODEL_NAME: str = "frozen_model_1_no_borough"
FROZEN_OPERATING_CUTOFF: float = 0.28
MODEL_DISCLAIMER: str = (
    "Score is an uncalibrated ranking score; probability calibration was not established. "
    "The above_frozen_cutoff indicator reflects the frozen validation cutoff of 0.28, which is "
    "not an operational review decision and does not guarantee a 20% review capacity (it flagged "
    "41.4% of intake during the final test period)."
)


class IntakeFeaturesRequest(BaseModel):
    """Features provided at intake time for scoring slow-resolution risk."""

    model_config = ConfigDict(extra="forbid")

    descriptor: Optional[str] = Field(
        default=None,
        description="Primary complaint descriptor (e.g. 'APARTMENT ONLY'). Missing values map to 'MISSING'.",
    )
    descriptor_2: Optional[str] = Field(
        default=None,
        description="Secondary complaint descriptor (e.g. 'NO HEAT'). Missing values map to 'MISSING'.",
    )
    location_type: Optional[str] = Field(
        default=None,
        description="Location type (e.g. 'RESIDENTIAL BUILDING'). Missing values map to 'MISSING'.",
    )
    open_data_channel_type: Optional[str] = Field(
        default=None,
        description="Submission channel (e.g. 'ONLINE', 'PHONE'). Missing values map to 'MISSING'.",
    )
    agency: Optional[str] = Field(
        default=None,
        description="Agency code (e.g. 'HPD'). Missing values map to 'MISSING'.",
    )
    created_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of intake timestamp in local time (0–23).",
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

    @model_validator(mode="after")
    def validate_weekend_consistency(self) -> IntakeFeaturesRequest:
        """Enforce strict consistency between created_day_of_week and created_is_weekend."""
        expected_weekend = 1 if self.created_day_of_week in (5, 6) else 0
        if self.created_is_weekend != expected_weekend:
            day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][self.created_day_of_week]
            raise ValueError(
                f"Inconsistent weekend flag: created_day_of_week={self.created_day_of_week} ({day_name}) "
                f"requires created_is_weekend={expected_weekend}, but got {self.created_is_weekend}."
            )
        return self


class IntakeRiskScoreResponse(BaseModel):
    """Response returned after scoring an intake request with frozen Model 1."""

    model_name: str = Field(
        default=FROZEN_MODEL_NAME,
        description="Identifier of the frozen model used for inference.",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated risk of slow resolution (uncalibrated ranking score).",
    )
    operating_cutoff: float = Field(
        default=FROZEN_OPERATING_CUTOFF,
        description="Frozen score threshold derived from validation capacity analysis.",
    )
    above_frozen_cutoff: bool = Field(
        ...,
        description="Whether risk_score >= operating_cutoff (0.28).",
    )
    calibrated: bool = Field(
        default=False,
        description="Flag indicating whether scores are calibrated probabilities.",
    )
    disclaimer: str = Field(
        default=MODEL_DISCLAIMER,
        description="Mandatory caveat explaining score properties and workload shift.",
    )


class IntakeRiskModel:
    """Encapsulates loading, verification, and inference for frozen Model 1."""

    def __init__(self, pipeline: Any, metadata: dict[str, Any], model_path: Path) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.model_path = model_path

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> IntakeRiskModel:
        """Load the model artifact after verifying its existence and SHA-256 sidecar."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata sidecar not found at {meta_path}")

        meta_bytes = meta_path.read_text(encoding="utf-8")
        metadata = json.loads(meta_bytes)
        expected_sha256 = metadata.get("sha256")
        if not expected_sha256:
            raise ValueError(f"No 'sha256' field found in metadata sidecar: {meta_path}")

        artifact_bytes = model_path.read_bytes()
        actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()

        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Model artifact SHA-256 checksum mismatch!\n"
                f"  Expected (sidecar): {expected_sha256}\n"
                f"  Actual (computed) : {actual_sha256}\n"
                f"Halting load to prevent serving unverified model artifact."
            )

        pipeline = joblib.load(model_path)
        return cls(pipeline=pipeline, metadata=metadata, model_path=model_path)

    def prepare_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a DataFrame strictly matching the training preprocessing format.

        Categorical nulls are mapped to 'MISSING'. Blank strings are preserved as-is.
        """
        df = pd.DataFrame(records)
        # Ensure all expected columns are present
        for col in MODEL_FEATURE_COLS:
            if col not in df.columns:
                df[col] = None

        df = df[MODEL_FEATURE_COLS].copy()

        # Preserve exact training normalization: only map null/None to 'MISSING'.
        # Do not convert blank strings to 'MISSING'.
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].apply(lambda v: "MISSING" if v is None else str(v))

        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col])

        return df

    def predict_proba(self, records: list[dict[str, Any]]) -> np.ndarray:
        """Return positive-class probability estimates for the given input records."""
        df = self.prepare_dataframe(records)
        probas = self.pipeline.predict_proba(df)[:, 1]
        return probas

    def score_single(self, request: IntakeFeaturesRequest) -> IntakeRiskScoreResponse:
        """Score a single validated intake request."""
        record = request.model_dump()
        score = float(self.predict_proba([record])[0])
        above_cutoff = bool(score >= FROZEN_OPERATING_CUTOFF)

        return IntakeRiskScoreResponse(
            model_name=FROZEN_MODEL_NAME,
            risk_score=score,
            operating_cutoff=FROZEN_OPERATING_CUTOFF,
            above_frozen_cutoff=above_cutoff,
            calibrated=False,
            disclaimer=MODEL_DISCLAIMER,
        )
