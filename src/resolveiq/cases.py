# resolveiq/src/resolveiq/cases.py
"""Similar case retrieval, privacy safeguards, and temporal outcome masking.

Queries the frozen Qdrant index (resolveiq-cases-collection-01) using structured
complaint features. Restricts embedding text strictly to allowlisted intake
fields to prevent any PII or street addresses from reaching external APIs.

Enforces two strict temporal gates:
1. Admission Gate: A case is eligible for retrieval ONLY if its created_date is strictly
   earlier than the new ticket's intake_time (created_date < intake_time).
2. Outcome Gate: Outcome fields (status, closed_date, resolution_description) are exposed
   ONLY if closed_date is strictly earlier than intake_time (closed_date < intake_time).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional, Union
from zoneinfo import ZoneInfo

import httpx
from google import genai
from google.genai.errors import APIError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from qdrant_client.models import FieldCondition, Filter, MatchValue

from resolveiq.config import get_settings

NYC_TZ = ZoneInfo("America/New_York")

# Strict allowlists matching the canonical shapes in resolveiq-cases-collection-01
AllowedComplaintType = Literal["HEAT/HOT WATER"]
AllowedLocationType = Literal["RESIDENTIAL BUILDING"]
AllowedDescriptor = Literal["APARTMENT ONLY", "ENTIRE BUILDING"]
AllowedDescriptor2 = Literal["NO HEAT", "NO HEAT AND NO HOT WATER", "NO HOT WATER"]
AllowedBorough = Literal["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]

TEMPORAL_RULE_EXPLANATION: str = (
    "A case is eligible for retrieval ONLY if created_date is strictly earlier than intake_time. "
    "Outcome fields (status, closed_date, resolution_description) are exposed ONLY when closed_date "
    "is strictly earlier than intake_time (<, not <=) on a normalized UTC basis. Ambiguous naive local "
    "timestamps are treated conservatively (excluded or outcomes masked)."
)


class RetrievalUnavailableError(RuntimeError):
    """Raised when an optional embedding or Qdrant dependency is unavailable."""


class RetrievalRequestError(RuntimeError):
    """Raised when an optional retrieval request is invalid or misconfigured."""


RETRYABLE_RETRIEVAL_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _exception_status_code(exc: BaseException) -> int | None:
    """Read a provider status/code without exposing provider details."""
    for name in ("code", "status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _is_retryable_retrieval_error(exc: BaseException) -> bool:
    """Return whether a dependency failure is likely transient."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError, httpx.RequestError)):
        return True
    if isinstance(exc, ResponseHandlingException):
        return _is_retryable_retrieval_error(exc.source)
    return _exception_status_code(exc) in RETRYABLE_RETRIEVAL_STATUS_CODES


def parse_and_normalize_timestamp(val: Union[datetime, str]) -> tuple[datetime, bool]:
    """Parse and normalize any timestamp to UTC.

    Returns:
        (utc_datetime, is_ambiguous)

    Raises:
        ValueError if the naive timestamp is nonexistent in America/New_York.
    """
    if isinstance(val, str):
        clean_str = val.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
    else:
        dt = val

    if dt.tzinfo is None:
        # 1. Nonexistent time check (e.g. spring forward gap)
        dt_aware = dt.replace(tzinfo=NYC_TZ)
        dt_wall = dt_aware.astimezone(timezone.utc).astimezone(NYC_TZ).replace(tzinfo=None)
        if dt_wall != dt:
            raise ValueError(f"Nonexistent local time in America/New_York: {dt.isoformat()}")

        # 2. Ambiguity check (e.g. fall back fold)
        dt0 = dt.replace(tzinfo=NYC_TZ, fold=0)
        dt1 = dt.replace(tzinfo=NYC_TZ, fold=1)
        is_ambiguous = dt0.utcoffset() != dt1.utcoffset()
        utc_dt = dt0.astimezone(timezone.utc)
        return utc_dt, is_ambiguous

    # Already timezone-aware: preserve exact instant and convert to UTC
    return dt.astimezone(timezone.utc), False


def build_embedding_text(
    complaint_type: str,
    location_type: str,
    descriptor: str,
    descriptor_2: Optional[str] = None,
) -> str:
    """Construct deterministic embedding text solely from approved categorical fields."""
    parts = [complaint_type, location_type, descriptor]
    if descriptor_2:
        parts.append(descriptor_2)
    return ", ".join(parts)


class SimilarCasesRequest(BaseModel):
    """Structured request for finding similar historical cases."""

    model_config = ConfigDict(extra="forbid")

    complaint_type: AllowedComplaintType = Field(
        default="HEAT/HOT WATER",
        description="Controlled complaint type allowlist.",
    )
    location_type: AllowedLocationType = Field(
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
    borough: Optional[AllowedBorough] = Field(
        default=None,
        description="Optional borough filter matching exact collection payload.",
    )
    intake_time: str = Field(
        ...,
        description="ISO 8601 intake timestamp. Naive times are localized to America/New_York.",
        examples=["2024-01-27T18:00:00", "2024-01-27T23:00:00Z"],
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of similar cases to retrieve.",
    )

    @field_validator("intake_time")
    @classmethod
    def validate_intake_time(cls, v: str) -> str:
        """Reject nonexistent local intake times and ensure valid ISO string."""
        parse_and_normalize_timestamp(v)
        return v


class RetrievedCaseItem(BaseModel):
    """A similar historical case with conditional outcome masking."""

    ticket: str = Field(..., description="Unique 311 service request key.")
    shape_id: int = Field(..., description="Canonical complaint shape ID (1–6).")
    shape_slug: str = Field(..., description="Canonical complaint shape slug.")
    complaint_type: str = Field(..., description="Complaint type.")
    location_type: str = Field(..., description="Location type.")
    descriptor: Optional[str] = Field(default=None, description="Primary descriptor.")
    descriptor_2: Optional[str] = Field(default=None, description="Secondary descriptor.")
    borough: str = Field(..., description="Borough.")
    created_date: str = Field(..., description="Case creation timestamp.")
    similarity_score: float = Field(..., description="Cosine similarity score.")
    is_historical_evidence: bool = Field(
        ...,
        description="True ONLY if closed_date is strictly earlier than intake_time.",
    )
    status: Optional[str] = Field(
        default=None,
        description="Closure status (exposed only if is_historical_evidence=True).",
    )
    closed_date: Optional[str] = Field(
        default=None,
        description="Closure timestamp (exposed only if is_historical_evidence=True).",
    )
    resolution_description: Optional[str] = Field(
        default=None,
        description="Action notes (exposed only if is_historical_evidence=True).",
    )


class SimilarCasesResponse(BaseModel):
    """Response containing retrieved similar cases and safety metadata."""

    collection: str = Field(..., description="Qdrant collection queried.")
    borough_filter: Optional[str] = Field(default=None, description="Applied borough filter.")
    intake_time: str = Field(..., description="Intake timestamp used as temporal boundary.")
    cases: list[RetrievedCaseItem] = Field(..., description="List of similar cases.")
    temporal_rule: str = Field(
        default=TEMPORAL_RULE_EXPLANATION,
        description="Rule explanation for admission and outcome masking.",
    )


def format_retrieved_case(
    payload: dict[str, Any],
    similarity_score: float,
    intake_time_str: str,
) -> Optional[dict[str, Any]]:
    """Sole gateway for returning retrieved case payloads with strict temporal masking.

    1. Admission Gate: If created_date >= intake_time (or created_date is missing/ambiguous),
       the case is excluded entirely (returns None).
    2. Outcome Gate: If closed_date < intake_time (and not ambiguous), outcome fields are
       exposed; otherwise, outcome fields are masked to None.
    """
    intake_utc, intake_ambig = parse_and_normalize_timestamp(intake_time_str)

    # 1. Admission Gate: created_date must be strictly earlier than intake_time
    created_raw = payload.get("created_date")
    if not created_raw:
        return None

    try:
        created_utc, created_ambig = parse_and_normalize_timestamp(created_raw)
        if created_utc >= intake_utc or created_ambig:
            return None
    except Exception:
        return None

    # 2. Outcome Gate: closed_date strictly earlier than intake_time
    c_date_raw = payload.get("closed_date")
    is_historical = False
    if c_date_raw:
        try:
            closed_utc, closed_ambig = parse_and_normalize_timestamp(c_date_raw)
            if not intake_ambig and not closed_ambig:
                is_historical = closed_utc < intake_utc
        except Exception:
            is_historical = False

    return {
        "ticket": str(payload.get("unique_key", "")),
        "shape_id": int(payload.get("shape_id", 0)),
        "shape_slug": str(payload.get("shape_slug", "")),
        "complaint_type": str(payload.get("complaint_type", "")),
        "location_type": str(payload.get("location_type", "")),
        "descriptor": payload.get("descriptor"),
        "descriptor_2": payload.get("descriptor_2"),
        "borough": str(payload.get("borough", "")),
        "created_date": str(payload.get("created_date", "")),
        "similarity_score": similarity_score,
        "is_historical_evidence": is_historical,
        "status": payload.get("status") if is_historical else None,
        "closed_date": payload.get("closed_date") if is_historical else None,
        "resolution_description": payload.get("resolution_description") if is_historical else None,
    }


def embed_text_gemini(text: str) -> list[float]:
    """Generate 3072-dimensional embedding using Gemini API (used only on live requests)."""
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    resp = client.models.embed_content(
        model=settings.gemini_embed_model,
        contents=text,
    )
    return resp.embeddings[0].values


def retrieve_similar_cases(
    request: SimilarCasesRequest,
    query_vector: Optional[list[float]] = None,
    qdrant_client: Optional[QdrantClient] = None,
    embedder: Optional[Callable[[str], list[float]]] = None,
) -> SimilarCasesResponse:
    """Retrieve similar cases from Qdrant with optional borough filtering and temporal masking."""
    settings = get_settings()
    collection_name = settings.qdrant_collection_cases

    # 1. Obtain query vector (from injected test vector or embedder)
    try:
        if query_vector is None:
            text_to_embed = build_embedding_text(
                complaint_type=request.complaint_type,
                location_type=request.location_type,
                descriptor=request.descriptor,
                descriptor_2=request.descriptor_2,
            )
            embed_fn = embedder or embed_text_gemini
            query_vector = embed_fn(text_to_embed)

        # 2. Build Qdrant filter
        q_filter = None
        if request.borough:
            q_filter = Filter(
                must=[FieldCondition(key="borough", match=MatchValue(value=request.borough))]
            )

        # 3. Query Qdrant
        client = qdrant_client or QdrantClient(url=settings.qdrant_url)
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=q_filter,
            limit=request.limit,
            with_payload=True,
        )
    except (APIError, ApiException, ConnectionError, TimeoutError, OSError, httpx.RequestError) as exc:
        if _is_retryable_retrieval_error(exc):
            raise RetrievalUnavailableError(
                "Optional similar-case retrieval is unavailable."
            ) from exc
        raise RetrievalRequestError(
            "Optional similar-case retrieval could not be completed for this request."
        ) from exc

    # 4. Format each retrieved point through the temporal masking gateway
    formatted_cases: list[RetrievedCaseItem] = []
    for hit in search_result.points:
        raw_payload = hit.payload or {}
        case_dict = format_retrieved_case(
            payload=raw_payload,
            similarity_score=float(hit.score),
            intake_time_str=request.intake_time,
        )
        if case_dict is not None:
            formatted_cases.append(RetrievedCaseItem(**case_dict))

    return SimilarCasesResponse(
        collection=collection_name,
        borough_filter=request.borough,
        intake_time=request.intake_time,
        cases=formatted_cases,
        temporal_rule=TEMPORAL_RULE_EXPLANATION,
    )
