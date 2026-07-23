"""Validated provider-independent batch records and grouping rules."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field, field_validator, model_validator

from oneoxygen_sandbox.models import (
    InferenceTransport,
    ModelProvider,
    ModelUsage,
    StrictModel,
)


class BatchState(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class DiscountMetadata(StrictModel):
    discount_type: str = Field(min_length=1, max_length=64)
    documented_discount_fraction: float | None = Field(default=None, ge=0, le=1)
    source_url: str = Field(min_length=1, max_length=1_024)
    source_verification_date: str
    estimated: bool = False
    provider_reported: bool = True


class BatchCapabilities(StrictModel):
    provider: ModelProvider
    endpoints: tuple[str, ...]
    maximum_requests_per_batch: int | None = Field(default=None, ge=1)
    maximum_jsonl_bytes: int | None = Field(default=None, ge=1)
    completion_windows: tuple[str, ...] = ()
    custom_id_pattern: str = Field(min_length=1, max_length=256)
    custom_id_maximum_length: int = Field(ge=1, le=1_024)
    supports_cancellation: bool = True
    supports_partial_results: bool = True
    discount: DiscountMetadata | None = None
    source_url: str = Field(min_length=1, max_length=1_024)
    source_verification_date: str


class BatchOrchestrationConfig(StrictModel):
    maximum_requests_per_batch: int = Field(default=50_000, ge=1)
    maximum_jsonl_bytes: int = Field(default=200 * 1024 * 1024, ge=1)
    maximum_wait_before_flush_seconds: float = Field(default=60, ge=0, le=86_400)
    minimum_preferred_batch_size: int = Field(default=1, ge=1)
    maximum_simultaneously_active_batches: int = Field(default=4, ge=1, le=10_000)
    maximum_batch_retries: int = Field(default=2, ge=0, le=99)
    maximum_total_age_seconds: float = Field(default=48 * 60 * 60, gt=0)

    @model_validator(mode="after")
    def validate_batch_size(self) -> BatchOrchestrationConfig:
        if self.minimum_preferred_batch_size > self.maximum_requests_per_batch:
            raise ValueError("preferred batch size cannot exceed the maximum")
        return self


class BatchUsageAccounting(StrictModel):
    usage: ModelUsage = Field(default_factory=ModelUsage)
    billing_transport: InferenceTransport = InferenceTransport.PROVIDER_BATCH
    documented_discount_fraction: float | None = Field(default=None, ge=0, le=1)
    synchronous_equivalent_usage: ModelUsage = Field(default_factory=ModelUsage)
    estimated_savings: bool = True
    provider_invoice_amount: float | None = None


class BatchRequest(StrictModel):
    internal_request_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    turn_number: int = Field(ge=1)
    attempt_number: int = Field(ge=1, le=99)
    provider: ModelProvider
    transport: InferenceTransport = InferenceTransport.PROVIDER_BATCH
    model: str = Field(min_length=1, max_length=256)
    endpoint: str = Field(min_length=1, max_length=128)
    compiled_request_body_sha256: str
    compiled_request_body_reference: str = Field(min_length=1, max_length=1_024)
    creation_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_schema_sha256: str
    system_prompt_version: str = Field(min_length=1, max_length=128)
    schema_mode: str = Field(min_length=1, max_length=64)
    effective_generation_settings_sha256: str
    data_policy_class: str | None = Field(default=None, max_length=64)
    provider_custom_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator(
        "compiled_request_body_sha256",
        "tool_schema_sha256",
        "effective_generation_settings_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("batch hashes must be lowercase SHA-256 digests")
        return value

    @field_validator("compiled_request_body_reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("batch payload references must be safe relative POSIX paths")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_transport(self) -> BatchRequest:
        if self.transport is not InferenceTransport.PROVIDER_BATCH:
            raise ValueError("batch requests require provider_batch transport")
        return self


class BatchJob(StrictModel):
    internal_batch_id: str = Field(min_length=1, max_length=128)
    provider_batch_id: str | None = Field(default=None, min_length=1, max_length=512)
    provider: ModelProvider
    model: str = Field(min_length=1, max_length=256)
    endpoint: str = Field(min_length=1, max_length=128)
    state: BatchState = BatchState.DRAFT
    input_file_reference: str | None = Field(default=None, max_length=1_024)
    output_file_reference: str | None = Field(default=None, max_length=1_024)
    error_file_reference: str | None = Field(default=None, max_length=1_024)
    total_items: int = Field(default=0, ge=0)
    succeeded_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    discount: DiscountMetadata | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    request_ids: tuple[str, ...] = ()
    submission_unknown: bool = False

    @model_validator(mode="after")
    def validate_counts(self) -> BatchJob:
        if self.succeeded_items + self.failed_items > self.total_items:
            raise ValueError("batch item counts exceed the total")
        return self


class BatchItemResult(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    provider_custom_id: str = Field(min_length=1, max_length=128)
    success: bool
    response: dict[str, Any] | None = None
    normalized_error: dict[str, Any] | None = None
    retryable: bool = False
    usage: ModelUsage = Field(default_factory=ModelUsage)
    created_at: datetime | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    accounting: BatchUsageAccounting | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> BatchItemResult:
        if self.success and self.response is None:
            raise ValueError("successful batch results require a response")
        if self.success and self.normalized_error is not None:
            raise ValueError("successful batch results cannot contain an error")
        if not self.success and self.normalized_error is None:
            raise ValueError("failed batch results require a normalized error")
        return self


def deterministic_custom_id(run_id: str, turn_number: int, attempt_number: int) -> str:
    """Build a provider-safe opaque ID without customer or task information."""
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    value = f"o2_{digest}_t{turn_number:04d}_a{attempt_number:02d}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("generated custom ID is not provider-safe")
    return value


def retry_failed_requests(
    requests: tuple[BatchRequest, ...] | list[BatchRequest],
    results: tuple[BatchItemResult, ...] | list[BatchItemResult],
    *,
    maximum_retries: int,
) -> tuple[BatchRequest, ...]:
    """Create new-attempt records only for retryable failures."""
    if maximum_retries < 0:
        raise ValueError("maximum retries cannot be negative")
    by_id = {request.internal_request_id: request for request in requests}
    seen_results: set[str] = set()
    retries: list[BatchRequest] = []
    for result in results:
        if result.request_id in seen_results:
            raise ValueError("retry selection received duplicate result IDs")
        seen_results.add(result.request_id)
        request = by_id.get(result.request_id)
        if request is None:
            raise ValueError("retry selection received an unknown result ID")
        if result.success or not result.retryable:
            continue
        retries_used = request.attempt_number - 1
        if retries_used >= maximum_retries:
            continue
        next_attempt = request.attempt_number + 1
        digest = hashlib.sha256(
            f"{request.internal_request_id}:{next_attempt}".encode()
        ).hexdigest()[:32]
        retries.append(
            request.model_copy(
                update={
                    "internal_request_id": f"req_{digest}",
                    "attempt_number": next_attempt,
                    "creation_timestamp": datetime.now(UTC),
                    "provider_custom_id": None,
                }
            )
        )
    return tuple(sorted(retries, key=lambda item: item.internal_request_id))


def group_compatible_requests(
    requests: tuple[BatchRequest, ...] | list[BatchRequest],
) -> tuple[tuple[BatchRequest, ...], ...]:
    """Group only requests that match every benchmark-affecting dimension."""
    grouped: dict[tuple[Any, ...], list[BatchRequest]] = defaultdict(list)
    for request in requests:
        key = (
            request.provider,
            request.transport,
            request.model,
            request.endpoint,
            request.tool_schema_sha256,
            request.system_prompt_version,
            request.schema_mode,
            request.effective_generation_settings_sha256,
            request.data_policy_class,
        )
        grouped[key].append(request)
    return tuple(
        tuple(sorted(items, key=lambda item: item.internal_request_id))
        for _key, items in sorted(grouped.items(), key=lambda item: repr(item[0]))
    )
