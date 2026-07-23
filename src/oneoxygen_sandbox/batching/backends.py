"""Mock and official OpenAI implementations of the batch backend contract."""

from __future__ import annotations

import importlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

from oneoxygen_sandbox.batching.models import (
    BatchCapabilities,
    BatchItemResult,
    BatchJob,
    BatchRequest,
    BatchState,
    BatchUsageAccounting,
    DiscountMetadata,
    deterministic_custom_id,
)
from oneoxygen_sandbox.batching.store import BatchArtifactStore, SQLiteBatchStore
from oneoxygen_sandbox.errors import ConfigurationError, ModelError
from oneoxygen_sandbox.model_adapters.openai import parse_openai_responses_response
from oneoxygen_sandbox.models import (
    InferenceTransport,
    ModelErrorCode,
    ModelProvider,
    ModelRunConfig,
    ModelUsage,
    ProvenanceClassification,
)

_SDK_UNSET: Final = object()
OPENAI_BATCH_SOURCE = "https://developers.openai.com/api/docs/guides/batch"
OPENAI_BATCH_VERIFIED_DATE = "2026-07-23"


@runtime_checkable
class BatchBackend(Protocol):
    def capabilities(self) -> BatchCapabilities: ...

    def validate_requests(self, requests: tuple[BatchRequest, ...]) -> None: ...

    def build_batch(self, requests: tuple[BatchRequest, ...]) -> BatchJob: ...

    def submit_batch(self, job: BatchJob) -> BatchJob: ...

    def get_status(self, job: BatchJob) -> BatchJob: ...

    def retrieve_results(self, job: BatchJob) -> tuple[BatchItemResult, ...]: ...

    def cancel_batch(self, job: BatchJob) -> BatchJob: ...

    def normalize_result(
        self, request: BatchRequest, raw_result: Mapping[str, Any]
    ) -> BatchItemResult: ...

    def classify_error(self, raw_error: Mapping[str, Any]) -> tuple[str, bool]: ...


def _discount() -> DiscountMetadata:
    return DiscountMetadata(
        discount_type="fixed_fraction",
        documented_discount_fraction=0.5,
        source_url=OPENAI_BATCH_SOURCE,
        source_verification_date=OPENAI_BATCH_VERIFIED_DATE,
        estimated=False,
        provider_reported=True,
    )


class MockBatchBackend:
    """Deterministic, restart-safe backend with injectable failure scenarios."""

    def __init__(
        self,
        artifacts: BatchArtifactStore,
        store: SQLiteBatchStore,
        *,
        responses: Mapping[str, Mapping[str, Any]] | None = None,
        errors: Mapping[str, Mapping[str, Any]] | None = None,
        delayed_status_polls: int = 0,
        provider_failure: bool = False,
        expire: bool = False,
        duplicate_result_id: str | None = None,
        missing_result_ids: tuple[str, ...] = (),
        include_unknown_result: bool = False,
    ) -> None:
        self.artifacts = artifacts
        self.store = store
        self.responses = dict(responses or {})
        self.errors = dict(errors or {})
        self.delayed_status_polls = delayed_status_polls
        self.provider_failure = provider_failure
        self.expire = expire
        self.duplicate_result_id = duplicate_result_id
        self.missing_result_ids = frozenset(missing_result_ids)
        self.include_unknown_result = include_unknown_result

    def capabilities(self) -> BatchCapabilities:
        return BatchCapabilities(
            provider=ModelProvider.SCRIPTED,
            endpoints=("/v1/responses",),
            maximum_requests_per_batch=50_000,
            maximum_jsonl_bytes=200 * 1024 * 1024,
            completion_windows=("deterministic",),
            custom_id_pattern=r"^[A-Za-z0-9_-]{1,64}$",
            custom_id_maximum_length=64,
            source_url="scripted://oneoxygen/mock-batch",
            source_verification_date=OPENAI_BATCH_VERIFIED_DATE,
        )

    def validate_requests(self, requests: tuple[BatchRequest, ...]) -> None:
        if not requests:
            raise ConfigurationError("cannot build an empty batch")
        if len({request.internal_request_id for request in requests}) != len(requests):
            raise ConfigurationError("batch request IDs must be unique")
        first = requests[0]
        for request in requests:
            if (
                request.model != first.model
                or request.endpoint != first.endpoint
                or request.provider != first.provider
            ):
                raise ConfigurationError("mock batch requests must be grouped compatibly")
            self._validated_body(request)

    def build_batch(self, requests: tuple[BatchRequest, ...]) -> BatchJob:
        self.validate_requests(requests)
        internal_id = f"mock-{uuid.uuid4().hex}"
        lines: list[bytes] = []
        custom_ids: set[str] = set()
        for request in sorted(requests, key=lambda item: item.internal_request_id):
            custom_id = deterministic_custom_id(
                request.run_id, request.turn_number, request.attempt_number
            )
            if custom_id in custom_ids:
                raise ConfigurationError("generated batch custom IDs are not unique")
            custom_ids.add(custom_id)
            self.store.save_request(request.model_copy(update={"provider_custom_id": custom_id}))
            lines.append(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": request.endpoint,
                        "body": self._validated_body(request),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        reference = f"batches/{internal_id}/input.jsonl"
        self.artifacts.write_bytes(reference, b"".join(lines))
        first = requests[0]
        job = BatchJob(
            internal_batch_id=internal_id,
            provider=first.provider,
            model=first.model,
            endpoint=first.endpoint,
            input_file_reference=reference,
            total_items=len(requests),
            request_ids=tuple(request.internal_request_id for request in requests),
        )
        self.store.save_job(job)
        for request in sorted(requests, key=lambda item: item.internal_request_id):
            # Recompute instead of relying on a positional parse of the custom ID.
            actual = deterministic_custom_id(
                request.run_id, request.turn_number, request.attempt_number
            )
            self.store.save_mapping(internal_id, actual, request.internal_request_id)
        return job

    def submit_batch(self, job: BatchJob) -> BatchJob:
        current = self.store.load_job(job.internal_batch_id)
        if current.provider_batch_id is not None:
            return current
        if current.submission_unknown:
            raise ModelError(
                ModelErrorCode.REMOTE_STATE_UNKNOWN,
                "mock batch remote submission state is unknown",
            )
        updated = current.model_copy(
            update={
                "provider_batch_id": f"mock_remote_{current.internal_batch_id}",
                "state": BatchState.SUBMITTED,
                "submitted_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
        )
        self.store.save_job(updated)
        return updated

    def get_status(self, job: BatchJob) -> BatchJob:
        current = self.store.load_job(job.internal_batch_id)
        if current.state is BatchState.CANCELLED:
            return current
        polls = int(current.provider_metadata.get("status_polls", 0)) + 1
        if self.provider_failure:
            state = BatchState.FAILED
        elif self.expire:
            state = BatchState.EXPIRED
        elif polls <= self.delayed_status_polls:
            state = BatchState.IN_PROGRESS
        else:
            state = BatchState.COMPLETED
        updated = current.model_copy(
            update={
                "state": state,
                "updated_at": datetime.now(UTC),
                "completed_at": (
                    datetime.now(UTC)
                    if state in {BatchState.COMPLETED, BatchState.FAILED, BatchState.EXPIRED}
                    else None
                ),
                "provider_metadata": {
                    **current.provider_metadata,
                    "status_polls": polls,
                    "provider_status": state.value,
                },
            }
        )
        self.store.save_job(updated)
        return updated

    def retrieve_results(self, job: BatchJob) -> tuple[BatchItemResult, ...]:
        current = self.store.load_job(job.internal_batch_id)
        if current.state not in {
            BatchState.COMPLETED,
            BatchState.PARTIALLY_COMPLETED,
            BatchState.EXPIRED,
        }:
            raise ConfigurationError("mock batch results are not ready")
        mappings = self.store.mappings(current.internal_batch_id)
        rows: list[dict[str, Any]] = []
        for custom_id, request_id in reversed(tuple(mappings.items())):
            if request_id in self.missing_result_ids:
                continue
            if request_id in self.errors:
                rows.append(
                    {
                        "custom_id": custom_id,
                        "response": None,
                        "error": dict(self.errors[request_id]),
                    }
                )
            else:
                rows.append(
                    {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": 200,
                            "body": dict(
                                self.responses.get(
                                    request_id,
                                    {
                                        "id": f"mock_response_{request_id}",
                                        "model": current.model,
                                        "status": "completed",
                                        "output": [],
                                        "usage": {
                                            "input_tokens": 0,
                                            "output_tokens": 0,
                                            "total_tokens": 0,
                                        },
                                    },
                                )
                            ),
                        },
                        "error": None,
                    }
                )
        if self.duplicate_result_id is not None:
            duplicate_custom = next(
                (
                    custom
                    for custom, request_id in mappings.items()
                    if request_id == self.duplicate_result_id
                ),
                None,
            )
            if duplicate_custom is not None:
                duplicate = next(row for row in rows if row["custom_id"] == duplicate_custom)
                rows.append(dict(duplicate))
        if self.include_unknown_result:
            rows.append(
                {
                    "custom_id": "o2_unknown_t0001_a01",
                    "response": {"status_code": 200, "body": {}},
                    "error": None,
                }
            )
        return self._correlate_rows(current, rows)

    def cancel_batch(self, job: BatchJob) -> BatchJob:
        current = self.store.load_job(job.internal_batch_id)
        updated = current.model_copy(
            update={
                "state": BatchState.CANCELLED,
                "updated_at": datetime.now(UTC),
                "completed_at": datetime.now(UTC),
            }
        )
        self.store.save_job(updated)
        return updated

    def normalize_result(
        self, request: BatchRequest, raw_result: Mapping[str, Any]
    ) -> BatchItemResult:
        custom_id = str(raw_result.get("custom_id", request.provider_custom_id or "unknown"))
        error = raw_result.get("error")
        response = raw_result.get("response")
        if error is not None:
            code, retryable = self.classify_error(error)
            return BatchItemResult(
                request_id=request.internal_request_id,
                provider_custom_id=custom_id,
                success=False,
                normalized_error={"code": code, "message": _error_message(error)},
                retryable=retryable,
            )
        if not isinstance(response, Mapping):
            return BatchItemResult(
                request_id=request.internal_request_id,
                provider_custom_id=custom_id,
                success=False,
                normalized_error={
                    "code": "invalid_provider_response",
                    "message": "batch response is missing",
                },
            )
        body = response.get("body")
        if not isinstance(body, Mapping):
            return BatchItemResult(
                request_id=request.internal_request_id,
                provider_custom_id=custom_id,
                success=False,
                normalized_error={
                    "code": "invalid_provider_response",
                    "message": "batch response body is missing",
                },
            )
        return BatchItemResult(
            request_id=request.internal_request_id,
            provider_custom_id=custom_id,
            success=True,
            response=dict(body),
            usage=_usage_from_body(body),
        )

    def classify_error(self, raw_error: Mapping[str, Any]) -> tuple[str, bool]:
        code = str(raw_error.get("code") or "provider_error")[:128]
        retryable = bool(raw_error.get("retryable")) or code in {
            "rate_limited",
            "request_timeout",
            "server_error",
            "provider_unavailable",
            "batch_expired",
        }
        return code, retryable

    def _validated_body(self, request: BatchRequest) -> dict[str, Any]:
        body = self.artifacts.read_json(request.compiled_request_body_reference)
        if not isinstance(body, dict):
            raise ConfigurationError("compiled batch request body must be an object")
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
        if _sha256(encoded) != request.compiled_request_body_sha256:
            raise ConfigurationError("compiled batch request body hash does not match")
        return body

    def _correlate_rows(
        self, job: BatchJob, rows: list[dict[str, Any]]
    ) -> tuple[BatchItemResult, ...]:
        mappings = self.store.mappings(job.internal_batch_id)
        seen: set[str] = set()
        results: list[BatchItemResult] = []
        for row in rows:
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or custom_id not in mappings:
                raise ModelError(
                    ModelErrorCode.BATCH_CORRELATION_ERROR,
                    "batch results contained an unknown custom ID",
                )
            if custom_id in seen:
                raise ModelError(
                    ModelErrorCode.BATCH_CORRELATION_ERROR,
                    "batch results contained a duplicate custom ID",
                )
            seen.add(custom_id)
            request = self.store.load_request(mappings[custom_id])
            results.append(self.normalize_result(request, row))
        for custom_id, request_id in mappings.items():
            if custom_id not in seen:
                results.append(
                    BatchItemResult(
                        request_id=request_id,
                        provider_custom_id=custom_id,
                        success=False,
                        normalized_error={
                            "code": "missing_result",
                            "message": "provider result was missing",
                        },
                        retryable=True,
                    )
                )
        return tuple(sorted(results, key=lambda item: item.request_id))


class OpenAIBatchBackend(MockBatchBackend):
    """Official OpenAI Batch API backend using Files + `/v1/responses`."""

    def __init__(
        self,
        artifacts: BatchArtifactStore,
        store: SQLiteBatchStore,
        *,
        client: Any | None = None,
        sdk_module: Any = _SDK_UNSET,
        environ: Mapping[str, str] | None = None,
        maximum_requests_per_batch: int = 50_000,
        maximum_jsonl_bytes: int = 200 * 1024 * 1024,
    ) -> None:
        super().__init__(artifacts, store)
        self._client = client
        self._sdk_module = sdk_module
        self._environ = os.environ if environ is None else environ
        self.maximum_requests_per_batch = min(maximum_requests_per_batch, 50_000)
        self.maximum_jsonl_bytes = min(maximum_jsonl_bytes, 200 * 1024 * 1024)

    def capabilities(self) -> BatchCapabilities:
        return BatchCapabilities(
            provider=ModelProvider.OPENAI,
            endpoints=("/v1/responses",),
            maximum_requests_per_batch=50_000,
            maximum_jsonl_bytes=200 * 1024 * 1024,
            completion_windows=("24h",),
            custom_id_pattern=r"^[A-Za-z0-9_-]{1,64}$",
            custom_id_maximum_length=64,
            discount=_discount(),
            source_url=OPENAI_BATCH_SOURCE,
            source_verification_date=OPENAI_BATCH_VERIFIED_DATE,
        )

    def validate_requests(self, requests: tuple[BatchRequest, ...]) -> None:
        if not requests:
            raise ConfigurationError("cannot build an empty OpenAI batch")
        if len(requests) > self.maximum_requests_per_batch:
            raise ConfigurationError("OpenAI batch exceeds the configured request limit")
        if len({request.internal_request_id for request in requests}) != len(requests):
            raise ConfigurationError("OpenAI batch request IDs must be unique")
        models = {request.model for request in requests}
        if len(models) != 1:
            raise ConfigurationError("an OpenAI input file can contain only one model")
        for request in requests:
            if (
                request.provider is not ModelProvider.OPENAI
                or request.transport is not InferenceTransport.PROVIDER_BATCH
                or request.endpoint != "/v1/responses"
            ):
                raise ConfigurationError("OpenAI batch request route is invalid")
            body = self._validated_body(request)
            if body.get("model") != request.model:
                raise ConfigurationError("OpenAI request body model does not match its batch")
            if body.get("stream") is True:
                raise ConfigurationError("OpenAI Batch does not support streaming")
            if _contains_secret_field(body):
                raise ConfigurationError("credentials are forbidden in OpenAI JSONL")

    def build_batch(self, requests: tuple[BatchRequest, ...]) -> BatchJob:
        self.validate_requests(requests)
        internal_id = f"oai-{uuid.uuid4().hex}"
        lines: list[bytes] = []
        custom_ids: set[str] = set()
        prepared: list[tuple[BatchRequest, str]] = []
        for request in sorted(requests, key=lambda item: item.internal_request_id):
            custom_id = deterministic_custom_id(
                request.run_id, request.turn_number, request.attempt_number
            )
            if custom_id in custom_ids:
                raise ConfigurationError("OpenAI custom IDs must be unique within a batch")
            custom_ids.add(custom_id)
            prepared_request = request.model_copy(update={"provider_custom_id": custom_id})
            prepared.append((prepared_request, custom_id))
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": self._validated_body(request),
            }
            encoded = (
                json.dumps(
                    line,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            lines.append(encoded)
        payload = b"".join(lines)
        if len(payload) > self.maximum_jsonl_bytes:
            raise ConfigurationError("OpenAI JSONL exceeds the configured byte limit")
        self._validate_jsonl(payload, len(requests))
        reference = f"batches/{internal_id}/input.jsonl"
        self.artifacts.write_bytes(reference, payload)
        first = requests[0]
        job = BatchJob(
            internal_batch_id=internal_id,
            provider=ModelProvider.OPENAI,
            model=first.model,
            endpoint="/v1/responses",
            state=BatchState.DRAFT,
            input_file_reference=reference,
            total_items=len(requests),
            request_ids=tuple(request.internal_request_id for request in requests),
            discount=_discount(),
            provider_metadata={
                "completion_window": "24h",
                "documentation_verified": OPENAI_BATCH_VERIFIED_DATE,
            },
        )
        self.store.save_job(job)
        for request, custom_id in prepared:
            self.store.save_request(request)
            self.store.save_mapping(internal_id, custom_id, request.internal_request_id)
        return job

    def submit_batch(self, job: BatchJob) -> BatchJob:
        current = self.store.load_job(job.internal_batch_id)
        if current.provider_batch_id is not None:
            return current
        if current.submission_unknown:
            raise ModelError(
                ModelErrorCode.REMOTE_STATE_UNKNOWN,
                "OpenAI batch submission state is unknown; inspect it before retrying.",
            )
        if current.input_file_reference is None:
            raise ConfigurationError("OpenAI batch input file is missing")
        self._ensure_client()
        input_path = self.artifacts.resolve(current.input_file_reference)
        try:
            with input_path.open("rb") as stream:
                uploaded = self._client.files.create(file=stream, purpose="batch")
            uploaded_id = _required_string(_read(uploaded, "id"), "uploaded file ID")
            remote = self._client.batches.create(
                input_file_id=uploaded_id,
                endpoint="/v1/responses",
                completion_window="24h",
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and 400 <= status_code < 500:
                failed = current.model_copy(
                    update={
                        "state": BatchState.FAILED,
                        "updated_at": datetime.now(UTC),
                        "provider_metadata": {
                            **current.provider_metadata,
                            "submission_error_status": status_code,
                        },
                    }
                )
                self.store.save_job(failed)
                raise ModelError(
                    ModelErrorCode.INVALID_REQUEST,
                    "OpenAI rejected the batch submission.",
                ) from None
            unknown = current.model_copy(
                update={
                    "state": BatchState.SUBMITTED,
                    "submission_unknown": True,
                    "updated_at": datetime.now(UTC),
                    "provider_metadata": {
                        **current.provider_metadata,
                        "remote_state": "unknown_after_submission_attempt",
                    },
                }
            )
            self.store.save_job(unknown)
            raise ModelError(
                ModelErrorCode.REMOTE_STATE_UNKNOWN,
                "OpenAI batch submission may have succeeded; blind resubmission is disabled.",
            ) from exc
        provider_id = _required_string(_read(remote, "id"), "provider batch ID")
        updated = current.model_copy(
            update={
                "provider_batch_id": provider_id,
                "state": _openai_state(_read(remote, "status")),
                "submitted_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "provider_metadata": {
                    **current.provider_metadata,
                    "input_file_id": uploaded_id,
                    "provider_status": _bounded(_read(remote, "status")),
                },
            }
        )
        # The durable remote ID is committed before the caller sees success.
        self.store.save_job(updated)
        return updated

    def get_status(self, job: BatchJob) -> BatchJob:
        current = self.store.load_job(job.internal_batch_id)
        if current.provider_batch_id is None:
            raise ConfigurationError("OpenAI batch has not been submitted")
        self._ensure_client()
        try:
            remote = self._client.batches.retrieve(current.provider_batch_id)
        except Exception as exc:
            raise ModelError(
                ModelErrorCode.REMOTE_STATE_UNKNOWN,
                "OpenAI batch status could not be verified.",
                retryable=True,
            ) from exc
        counts = _read(remote, "request_counts")
        state = _openai_state(_read(remote, "status"))
        completed = _nonnegative_integer(_read(counts, "completed")) or 0
        failed = _nonnegative_integer(_read(counts, "failed")) or 0
        if state is BatchState.COMPLETED and failed and completed:
            state = BatchState.PARTIALLY_COMPLETED
        updated = current.model_copy(
            update={
                "state": state,
                "succeeded_items": completed,
                "failed_items": failed,
                "updated_at": datetime.now(UTC),
                "completed_at": (
                    datetime.now(UTC)
                    if state
                    in {
                        BatchState.COMPLETED,
                        BatchState.PARTIALLY_COMPLETED,
                        BatchState.FAILED,
                        BatchState.EXPIRED,
                        BatchState.CANCELLED,
                    }
                    else None
                ),
                "provider_metadata": {
                    **current.provider_metadata,
                    "provider_status": _bounded(_read(remote, "status")),
                    "output_file_id": _bounded(_read(remote, "output_file_id")),
                    "error_file_id": _bounded(_read(remote, "error_file_id")),
                },
            }
        )
        self.store.save_job(updated)
        return updated

    def retrieve_results(self, job: BatchJob) -> tuple[BatchItemResult, ...]:
        current = self.store.load_job(job.internal_batch_id)
        if current.state not in {
            BatchState.COMPLETED,
            BatchState.PARTIALLY_COMPLETED,
            BatchState.EXPIRED,
        }:
            raise ConfigurationError("OpenAI batch results are not ready")
        self._ensure_client()
        rows: list[dict[str, Any]] = []
        output_id = current.provider_metadata.get("output_file_id")
        error_id = current.provider_metadata.get("error_file_id")
        updates: dict[str, Any] = {}
        if isinstance(output_id, str) and output_id:
            output_data = self._download_file(output_id)
            output_reference = f"batches/{current.internal_batch_id}/output.jsonl"
            self.artifacts.write_bytes(output_reference, output_data)
            updates["output_file_reference"] = output_reference
            rows.extend(_parse_jsonl(output_data))
        if isinstance(error_id, str) and error_id:
            error_data = self._download_file(error_id)
            error_reference = f"batches/{current.internal_batch_id}/errors.jsonl"
            self.artifacts.write_bytes(error_reference, error_data)
            updates["error_file_reference"] = error_reference
            rows.extend(_parse_jsonl(error_data))
        persisted = current.model_copy(update={**updates, "updated_at": datetime.now(UTC)})
        self.store.save_job(persisted)
        return self._correlate_rows(persisted, rows)

    def cancel_batch(self, job: BatchJob) -> BatchJob:
        current = self.store.load_job(job.internal_batch_id)
        if current.provider_batch_id is None:
            raise ConfigurationError("OpenAI batch has not been submitted")
        self._ensure_client()
        try:
            remote = self._client.batches.cancel(current.provider_batch_id)
        except Exception as exc:
            raise ModelError(
                ModelErrorCode.REMOTE_STATE_UNKNOWN,
                "OpenAI batch cancellation state could not be verified.",
                retryable=True,
            ) from exc
        updated = current.model_copy(
            update={
                "state": _openai_state(_read(remote, "status")),
                "updated_at": datetime.now(UTC),
                "provider_metadata": {
                    **current.provider_metadata,
                    "provider_status": _bounded(_read(remote, "status")),
                },
            }
        )
        self.store.save_job(updated)
        return updated

    def normalize_result(
        self, request: BatchRequest, raw_result: Mapping[str, Any]
    ) -> BatchItemResult:
        normalized = super().normalize_result(request, raw_result)
        if not normalized.success or normalized.response is None:
            return normalized
        response_wrapper = raw_result.get("response")
        status = (
            response_wrapper.get("status_code") if isinstance(response_wrapper, Mapping) else None
        )
        if isinstance(status, int) and status != 200:
            code, retryable = self.classify_error({"code": f"http_{status}", "status_code": status})
            return BatchItemResult(
                request_id=request.internal_request_id,
                provider_custom_id=normalized.provider_custom_id,
                success=False,
                normalized_error={
                    "code": code,
                    "message": "OpenAI batch item request failed",
                },
                retryable=retryable,
            )
        return normalized.model_copy(
            update={
                "accounting": BatchUsageAccounting(
                    usage=normalized.usage,
                    documented_discount_fraction=0.5,
                    synchronous_equivalent_usage=normalized.usage,
                    estimated_savings=True,
                    provider_invoice_amount=None,
                )
            }
        )

    def normalize_model_response(
        self, request: BatchRequest, result: BatchItemResult
    ) -> tuple[Any, list[dict[str, Any]]]:
        if not result.success or result.response is None:
            raise ConfigurationError("cannot normalize a failed OpenAI batch item")
        config = ModelRunConfig(
            provider=ModelProvider.OPENAI,
            model=request.model,
            transport=InferenceTransport.PROVIDER_BATCH,
            provenance=ProvenanceClassification.OFFICIAL_PROVIDER,
        )
        response, serialized = parse_openai_responses_response(
            config, result.response, latency_seconds=0
        )
        response = response.model_copy(
            update={
                "batch_request_id": request.internal_request_id,
                "transport": InferenceTransport.PROVIDER_BATCH,
            }
        )
        return response, serialized

    def classify_error(self, raw_error: Mapping[str, Any]) -> tuple[str, bool]:
        status = raw_error.get("status_code")
        code = str(raw_error.get("code") or "provider_error")[:128]
        retryable = status in {408, 409, 429, 500, 502, 503, 504} or code in {
            "batch_expired",
            "rate_limit_exceeded",
            "server_error",
            "request_timeout",
        }
        return code, retryable

    def _ensure_client(self) -> None:
        key = self._environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ModelError(
                ModelErrorCode.MISSING_API_KEY,
                "OPENAI_API_KEY is required for OpenAI Batch.",
            )
        if self._client is not None:
            return
        sdk = self._resolve_sdk()
        constructor = getattr(sdk, "OpenAI", None)
        if not callable(constructor):
            raise ModelError(
                ModelErrorCode.PROVIDER_NOT_CONFIGURED,
                "the installed OpenAI SDK does not expose the required client",
            )
        self._client = constructor(api_key=key, max_retries=0)

    def _resolve_sdk(self) -> Any:
        if self._sdk_module is None:
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                "OpenAI Batch requires the 'openai' optional dependency.",
            )
        if self._sdk_module is not _SDK_UNSET:
            return self._sdk_module
        try:
            self._sdk_module = importlib.import_module("openai")
        except ImportError:
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                "OpenAI Batch requires the 'openai' optional dependency.",
            ) from None
        return self._sdk_module

    def _download_file(self, file_id: str) -> bytes:
        response = self._client.files.content(file_id)
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            return content
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text.encode("utf-8")
        read = getattr(response, "read", None)
        if callable(read):
            data = read()
            if isinstance(data, bytes):
                return data
        raise ModelError(
            ModelErrorCode.INVALID_PROVIDER_RESPONSE,
            "OpenAI returned an unreadable batch output file.",
        )

    def _validate_jsonl(self, payload: bytes, expected_lines: int) -> None:
        rows = _parse_jsonl(payload)
        if len(rows) != expected_lines:
            raise ConfigurationError("OpenAI JSONL line count is invalid")
        custom_ids: set[str] = set()
        models: set[str] = set()
        for row in rows:
            if set(row) != {"body", "custom_id", "method", "url"}:
                raise ConfigurationError("OpenAI JSONL contains unsupported fields")
            custom_id = row["custom_id"]
            body = row["body"]
            if (
                not isinstance(custom_id, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", custom_id)
                or custom_id in custom_ids
            ):
                raise ConfigurationError("OpenAI JSONL custom IDs are invalid")
            if row["method"] != "POST" or row["url"] != "/v1/responses":
                raise ConfigurationError("OpenAI JSONL route is invalid")
            if not isinstance(body, dict) or body.get("stream") is True:
                raise ConfigurationError("OpenAI JSONL body is invalid")
            if _contains_secret_field(body):
                raise ConfigurationError("credentials are forbidden in OpenAI JSONL")
            custom_ids.add(custom_id)
            model = body.get("model")
            if not isinstance(model, str):
                raise ConfigurationError("OpenAI JSONL body is missing its model")
            models.add(model)
        if len(models) != 1:
            raise ConfigurationError("OpenAI JSONL can contain only one model")


def _parse_jsonl(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError("batch JSONL is not UTF-8") from exc
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("batch JSONL contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise ConfigurationError("batch JSONL lines must be objects")
        rows.append(row)
    return rows


def _contains_secret_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(
                marker in normalized
                for marker in (
                    "authorization",
                    "api_key",
                    "apikey",
                    "credential",
                    "password",
                    "secret",
                )
            ):
                return True
            if _contains_secret_field(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_field(item) for item in value)
    return False


def _openai_state(value: Any) -> BatchState:
    mapping = {
        "validating": BatchState.VALIDATING,
        "failed": BatchState.FAILED,
        "in_progress": BatchState.IN_PROGRESS,
        "finalizing": BatchState.FINALIZING,
        "completed": BatchState.COMPLETED,
        "expired": BatchState.EXPIRED,
        "cancelling": BatchState.CANCELLING,
        "cancelled": BatchState.CANCELLED,
    }
    try:
        return mapping[str(value)]
    except KeyError as exc:
        raise ModelError(
            ModelErrorCode.INVALID_PROVIDER_RESPONSE,
            "OpenAI returned an unknown batch status.",
        ) from exc


def _usage_from_body(body: Mapping[str, Any]) -> ModelUsage:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return ModelUsage()
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    return ModelUsage(
        input_tokens=_nonnegative_integer(usage.get("input_tokens")),
        output_tokens=_nonnegative_integer(usage.get("output_tokens")),
        cached_input_tokens=(
            _nonnegative_integer(input_details.get("cached_tokens"))
            if isinstance(input_details, Mapping)
            else None
        ),
        reasoning_tokens=(
            _nonnegative_integer(output_details.get("reasoning_tokens"))
            if isinstance(output_details, Mapping)
            else None
        ),
        total_tokens=_nonnegative_integer(usage.get("total_tokens")),
    )


def _error_message(error: Any) -> str:
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str):
            return message[:2_000]
    return "batch item failed"


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelError(
            ModelErrorCode.INVALID_PROVIDER_RESPONSE,
            f"OpenAI returned an invalid {label}.",
        )
    return value


def _read(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _nonnegative_integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _bounded(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:512]


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
