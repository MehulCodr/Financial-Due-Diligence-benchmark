"""Validated configuration and run-record models."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FORBIDDEN_SECRET_NAMES = {
    "AIRFORCE_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "COHERE_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "TOGETHER_API_KEY",
    "XAI_API_KEY",
}
MODEL_PROVIDER_MARKERS = (
    "AIRFORCE",
    "ANTHROPIC",
    "AZURE_OPENAI",
    "COHERE",
    "DEEPSEEK",
    "FIREWORKS",
    "GEMINI",
    "GROQ",
    "HUGGINGFACE",
    "MISTRAL",
    "OPENAI",
    "OPENROUTER",
    "PERPLEXITY",
    "TOGETHER",
    "XAI",
)
MAXIMUM_MODEL_METADATA_BYTES = 16 * 1024
MAXIMUM_MODEL_EVENT_TEXT_BYTES = 64 * 1024


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _FrozenJSONDict(dict[str, Any]):
    """A JSON mapping that retains normal serialization but rejects mutation."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("model configuration is immutable")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> _FrozenJSONDict:
        return self


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenJSONDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


class NetworkPolicy(StrEnum):
    DISABLED = "disabled"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    INCOMPLETE = "incomplete"
    REFUSED = "refused"
    LIMIT_EXCEEDED = "limit_exceeded"
    PROVIDER_ERROR = "provider_error"
    SANDBOX_ERROR = "sandbox_error"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class AgentTerminationReason(StrEnum):
    SUCCESSFUL_SUBMISSION = "successful_submission"
    FINAL_TEXT_WITHOUT_SUBMISSION = "final_text_without_submission"
    MAXIMUM_TURNS_REACHED = "maximum_turns_reached"
    MAXIMUM_PROVIDER_REQUESTS_REACHED = "maximum_provider_requests_reached"
    INPUT_TOKEN_LIMIT_REACHED = "input_token_limit_reached"
    OUTPUT_TOKEN_LIMIT_REACHED = "output_token_limit_reached"
    TOTAL_TOKEN_LIMIT_REACHED = "total_token_limit_reached"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    OVERALL_WALL_TIME_LIMIT_REACHED = "overall_wall_time_limit_reached"
    REPEATED_PROVIDER_FAILURE = "repeated_provider_failure"
    MODEL_REFUSAL = "model_refusal"
    SANDBOX_FAILURE = "sandbox_failure"
    TOOL_FAILURE = "tool_failure"
    USER_INTERRUPTION = "user_interruption"
    INTERNAL_ORCHESTRATION_ERROR = "internal_orchestration_error"


class FinalTextBehavior(StrEnum):
    INCOMPLETE = "incomplete"
    SUCCEED = "succeed"


class ModelProvider(StrEnum):
    SCRIPTED = "scripted"
    OPENAI = "openai"
    AIRFORCE = "airforce"


class InferenceTransport(StrEnum):
    DIRECT = "direct"
    PROVIDER_BATCH = "provider_batch"
    GATEWAY_DIRECT = "gateway_direct"


class ProvenanceClassification(StrEnum):
    OFFICIAL_PROVIDER = "official_provider"
    THIRD_PARTY_GATEWAY_UNVERIFIED = "third_party_gateway_unverified"
    SCRIPTED_TEST = "scripted_test"


class DataClassification(StrEnum):
    SYNTHETIC = "synthetic"
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ToolSchemaMode(StrEnum):
    PORTABLE = "portable"
    NATIVE_STRICT = "native_strict"


class NormalizedFinishReason(StrEnum):
    TOOL_CALLS = "tool_calls"
    COMPLETED = "completed"
    LENGTH = "length"
    REFUSED = "refused"
    CONTENT_FILTER = "content_filter"
    CANCELLED = "cancelled"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


class ModelErrorCode(StrEnum):
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    MISSING_DEPENDENCY = "missing_dependency"
    MISSING_API_KEY = "missing_api_key"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    REQUEST_TIMEOUT = "request_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    DUPLICATE_TOOL_CALL_ID = "duplicate_tool_call_id"
    MALFORMED_TOOL_ARGUMENTS = "malformed_tool_arguments"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    MODEL_REFUSAL = "model_refusal"
    INTERNAL_ADAPTER_ERROR = "internal_adapter_error"
    CANCELLED = "cancelled"
    DATA_POLICY_VIOLATION = "data_policy_violation"
    BATCH_CORRELATION_ERROR = "batch_correlation_error"
    REMOTE_STATE_UNKNOWN = "remote_state_unknown"


class ToolErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    CALL_LIMIT_EXCEEDED = "call_limit_exceeded"
    PATH_NOT_ALLOWED = "path_not_allowed"
    FILE_NOT_FOUND = "file_not_found"
    BINARY_FILE = "binary_file"
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"
    EXECUTION_TIMEOUT = "execution_timeout"
    EXECUTION_FAILED = "execution_failed"
    ALREADY_SUBMITTED = "already_submitted"
    INTERNAL_TOOL_ERROR = "internal_tool_error"


class ToolEventStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _validate_identifier(value: str, label: str) -> str:
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{label} must begin with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return value


def is_forbidden_environment_name(value: str) -> bool:
    upper = value.upper()
    if upper in FORBIDDEN_SECRET_NAMES or upper.endswith("_API_KEY"):
        return True
    return any(marker in upper for marker in MODEL_PROVIDER_MARKERS) and any(
        secret_word in upper for secret_word in ("CREDENTIAL", "KEY", "SECRET", "TOKEN")
    )


def sanitize_model_trace_text(value: str) -> str:
    """Redact credentials and local paths before model data is persisted."""
    sanitized = str(value).replace("\x00", "")
    for name, secret in os.environ.items():
        if secret and len(secret) >= 8 and is_forbidden_environment_name(name):
            sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", sanitized)
    sanitized = re.sub(
        r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+",
        r"\1[REDACTED]",
        sanitized,
    )
    for path in (Path.cwd(), Path.home()):
        path_text = str(path)
        if path_text:
            sanitized = sanitized.replace(path_text, "[HOST_PATH]")
            sanitized = sanitized.replace(path_text.replace("\\", "/"), "[HOST_PATH]")
    return re.sub(r"(?<![A-Za-z0-9])[A-Za-z]:\\[^\r\n\t\"']+", "[HOST_PATH]", sanitized)


def _sanitize_model_metadata(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_model_trace_text(value)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).upper()
            if is_forbidden_environment_name(normalized) or any(
                marker in normalized
                for marker in ("AUTHORIZATION", "API_KEY", "CREDENTIAL", "PASSWORD", "SECRET")
            ):
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _sanitize_model_metadata(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_model_metadata(item) for item in value]
    return value


def normalize_finish_reason(value: str | None) -> NormalizedFinishReason:
    """Map common provider finish labels without inventing provider-specific detail."""
    if value is None:
        return NormalizedFinishReason.UNKNOWN
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "tool_call": NormalizedFinishReason.TOOL_CALLS,
        "function_call": NormalizedFinishReason.TOOL_CALLS,
        "function_calls": NormalizedFinishReason.TOOL_CALLS,
        "stop": NormalizedFinishReason.COMPLETED,
        "complete": NormalizedFinishReason.COMPLETED,
        "end_turn": NormalizedFinishReason.COMPLETED,
        "max_tokens": NormalizedFinishReason.LENGTH,
        "max_output_tokens": NormalizedFinishReason.LENGTH,
        "refusal": NormalizedFinishReason.REFUSED,
        "blocked": NormalizedFinishReason.CONTENT_FILTER,
        "canceled": NormalizedFinishReason.CANCELLED,
        "error": NormalizedFinishReason.PROVIDER_ERROR,
    }
    try:
        return NormalizedFinishReason(normalized)
    except ValueError:
        return aliases.get(normalized, NormalizedFinishReason.UNKNOWN)


def _bounded_json_mapping(value: dict[str, Any], label: str) -> dict[str, Any]:
    value = _sanitize_model_metadata(value)
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc
    if len(serialized) > MAXIMUM_MODEL_METADATA_BYTES:
        raise ValueError(f"{label} exceeds {MAXIMUM_MODEL_METADATA_BYTES} bytes")
    return value


def _validate_provider_settings(value: dict[str, Any]) -> dict[str, Any]:
    def inspect(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).upper()
                if is_forbidden_environment_name(normalized) or any(
                    marker in normalized
                    for marker in (
                        "AUTHORIZATION",
                        "API_KEY",
                        "CREDENTIAL",
                        "PASSWORD",
                        "SECRET",
                    )
                ):
                    raise ValueError("provider settings may not contain credentials or secrets")
                inspect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                inspect(child)
        elif isinstance(item, str):
            sanitized = sanitize_model_trace_text(item)
            if "[REDACTED]" in sanitized or "[HOST_PATH]" in sanitized:
                raise ValueError("provider settings may not contain credentials or host paths")

    inspect(value)
    return _bounded_json_mapping(value, "provider_settings")


class ModelRunConfig(StrictModel):
    provider: ModelProvider
    model: str = Field(min_length=1, max_length=256)
    maximum_output_tokens: int = Field(default=4_096, ge=1, le=1_000_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    model_call_timeout_seconds: float = Field(default=60.0, gt=0, le=3_600)
    maximum_retry_attempts: int = Field(default=2, ge=0, le=20)
    initial_retry_delay_seconds: float = Field(default=1.0, ge=0, le=300)
    provider_settings: dict[str, Any] = Field(default_factory=dict)
    tool_schema_mode: ToolSchemaMode = ToolSchemaMode.PORTABLE
    store_provider_response: bool = False
    transport: InferenceTransport | None = None
    provenance: ProvenanceClassification | None = None
    api_host: str | None = Field(default=None, min_length=1, max_length=255)
    upstream_provider_verifiable: bool | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized or any(char in normalized for char in "\r\n"):
            raise ValueError("model must be a non-empty single-line identifier")
        sanitized = sanitize_model_trace_text(normalized)
        if "[REDACTED]" in sanitized or "[HOST_PATH]" in sanitized:
            raise ValueError("model identifier may not contain credentials or host paths")
        return normalized

    @field_validator("provider_settings")
    @classmethod
    def validate_provider_settings(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _freeze_json(_validate_provider_settings(value))

    @model_validator(mode="after")
    def normalize_route(self) -> ModelRunConfig:
        if self.provider is ModelProvider.OPENAI:
            allowed = {InferenceTransport.DIRECT, InferenceTransport.PROVIDER_BATCH}
            transport = self.transport or InferenceTransport.DIRECT
            provenance = self.provenance or ProvenanceClassification.OFFICIAL_PROVIDER
            host = self.api_host or "api.openai.com"
            verifiable = (
                True
                if self.upstream_provider_verifiable is None
                else self.upstream_provider_verifiable
            )
            if (
                transport not in allowed
                or provenance is not ProvenanceClassification.OFFICIAL_PROVIDER
                or host != "api.openai.com"
                or not verifiable
            ):
                raise ValueError("OpenAI routes must use the official api.openai.com service")
        elif self.provider is ModelProvider.AIRFORCE:
            transport = self.transport or InferenceTransport.GATEWAY_DIRECT
            provenance = self.provenance or ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED
            host = self.api_host or "api.airforce"
            verifiable = (
                False
                if self.upstream_provider_verifiable is None
                else self.upstream_provider_verifiable
            )
            if (
                transport is not InferenceTransport.GATEWAY_DIRECT
                or provenance is not ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED
                or host != "api.airforce"
                or verifiable
            ):
                raise ValueError("Airforce is restricted to its unverified gateway route")
        else:
            transport = self.transport or InferenceTransport.DIRECT
            provenance = self.provenance or ProvenanceClassification.SCRIPTED_TEST
            host = self.api_host or "scripted.local"
            verifiable = (
                False
                if self.upstream_provider_verifiable is None
                else self.upstream_provider_verifiable
            )
            if provenance is not ProvenanceClassification.SCRIPTED_TEST or verifiable:
                raise ValueError("scripted runs must retain scripted-test provenance")

        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "api_host", host)
        object.__setattr__(self, "upstream_provider_verifiable", verifiable)
        return self

    def requested_settings(self) -> dict[str, Any]:
        """Return the sanitized, canonical settings requested for this run."""
        return self.model_dump(mode="json")


class ModelCapabilities(StrictModel):
    tool_calling: bool = False
    multiple_tool_calls_per_turn: bool = False
    reasoning_token_reporting: bool = False
    cached_token_reporting: bool = False
    temperature_support: bool = False
    seed_support: bool = False
    strict_tool_schemas: bool = False
    response_storage_control: bool = False


class ModelUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider_metadata")
    @classmethod
    def validate_provider_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_json_mapping(value, "usage provider metadata")

    @classmethod
    def aggregate(cls, values: tuple[ModelUsage, ...] | list[ModelUsage]) -> ModelUsage:
        items = tuple(values)

        def exact_sum(field_name: str) -> int | None:
            if not items:
                return None
            components = [getattr(item, field_name) for item in items]
            return None if any(component is None for component in components) else sum(components)

        return cls(
            input_tokens=exact_sum("input_tokens"),
            output_tokens=exact_sum("output_tokens"),
            reasoning_tokens=exact_sum("reasoning_tokens"),
            cached_input_tokens=exact_sum("cached_input_tokens"),
            total_tokens=exact_sum("total_tokens"),
        )


def _workspace_relative(value: str, label: str, *, allow_root: bool) -> PurePosixPath:
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must use a safe POSIX container path")
    path = PurePosixPath(value)
    if path.is_absolute():
        try:
            path = path.relative_to("/workspace")
        except ValueError as exc:
            raise ValueError(f"{label} must be /workspace or a child of it") from exc
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} contains an unsafe path component")
    if not allow_root and not path.parts:
        raise ValueError(f"{label} may not refer to the complete workspace")
    return path


def _task_relative_file(value: str, label: str) -> str:
    if "\x00" in value or "\\" in value:
        raise ValueError(f"{label} must use a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"{label} must be relative to the task directory")
    if any(part in {"", ".", ".."} for part in path.parts) or not path.parts:
        raise ValueError(f"{label} contains an unsafe path component")
    return path.as_posix()


class InputAsset(StrictModel):
    source: str = Field(min_length=1, max_length=512)
    destination: str = Field(min_length=1, max_length=512)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("asset source contains a null byte")
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or re.match(r"^[A-Za-z]:", value):
            raise ValueError("asset source must be relative to the task directory")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("asset source contains an unsafe path component")
        return value

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        relative = _workspace_relative(value, "asset destination", allow_root=False)
        return relative.as_posix()


class SandboxSpec(StrictModel):
    image: str = Field(min_length=1, max_length=255)
    task_id: str
    task_version: str
    working_directory: str = "/workspace"
    input_assets: tuple[InputAsset, ...] = ()
    output_directory: str = "/workspace/output"
    environment_allowlist: tuple[str, ...] = ()
    network_policy: NetworkPolicy = NetworkPolicy.DISABLED
    cpu_limit: float = Field(default=1.0, gt=0, le=64)
    memory_limit_bytes: int = Field(default=256 * 1024 * 1024, ge=16 * 1024 * 1024)
    pid_limit: int = Field(default=64, ge=1, le=4096)
    command_timeout_seconds: float = Field(default=30.0, gt=0, le=86_400)
    overall_timeout_seconds: float = Field(default=120.0, gt=0, le=86_400)
    maximum_output_size_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _validate_identifier(value, "task_id")

    @field_validator("task_version")
    @classmethod
    def validate_task_version(cls, value: str) -> str:
        return _validate_identifier(value, "task_version")

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: str) -> str:
        relative = _workspace_relative(value, "working_directory", allow_root=True)
        return "/workspace" if not relative.parts else f"/workspace/{relative.as_posix()}"

    @field_validator("output_directory")
    @classmethod
    def validate_output_directory(cls, value: str) -> str:
        relative = _workspace_relative(value, "output_directory", allow_root=False)
        return f"/workspace/{relative.as_posix()}"

    @field_validator("environment_allowlist")
    @classmethod
    def validate_environment_allowlist(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        deduplicated: list[str] = []
        for value in values:
            if not ENVIRONMENT_NAME.fullmatch(value):
                raise ValueError(f"invalid environment-variable name: {value!r}")
            if is_forbidden_environment_name(value):
                raise ValueError(f"model-provider secret may not be allowlisted: {value}")
            if value not in deduplicated:
                deduplicated.append(value)
        return tuple(deduplicated)

    @model_validator(mode="after")
    def validate_timeout_relationship(self) -> SandboxSpec:
        if self.overall_timeout_seconds < 0.1:
            raise ValueError("overall timeout is too small to create a sandbox")
        return self

    @property
    def output_relative_path(self) -> PurePosixPath:
        return PurePosixPath(self.output_directory).relative_to("/workspace")

    @property
    def working_relative_path(self) -> PurePosixPath:
        return PurePosixPath(self.working_directory).relative_to("/workspace")


class ToolPolicy(StrictModel):
    allowed_tool_names: tuple[str, ...] = (
        "list_files",
        "read_text_file",
        "write_text_file",
        "replace_text",
        "submit_result",
    )
    max_total_tool_calls: int = Field(default=50, ge=1, le=10_000)
    per_tool_call_limits: dict[str, int] = Field(default_factory=dict)
    max_read_size_bytes: int = Field(default=64 * 1024, ge=1, le=100 * 1024 * 1024)
    max_write_size_bytes: int = Field(default=64 * 1024, ge=1, le=100 * 1024 * 1024)
    max_file_list_entries: int = Field(default=200, ge=1, le=100_000)
    shell_timeout_seconds: float = Field(default=10.0, gt=0, le=86_400)
    python_timeout_seconds: float = Field(default=10.0, gt=0, le=86_400)
    max_tool_result_size_bytes: int = Field(default=64 * 1024, ge=256, le=10 * 1024 * 1024)
    shell_execution_allowed: bool = False
    python_execution_allowed: bool = False
    protected_workspace_paths: tuple[str, ...] = (
        ".oneoxygen",
        ".oneoxygen/tool-runtime",
    )

    @field_validator("allowed_tool_names")
    @classmethod
    def validate_allowed_tool_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        deduplicated: list[str] = []
        for value in values:
            _validate_tool_name(value)
            if value not in deduplicated:
                deduplicated.append(value)
        return tuple(deduplicated)

    @field_validator("per_tool_call_limits")
    @classmethod
    def validate_per_tool_call_limits(cls, values: dict[str, int]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for name, limit in values.items():
            _validate_tool_name(name)
            if limit < 1:
                raise ValueError("per-tool call limits must be at least 1")
            normalized[name] = limit
        return dict(sorted(normalized.items()))

    @field_validator("protected_workspace_paths")
    @classmethod
    def validate_protected_workspace_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        deduplicated: list[str] = []
        for value in values:
            relative = _workspace_relative(value, "protected_workspace_paths", allow_root=False)
            normalized = relative.as_posix().rstrip("/")
            if normalized not in deduplicated:
                deduplicated.append(normalized)
        return tuple(deduplicated)


class AgentTaskSpec(StrictModel):
    instruction_file: str = Field(min_length=1, max_length=512)
    system_prompt_file: str | None = Field(default=None, min_length=1, max_length=512)
    system_prompt_version: str = "standard_agent_v1"
    maximum_model_turns: int = Field(default=20, ge=1, le=10_000)
    maximum_provider_requests: int = Field(default=60, ge=1, le=100_000)
    maximum_total_input_tokens: int | None = Field(default=None, ge=1)
    maximum_total_output_tokens: int | None = Field(default=None, ge=1)
    maximum_total_tokens: int | None = Field(default=None, ge=1)
    overall_wall_time_seconds: float = Field(default=600.0, gt=0, le=86_400)
    required_submission: bool = True
    final_text_without_submission: FinalTextBehavior = FinalTextBehavior.INCOMPLETE
    data_classification: DataClassification | None = None

    @field_validator("instruction_file")
    @classmethod
    def validate_instruction_file(cls, value: str) -> str:
        return _task_relative_file(value, "instruction_file")

    @field_validator("system_prompt_file")
    @classmethod
    def validate_system_prompt_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _task_relative_file(value, "system_prompt_file")

    @field_validator("system_prompt_version")
    @classmethod
    def validate_system_prompt_version(cls, value: str) -> str:
        return _validate_identifier(value, "system_prompt_version")

    @model_validator(mode="after")
    def validate_submission_behavior(self) -> AgentTaskSpec:
        if (
            self.required_submission
            and self.final_text_without_submission is FinalTextBehavior.SUCCEED
        ):
            raise ValueError("final text cannot succeed when a submit_result call is required")
        return self


class SandboxTask(StrictModel):
    sandbox: SandboxSpec
    commands: tuple[str, ...] = Field(default=(), max_length=100)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    agent: AgentTaskSpec | None = None

    @field_validator("commands")
    @classmethod
    def validate_commands(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for command in values:
            if not command.strip():
                raise ValueError("commands may not be empty")
            if "\x00" in command:
                raise ValueError("commands may not contain null bytes")
        return values


class ExecResult(StrictModel):
    command: str
    stdout: str
    stderr: str
    exit_code: int
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float = Field(ge=0)
    timed_out: bool = False
    output_truncated: bool = False


class SandboxPolicy(StrictModel):
    network_policy: NetworkPolicy
    non_root_user: str
    read_only_root_filesystem: bool
    writable_mounts: tuple[str, ...]
    tmpfs_mounts: tuple[str, ...]
    dropped_capabilities: tuple[str, ...]
    no_new_privileges: bool
    cpu_limit: float
    memory_limit_bytes: int
    pid_limit: int
    command_timeout_seconds: float
    overall_timeout_seconds: float
    maximum_output_size_bytes: int
    environment_allowlist: tuple[str, ...]

    @classmethod
    def from_spec(cls, spec: SandboxSpec, non_root_user: str = "10001:10001") -> SandboxPolicy:
        return cls(
            network_policy=spec.network_policy,
            non_root_user=non_root_user,
            read_only_root_filesystem=True,
            writable_mounts=("/workspace",),
            tmpfs_mounts=("/tmp",),
            dropped_capabilities=("ALL",),
            no_new_privileges=True,
            cpu_limit=spec.cpu_limit,
            memory_limit_bytes=spec.memory_limit_bytes,
            pid_limit=spec.pid_limit,
            command_timeout_seconds=spec.command_timeout_seconds,
            overall_timeout_seconds=spec.overall_timeout_seconds,
            maximum_output_size_bytes=spec.maximum_output_size_bytes,
            environment_allowlist=spec.environment_allowlist,
        )


class ErrorInformation(StrictModel):
    type: str
    code: str
    message: str


class ArtifactMetadata(StrictModel):
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str


class ToolDefinition(StrictModel):
    name: str
    description: str
    arguments_schema: dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_tool_name(value)

    def to_provider_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ToolCall(StrictModel):
    call_id: str = Field(min_length=1, max_length=128)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    original_index: int | None = Field(default=None, ge=0)

    @field_validator("call_id")
    @classmethod
    def validate_call_id(cls, value: str) -> str:
        return _validate_identifier(value, "call_id")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _validate_tool_name(value)


class ModelToolCallTrace(StrictModel):
    """Bounded, content-redacted representation of a normalized model tool call."""

    call_id: str = Field(min_length=1, max_length=128)
    tool_name: str
    original_index: int = Field(ge=0)
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_sha256: str
    arguments_truncated: bool = False

    @field_validator("call_id")
    @classmethod
    def validate_call_id(cls, value: str) -> str:
        return _validate_identifier(value, "call_id")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _validate_tool_name(value)

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_json_mapping(value, "model tool-call trace arguments")

    @field_validator("arguments_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("arguments_sha256 must be a lowercase SHA-256 digest")
        return value


class ToolError(StrictModel):
    code: ToolErrorCode
    message: str


class ToolResult(StrictModel):
    call_id: str
    tool_name: str
    success: bool
    content: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float = Field(ge=0)
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelErrorInformation(StrictModel):
    code: ModelErrorCode
    message: str = Field(min_length=1, max_length=2_000)
    retryable: bool = False
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        sanitized = sanitize_model_trace_text(value).strip()[:2_000]
        return sanitized or "model provider request failed"

    @field_validator("provider_metadata")
    @classmethod
    def validate_provider_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_json_mapping(value, "model error provider metadata")


class ModelAttempt(StrictModel):
    attempt_number: int = Field(ge=1)
    start_timestamp: datetime
    end_timestamp: datetime
    latency_seconds: float = Field(ge=0)
    succeeded: bool
    error_code: ModelErrorCode | None = None
    retryable: bool = False
    retry_delay_seconds: float | None = Field(default=None, ge=0)
    request_timeout_seconds: float | None = Field(default=None, gt=0, le=3_600)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider_metadata")
    @classmethod
    def validate_provider_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_json_mapping(value, "model attempt provider metadata")

    @model_validator(mode="after")
    def validate_outcome(self) -> ModelAttempt:
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("model attempt end timestamp precedes its start timestamp")
        if self.succeeded and self.error_code is not None:
            raise ValueError("a successful model attempt cannot have an error code")
        if not self.succeeded and self.error_code is None:
            raise ValueError("a failed model attempt must have an error code")
        return self


class ModelTurnRequest(StrictModel):
    turn_number: int = Field(ge=1)
    system_prompt: str = Field(min_length=1, max_length=256_000)
    initial_task_instruction: str = Field(min_length=1, max_length=1_000_000)
    tool_definitions: tuple[ToolDefinition, ...]
    tool_results: tuple[ToolResult, ...] = ()
    run_config: ModelRunConfig
    request_timeout_seconds: float | None = Field(default=None, gt=0, le=3_600)


class ModelTurnResponse(StrictModel):
    response_id: str | None = Field(default=None, min_length=1, max_length=512)
    provider: ModelProvider
    requested_model: str = Field(min_length=1, max_length=256)
    returned_model: str | None = Field(default=None, min_length=1, max_length=256)
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: NormalizedFinishReason = NormalizedFinishReason.UNKNOWN
    usage: ModelUsage = Field(default_factory=ModelUsage)
    latency_seconds: float = Field(default=0, ge=0)
    attempts: tuple[ModelAttempt, ...] = ()
    attempt_count: int = Field(default=1, ge=1)
    warnings: tuple[str, ...] = Field(default=(), max_length=100)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    transport: InferenceTransport | None = None
    api_host: str | None = Field(default=None, min_length=1, max_length=255)
    provenance: ProvenanceClassification | None = None
    official_route: bool | None = None
    upstream_provider_verifiable: bool | None = None
    batch_job_id: str | None = Field(default=None, min_length=1, max_length=512)
    batch_request_id: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        sanitized = tuple(sanitize_model_trace_text(value).strip()[:1_000] for value in values)
        if any(not value for value in sanitized):
            raise ValueError("warnings must be non-empty and at most 1000 characters")
        return sanitized

    @field_validator("provider_metadata")
    @classmethod
    def validate_provider_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_json_mapping(value, "model response provider metadata")

    @model_validator(mode="after")
    def validate_attempt_count(self) -> ModelTurnResponse:
        if self.attempts and self.attempt_count != len(self.attempts):
            raise ValueError("attempt_count must match the number of attempt records")
        if self.provider is ModelProvider.OPENAI:
            object.__setattr__(self, "transport", self.transport or InferenceTransport.DIRECT)
            object.__setattr__(self, "api_host", self.api_host or "api.openai.com")
            object.__setattr__(
                self,
                "provenance",
                self.provenance or ProvenanceClassification.OFFICIAL_PROVIDER,
            )
            object.__setattr__(
                self,
                "upstream_provider_verifiable",
                True
                if self.upstream_provider_verifiable is None
                else self.upstream_provider_verifiable,
            )
        elif self.provider is ModelProvider.AIRFORCE:
            object.__setattr__(
                self, "transport", self.transport or InferenceTransport.GATEWAY_DIRECT
            )
            object.__setattr__(self, "api_host", self.api_host or "api.airforce")
            object.__setattr__(
                self,
                "provenance",
                self.provenance or ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED,
            )
            object.__setattr__(
                self,
                "upstream_provider_verifiable",
                False
                if self.upstream_provider_verifiable is None
                else self.upstream_provider_verifiable,
            )
        else:
            object.__setattr__(self, "transport", self.transport or InferenceTransport.DIRECT)
            object.__setattr__(self, "api_host", self.api_host or "scripted.local")
            object.__setattr__(
                self,
                "provenance",
                self.provenance or ProvenanceClassification.SCRIPTED_TEST,
            )
            object.__setattr__(
                self,
                "upstream_provider_verifiable",
                False
                if self.upstream_provider_verifiable is None
                else self.upstream_provider_verifiable,
            )
        object.__setattr__(
            self,
            "official_route",
            self.provenance is ProvenanceClassification.OFFICIAL_PROVIDER
            if self.official_route is None
            else self.official_route,
        )
        return self


class ModelEvent(StrictModel):
    schema_version: int = 1
    sequence_number: int = Field(ge=1)
    turn_number: int = Field(ge=1)
    provider: ModelProvider
    requested_model: str = Field(min_length=1, max_length=256)
    returned_model: str | None = Field(default=None, min_length=1, max_length=256)
    request_start_timestamp: datetime
    request_end_timestamp: datetime
    latency_seconds: float = Field(ge=0)
    attempt_count: int = Field(ge=1)
    attempts: tuple[ModelAttempt, ...] = ()
    finish_reason: NormalizedFinishReason
    text: str = ""
    text_sha256: str = ""
    text_truncated: bool = False
    tool_calls: tuple[ModelToolCallTrace, ...] = ()
    usage: ModelUsage = Field(default_factory=ModelUsage)
    requested_settings: dict[str, Any]
    effective_settings: dict[str, Any]
    tool_definitions_sha256: str
    prompt_sha256: str
    response_id: str | None = Field(default=None, min_length=1, max_length=512)
    warnings: tuple[str, ...] = Field(default=(), max_length=100)
    error: ModelErrorInformation | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    transport: InferenceTransport = InferenceTransport.DIRECT
    api_host: str = Field(default="scripted.local", min_length=1, max_length=255)
    provenance: ProvenanceClassification = ProvenanceClassification.SCRIPTED_TEST
    official_route: bool = False
    upstream_provider_verifiable: bool = False
    batch_job_id: str | None = Field(default=None, min_length=1, max_length=512)
    batch_request_id: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="before")
    @classmethod
    def bound_and_hash_text(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        updated = dict(value)
        text = updated.get("text", "")
        if not isinstance(text, str):
            return updated
        raw = text.encode("utf-8")
        updated["text_sha256"] = hashlib.sha256(raw).hexdigest()
        sanitized = sanitize_model_trace_text(text).encode("utf-8")
        updated["text"] = sanitized.decode("utf-8")
        if (
            len(raw) > MAXIMUM_MODEL_EVENT_TEXT_BYTES
            or len(sanitized) > MAXIMUM_MODEL_EVENT_TEXT_BYTES
        ):
            updated["text"] = sanitized[:MAXIMUM_MODEL_EVENT_TEXT_BYTES].decode(
                "utf-8", errors="ignore"
            )
            updated["text_truncated"] = True
        return updated

    @field_validator("requested_settings", "effective_settings", "provider_metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_json_mapping(value, "model event metadata")

    @field_validator("tool_definitions_sha256", "prompt_sha256", "text_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("hash values must be lowercase SHA-256 digests")
        return value

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        sanitized = tuple(sanitize_model_trace_text(value).strip()[:1_000] for value in values)
        if any(not value for value in sanitized):
            raise ValueError("warnings must be non-empty and at most 1000 characters")
        return sanitized

    @model_validator(mode="after")
    def validate_event(self) -> ModelEvent:
        if self.request_end_timestamp < self.request_start_timestamp:
            raise ValueError("model event end timestamp precedes its start timestamp")
        if self.attempts and self.attempt_count != len(self.attempts):
            raise ValueError("attempt_count must match the number of attempt records")
        return self


class SubmittedResult(StrictModel):
    summary: str
    artifact_paths: tuple[str, ...]
    findings: dict[str, Any] | None = None
    artifacts: tuple[ArtifactMetadata, ...] = ()


class ToolEvent(StrictModel):
    schema_version: int = 1
    sequence_number: int = Field(ge=1)
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    arguments_sha256: str
    arguments_truncated: bool
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float = Field(ge=0)
    status: ToolEventStatus
    result: dict[str, Any]
    result_sha256: str
    result_truncated: bool
    error_code: ToolErrorCode | None = None


class RunMetrics(StrictModel):
    model_turns: int = Field(default=0, ge=0)
    provider_attempts: int = Field(default=0, ge=0)
    successful_tool_calls: int = Field(default=0, ge=0)
    failed_tool_calls: int = Field(default=0, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_reasoning_tokens: int | None = Field(default=None, ge=0)
    total_cached_input_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    total_model_latency_seconds: float = Field(default=0, ge=0)
    total_tool_latency_seconds: float = Field(default=0, ge=0)
    total_wall_time_seconds: float = Field(default=0, ge=0)
    token_usage_incomplete: bool = False

    @classmethod
    def aggregate(
        cls,
        model_events: tuple[ModelEvent, ...] | list[ModelEvent],
        tool_events: tuple[ToolEvent, ...] | list[ToolEvent],
        *,
        total_wall_time_seconds: float = 0,
    ) -> RunMetrics:
        model_items = tuple(model_events)
        tool_items = tuple(tool_events)
        usage = ModelUsage.aggregate([event.usage for event in model_items])
        return cls(
            model_turns=len(model_items),
            provider_attempts=sum(event.attempt_count for event in model_items),
            successful_tool_calls=sum(
                event.status is ToolEventStatus.SUCCEEDED for event in tool_items
            ),
            failed_tool_calls=sum(event.status is ToolEventStatus.FAILED for event in tool_items),
            total_input_tokens=usage.input_tokens,
            total_output_tokens=usage.output_tokens,
            total_reasoning_tokens=usage.reasoning_tokens,
            total_cached_input_tokens=usage.cached_input_tokens,
            total_tokens=usage.total_tokens,
            total_model_latency_seconds=sum(event.latency_seconds for event in model_items),
            total_tool_latency_seconds=sum(event.duration_seconds for event in tool_items),
            total_wall_time_seconds=total_wall_time_seconds,
            token_usage_incomplete=bool(model_items)
            and any(
                event.usage.input_tokens is None
                or event.usage.output_tokens is None
                or event.usage.total_tokens is None
                for event in model_items
            ),
        )


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_schema_version: int = 3
    run_id: str
    task_id: str
    task_version: str
    requested_image: str
    resolved_image: str | None = None
    task_configuration_hash: str
    start_timestamp: datetime
    end_timestamp: datetime | None = None
    sandbox_policy: SandboxPolicy
    command_results: list[ExecResult] = Field(default_factory=list)
    tool_policy: ToolPolicy | None = None
    tool_events: list[ToolEvent] = Field(default_factory=list)
    submission: SubmittedResult | None = None
    model_configuration: ModelRunConfig | None = None
    effective_model_settings: dict[str, Any] = Field(default_factory=dict)
    model_events: list[ModelEvent] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    termination_reason: AgentTerminationReason | None = None
    system_prompt_version: str | None = None
    system_prompt_sha256: str | None = None
    system_prompt_content: str | None = Field(default=None, max_length=64_000)
    task_instruction_sha256: str | None = None
    final_status: RunStatus = RunStatus.RUNNING
    error: ErrorInformation | None = None
    artifacts: list[ArtifactMetadata] = Field(default_factory=list)
    inference_transport: InferenceTransport | None = None
    provenance: ProvenanceClassification | None = None
    logical_provider: ModelProvider | None = None
    requested_model: str | None = None
    returned_models: list[str] = Field(default_factory=list)
    api_host: str | None = None
    official_route: bool | None = None
    upstream_provider_verifiable: bool | None = None
    experiment_namespace: str | None = None
    batch_job_ids: list[str] = Field(default_factory=list)
    batch_request_ids: list[str] = Field(default_factory=list)


def _validate_tool_name(value: str) -> str:
    if not re.fullmatch(r"^[a-z][a-z0-9_]{0,63}$", value):
        raise ValueError("tool names must use lowercase letters, numbers, and underscores")
    return value
