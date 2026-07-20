"""Validated configuration and run-record models."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FORBIDDEN_SECRET_NAMES = {
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "COHERE_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "TOGETHER_API_KEY",
}
MODEL_PROVIDER_MARKERS = (
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
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NetworkPolicy(StrEnum):
    DISABLED = "disabled"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"


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


class SandboxTask(StrictModel):
    sandbox: SandboxSpec
    commands: tuple[str, ...] = Field(default=(), max_length=100)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)

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

    @field_validator("call_id")
    @classmethod
    def validate_call_id(cls, value: str) -> str:
        return _validate_identifier(value, "call_id")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _validate_tool_name(value)


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


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_schema_version: int = 2
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
    final_status: RunStatus = RunStatus.RUNNING
    error: ErrorInformation | None = None
    artifacts: list[ArtifactMetadata] = Field(default_factory=list)


def _validate_tool_name(value: str) -> str:
    if not re.fullmatch(r"^[a-z][a-z0-9_]{0,63}$", value):
        raise ValueError("tool names must use lowercase letters, numbers, and underscores")
    return value
