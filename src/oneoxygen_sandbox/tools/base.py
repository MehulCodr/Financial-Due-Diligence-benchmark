"""Small protocol and shared helpers for provider-neutral tools."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from oneoxygen_sandbox.filesystem import SecureWorkspace
from oneoxygen_sandbox.models import (
    SubmittedResult,
    ToolDefinition,
    ToolError,
    ToolErrorCode,
    ToolEvent,
    ToolEventStatus,
    ToolPolicy,
    ToolResult,
)
from oneoxygen_sandbox.session import SandboxSession


class Tool(Protocol):
    name: str
    description: str
    argument_model: type[BaseModel]

    def definition(self) -> ToolDefinition: ...

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]: ...


@dataclass
class SubmissionState:
    submitted: bool = False


@dataclass
class ToolContext:
    session: SandboxSession
    workspace: SecureWorkspace
    policy: ToolPolicy
    submission_state: SubmissionState


class BaseTool:
    name: str
    description: str
    argument_model: type[BaseModel]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            arguments_schema=self.argument_model.model_json_schema(),
        )


def now_utc() -> datetime:
    return datetime.now(UTC)


def monotonic() -> float:
    return time.monotonic()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bounded_json_value(value: Any, maximum_bytes: int) -> tuple[dict[str, Any], str, bool]:
    digest = sha256_json(value)
    raw = canonical_json_bytes(value)
    if len(raw) <= maximum_bytes:
        if isinstance(value, dict):
            return value, digest, False
        return {"value": value}, digest, False
    preview = raw[: max(0, maximum_bytes - 64)].decode("utf-8", errors="replace")
    return {"truncated_json_preview": preview}, digest, True


def redact_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(arguments)
    for key in ("content", "old_text", "replacement_text", "source_code"):
        if key in redacted and isinstance(redacted[key], str):
            value = redacted[key].encode("utf-8")
            redacted[key] = {
                "sha256": hashlib.sha256(value).hexdigest(),
                "size_bytes": len(value),
            }
    if tool_name == "submit_result" and isinstance(redacted.get("findings"), dict):
        redacted["findings"] = {"keys": sorted(str(key) for key in redacted["findings"])}
    return redacted


def successful_result(
    *,
    call_id: str,
    tool_name: str,
    content: dict[str, Any],
    start_timestamp: datetime,
    start_monotonic: float,
    maximum_result_bytes: int,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    bounded_content, content_sha, truncated = bounded_json_value(content, maximum_result_bytes)
    tool_reported_truncation = bool(content.get("truncated"))
    end_timestamp = now_utc()
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        success=True,
        content=bounded_content,
        error=None,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        duration_seconds=monotonic() - start_monotonic,
        truncated=truncated or tool_reported_truncation,
        metadata={"content_sha256": content_sha, **(metadata or {})},
    )


def failed_result(
    *,
    call_id: str,
    tool_name: str,
    code: ToolErrorCode,
    message: str,
    start_timestamp: datetime,
    start_monotonic: float,
    maximum_result_bytes: int,
    content: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    truncated: bool = False,
) -> ToolResult:
    bounded_content, content_sha, content_truncated = bounded_json_value(
        content or {}, maximum_result_bytes
    )
    end_timestamp = now_utc()
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        success=False,
        content=bounded_content,
        error=ToolError(code=code, message=message),
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        duration_seconds=monotonic() - start_monotonic,
        truncated=truncated or content_truncated,
        metadata={"content_sha256": content_sha, **(metadata or {})},
    )


def result_to_event(
    *,
    sequence_number: int,
    result: ToolResult,
    raw_arguments: dict[str, Any],
    argument_limit: int,
    result_limit: int,
) -> ToolEvent:
    bounded_arguments, arguments_sha, arguments_truncated = bounded_json_value(
        raw_arguments, argument_limit
    )
    result_payload = result.model_dump(mode="json")
    bounded_result, result_sha, result_truncated = bounded_json_value(result_payload, result_limit)
    return ToolEvent(
        sequence_number=sequence_number,
        call_id=result.call_id,
        tool_name=result.tool_name,
        arguments=bounded_arguments,
        arguments_sha256=arguments_sha,
        arguments_truncated=arguments_truncated,
        start_timestamp=result.start_timestamp,
        end_timestamp=result.end_timestamp,
        duration_seconds=result.duration_seconds,
        status=ToolEventStatus.SUCCEEDED if result.success else ToolEventStatus.FAILED,
        result=bounded_result,
        result_sha256=result_sha,
        result_truncated=result.truncated or result_truncated,
        error_code=result.error.code if result.error else None,
    )


def submission_from_content(content: dict[str, Any]) -> SubmittedResult:
    return SubmittedResult.model_validate(
        {
            "summary": content["summary"],
            "artifact_paths": tuple(content["artifact_paths"]),
            "findings": content.get("findings"),
            "artifacts": tuple(content["artifacts"]),
        }
    )
