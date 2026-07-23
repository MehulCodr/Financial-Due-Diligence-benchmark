"""Policy-checking dispatcher for normalized tool calls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from oneoxygen_sandbox.errors import SandboxError, ToolFailure
from oneoxygen_sandbox.filesystem import SecureWorkspace
from oneoxygen_sandbox.models import ToolCall, ToolErrorCode, ToolPolicy, ToolResult
from oneoxygen_sandbox.session import SandboxSession
from oneoxygen_sandbox.tools.base import (
    SubmissionState,
    ToolContext,
    bounded_json_value,
    failed_result,
    monotonic,
    now_utc,
    redact_arguments,
    result_to_event,
    submission_from_content,
    successful_result,
)
from oneoxygen_sandbox.tools.registry import ToolRegistry, default_tool_registry


class ToolDispatcher:
    def __init__(
        self,
        session: SandboxSession,
        registry: ToolRegistry | None = None,
        policy: ToolPolicy | None = None,
        initial_total_count: int = 0,
        initial_tool_counts: Mapping[str, int] | None = None,
        initial_sequence_number: int = 0,
    ) -> None:
        self.session = session
        self.registry = registry or default_tool_registry()
        self.policy = policy or session.task.tool_policy
        try:
            submitted = session.record.submission is not None
        except SandboxError:
            submitted = False
        self.submission_state = SubmissionState(submitted=submitted)
        if initial_total_count < 0 or initial_sequence_number < 0:
            raise ValueError("initial dispatcher counters cannot be negative")
        self._total_count = initial_total_count
        self._tool_counts: Counter[str] = Counter(initial_tool_counts or {})
        if any(value < 0 for value in self._tool_counts.values()):
            raise ValueError("initial per-tool counters cannot be negative")
        self._sequence_number = initial_sequence_number

    def dispatch(self, call: ToolCall) -> ToolResult:
        self._sequence_number += 1
        start_timestamp = now_utc()
        start_monotonic = monotonic()
        raw_arguments = redact_arguments(call.tool_name, call.arguments)
        result: ToolResult
        try:
            # Every provider-returned call consumes the deterministic budget,
            # including unknown, disallowed, and invalid calls.
            self._increment_counters(call.tool_name)
            if self.submission_state.submitted:
                raise ToolFailure(
                    ToolErrorCode.ALREADY_SUBMITTED.value,
                    "a result has already been submitted",
                )
            if not self.session.is_active or self.session.workspace_path is None:
                raise ToolFailure(
                    ToolErrorCode.INTERNAL_TOOL_ERROR.value,
                    "sandbox session is not active",
                )
            tool = self.registry.resolve(call.tool_name)
            self._enforce_policy(tool.name)
            arguments = tool.argument_model.model_validate(call.arguments)
            context = ToolContext(
                session=self.session,
                workspace=self._workspace(self.session.workspace_path),
                policy=self.policy,
                submission_state=self.submission_state,
            )
            content = tool.execute(arguments, context)
            result = successful_result(
                call_id=call.call_id,
                tool_name=call.tool_name,
                content=content,
                start_timestamp=start_timestamp,
                start_monotonic=start_monotonic,
                maximum_result_bytes=self.policy.max_tool_result_size_bytes,
            )
            if tool.name == "submit_result" and result.success:
                self.session.record.submission = submission_from_content(content)
        except ValidationError:
            result = failed_result(
                call_id=call.call_id,
                tool_name=call.tool_name,
                code=ToolErrorCode.INVALID_ARGUMENTS,
                message="tool arguments are invalid",
                start_timestamp=start_timestamp,
                start_monotonic=start_monotonic,
                maximum_result_bytes=self.policy.max_tool_result_size_bytes,
            )
        except ToolFailure as exc:
            result = failed_result(
                call_id=call.call_id,
                tool_name=call.tool_name,
                code=self._coerce_error_code(exc.tool_code),
                message=exc.message,
                start_timestamp=start_timestamp,
                start_monotonic=start_monotonic,
                maximum_result_bytes=self.policy.max_tool_result_size_bytes,
                content=exc.content,
                metadata=exc.metadata,
                truncated=exc.truncated,
            )
        except SandboxError:
            result = failed_result(
                call_id=call.call_id,
                tool_name=call.tool_name,
                code=ToolErrorCode.INTERNAL_TOOL_ERROR,
                message="tool failed inside the sandbox",
                start_timestamp=start_timestamp,
                start_monotonic=start_monotonic,
                maximum_result_bytes=self.policy.max_tool_result_size_bytes,
            )
        except Exception:
            result = failed_result(
                call_id=call.call_id,
                tool_name=call.tool_name,
                code=ToolErrorCode.INTERNAL_TOOL_ERROR,
                message="tool failed internally",
                start_timestamp=start_timestamp,
                start_monotonic=start_monotonic,
                maximum_result_bytes=self.policy.max_tool_result_size_bytes,
            )
        self._record_event(result, raw_arguments)
        return result

    def reject(self, call: ToolCall, code: ToolErrorCode, message: str) -> ToolResult:
        """Record a provider-returned call that orchestration can no longer execute."""
        self._sequence_number += 1
        self._increment_counters(call.tool_name)
        start_timestamp = now_utc()
        start_monotonic = monotonic()
        raw_arguments = redact_arguments(call.tool_name, call.arguments)
        result = failed_result(
            call_id=call.call_id,
            tool_name=call.tool_name,
            code=code,
            message=message,
            start_timestamp=start_timestamp,
            start_monotonic=start_monotonic,
            maximum_result_bytes=self.policy.max_tool_result_size_bytes,
        )
        self._record_event(result, raw_arguments)
        return result

    @property
    def required_submission_is_reachable(self) -> bool:
        """Whether at least one further submit_result attempt remains."""
        if self.submission_state.submitted:
            return True
        if self._total_count >= self.policy.max_total_tool_calls:
            return False
        submit_limit = self.policy.per_tool_call_limits.get("submit_result")
        return submit_limit is None or self._tool_counts["submit_result"] < submit_limit

    def definitions(self) -> tuple[dict[str, Any], ...]:
        return tuple(definition.to_provider_dict() for definition in self.registry.definitions())

    def counter_snapshot(self) -> tuple[int, dict[str, int], int]:
        return self._total_count, dict(self._tool_counts), self._sequence_number

    def _workspace(self, root: Path) -> SecureWorkspace:
        return SecureWorkspace(
            root,
            self.session.spec.output_relative_path,
            self.policy.protected_workspace_paths,
        )

    def _enforce_policy(self, tool_name: str) -> None:
        if tool_name not in self.policy.allowed_tool_names:
            raise ToolFailure(ToolErrorCode.TOOL_NOT_ALLOWED.value, "tool is not allowed")
        if tool_name == "execute_shell" and not self.policy.shell_execution_allowed:
            raise ToolFailure(
                ToolErrorCode.TOOL_NOT_ALLOWED.value,
                "shell execution is not allowed",
            )
        if tool_name == "execute_python" and not self.policy.python_execution_allowed:
            raise ToolFailure(
                ToolErrorCode.TOOL_NOT_ALLOWED.value,
                "Python execution is not allowed",
            )
        if self._total_count > self.policy.max_total_tool_calls:
            raise ToolFailure(
                ToolErrorCode.CALL_LIMIT_EXCEEDED.value,
                "maximum total tool calls exceeded",
            )
        per_tool_limit = self.policy.per_tool_call_limits.get(tool_name)
        if per_tool_limit is not None and self._tool_counts[tool_name] > per_tool_limit:
            raise ToolFailure(
                ToolErrorCode.CALL_LIMIT_EXCEEDED.value,
                "per-tool call limit exceeded",
            )

    def _increment_counters(self, tool_name: str) -> None:
        self._total_count += 1
        self._tool_counts[tool_name] += 1

    def _record_event(self, result: ToolResult, raw_arguments: dict[str, Any]) -> None:
        argument_limit = min(4096, self.policy.max_tool_result_size_bytes)
        _bounded, _digest, _truncated = bounded_json_value(raw_arguments, argument_limit)
        event = result_to_event(
            sequence_number=self._sequence_number,
            result=result,
            raw_arguments=raw_arguments,
            argument_limit=argument_limit,
            result_limit=self.policy.max_tool_result_size_bytes,
        )
        self.session.record.tool_events.append(event)

    def _coerce_error_code(self, value: str) -> ToolErrorCode:
        try:
            return ToolErrorCode(value)
        except ValueError:
            return ToolErrorCode.INTERNAL_TOOL_ERROR
