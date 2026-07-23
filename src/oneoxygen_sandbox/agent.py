"""Deterministic provider-neutral agent orchestration."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oneoxygen_sandbox.docker_adapter import DockerAdapter
from oneoxygen_sandbox.errors import ConfigurationError, ModelError, SandboxError
from oneoxygen_sandbox.model_adapters.base import ModelAdapter
from oneoxygen_sandbox.models import (
    AgentTerminationReason,
    ErrorInformation,
    FinalTextBehavior,
    ModelAttempt,
    ModelErrorCode,
    ModelEvent,
    ModelRunConfig,
    ModelToolCallTrace,
    ModelTurnRequest,
    ModelTurnResponse,
    NormalizedFinishReason,
    ProvenanceClassification,
    RunMetrics,
    RunRecord,
    RunStatus,
    SandboxTask,
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolResult,
)
from oneoxygen_sandbox.orchestration import AgentRunStatus, AgentStateMachine
from oneoxygen_sandbox.session import SandboxSession
from oneoxygen_sandbox.tools import ToolDispatcher, ToolRegistry, default_tool_registry
from oneoxygen_sandbox.tools.base import bounded_json_value, redact_arguments

_MAXIMUM_INSTRUCTION_BYTES = 1_000_000
_MAXIMUM_PROMPT_BYTES = 256_000
_MAXIMUM_RETRY_DELAY_SECONDS = 60.0
_TRANSIENT_MODEL_ERRORS = {
    ModelErrorCode.RATE_LIMITED,
    ModelErrorCode.REQUEST_TIMEOUT,
    ModelErrorCode.PROVIDER_UNAVAILABLE,
}


@dataclass(frozen=True)
class _AttemptFailure(Exception):
    error: ModelError
    attempts: tuple[ModelAttempt, ...]
    request_started_at: datetime
    request_ended_at: datetime
    provider_request_limit_prevented_retry: bool = False


class AgentRunner:
    """Own the model/tool loop while adapters retain provider-specific state."""

    def __init__(
        self,
        task: SandboxTask,
        task_directory: Path,
        model_config: ModelRunConfig,
        model_adapter: ModelAdapter,
        *,
        runs_directory: Path = Path("runs"),
        sandbox_adapter: DockerAdapter | None = None,
        tool_registry: ToolRegistry | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        if task.agent is None:
            raise ConfigurationError("agent-run requires an agent section in the task")
        self.task = task
        self.agent_spec = task.agent
        self.task_directory = task_directory.resolve()
        self.requested_model_config = model_config
        self.model_adapter = model_adapter
        self.model_config = model_adapter.validate_config(model_config)
        if self.model_config.provider is not model_adapter.provider:
            raise ConfigurationError("model configuration does not match the selected provider")
        self.runs_directory = runs_directory
        self.sandbox_adapter = sandbox_adapter
        self.tool_registry = tool_registry or default_tool_registry()
        self.clock = clock
        self.wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self.sleeper = sleeper
        self.jitter = jitter or (lambda delay: random.uniform(0.0, min(1.0, max(0.0, delay * 0.1))))
        self.session: SandboxSession | None = None
        self._run_started = 0.0
        self._provider_attempts = 0
        self._system_prompt = ""
        self._instruction = ""
        self._prompt_hash = ""
        self._instruction_hash = ""
        self._tool_definitions: tuple[ToolDefinition, ...] = ()
        self._tool_definitions_hash = ""
        self._seen_tool_call_ids: set[str] = set()
        self._state_machine = AgentStateMachine()

    @property
    def agent_state(self) -> AgentRunStatus:
        return self._state_machine.status

    @property
    def record_path(self) -> Path | None:
        return self.session.record_path if self.session is not None else None

    def run(self) -> RunRecord:
        """Run to one explicit terminal state and always finalize an opened session."""
        self._prepare_inputs()
        self._run_started = self.clock()
        session = SandboxSession(
            self.task,
            self.task_directory,
            self.runs_directory,
            self.sandbox_adapter,
        )
        self.session = session
        outcome_status = RunStatus.INTERNAL_ERROR
        outcome_reason = AgentTerminationReason.INTERNAL_ORCHESTRATION_ERROR
        outcome_error: BaseException | None = None
        should_collect_submission = False
        adapter_started = False

        try:
            session.start()
            self._initialize_record(session.record)
            self._state_machine.transition(AgentRunStatus.READY_FOR_MODEL)
            dispatcher = ToolDispatcher(session, registry=self.tool_registry)
            previous_results: tuple[ToolResult, ...] = ()
            first_request = self._request(1, previous_results)
            self.model_adapter.start_conversation(first_request)
            adapter_started = True

            for turn_number in range(1, self.agent_spec.maximum_model_turns + 1):
                if self._state_machine.status is AgentRunStatus.READY_FOR_NEXT_TURN:
                    self._state_machine.transition(AgentRunStatus.READY_FOR_MODEL)
                if self._wall_time_exceeded():
                    outcome_status = RunStatus.LIMIT_EXCEEDED
                    outcome_reason = AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
                    break
                if self._provider_attempts >= self.agent_spec.maximum_provider_requests:
                    outcome_status = RunStatus.LIMIT_EXCEEDED
                    outcome_reason = AgentTerminationReason.MAXIMUM_PROVIDER_REQUESTS_REACHED
                    break

                request = self._request(turn_number, previous_results)
                self._state_machine.transition(AgentRunStatus.WAITING_FOR_MODEL)
                try:
                    response, attempts, request_start, request_end = self._generate_with_retries(
                        request
                    )
                except _AttemptFailure as failure:
                    self._state_machine.transition(AgentRunStatus.FAILED)
                    self._record_failed_model_event(
                        turn_number,
                        failure,
                        request_timeout_seconds=(
                            failure.attempts[-1].request_timeout_seconds
                            if failure.attempts
                            else request.request_timeout_seconds
                        ),
                    )
                    outcome_error = failure.error
                    if self._wall_time_exceeded():
                        outcome_status = RunStatus.LIMIT_EXCEEDED
                        outcome_reason = AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
                    elif failure.provider_request_limit_prevented_retry:
                        outcome_status = RunStatus.LIMIT_EXCEEDED
                        outcome_reason = AgentTerminationReason.MAXIMUM_PROVIDER_REQUESTS_REACHED
                    elif failure.error.model_code is ModelErrorCode.MODEL_REFUSAL:
                        outcome_status = RunStatus.REFUSED
                        outcome_reason = AgentTerminationReason.MODEL_REFUSAL
                    elif failure.error.model_code is ModelErrorCode.CONTEXT_LIMIT_EXCEEDED:
                        outcome_status = RunStatus.LIMIT_EXCEEDED
                        outcome_reason = AgentTerminationReason.CONTEXT_LIMIT_EXCEEDED
                    elif failure.error.model_code is ModelErrorCode.CANCELLED:
                        outcome_status = RunStatus.CANCELLED
                        outcome_reason = AgentTerminationReason.USER_INTERRUPTION
                    else:
                        outcome_status = RunStatus.PROVIDER_ERROR
                        outcome_reason = AgentTerminationReason.REPEATED_PROVIDER_FAILURE
                    break

                response = self._normalize_response_indexes(response, attempts)
                self._state_machine.transition(AgentRunStatus.MODEL_RESULT_READY)
                usage_warning = self._usage_enforcement_warning(response)
                self._record_model_event(
                    turn_number,
                    response,
                    request_start,
                    request_end,
                    extra_warning=usage_warning,
                    request_timeout_seconds=attempts[-1].request_timeout_seconds,
                )

                if self._wall_time_exceeded():
                    outcome_status = RunStatus.LIMIT_EXCEEDED
                    outcome_reason = AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
                    break
                if not session.is_active:
                    outcome_status = RunStatus.SANDBOX_ERROR
                    outcome_reason = AgentTerminationReason.SANDBOX_FAILURE
                    outcome_error = SandboxError("the sandbox timed out during model execution")
                    break

                if response.finish_reason is NormalizedFinishReason.REFUSED:
                    outcome_status = RunStatus.REFUSED
                    outcome_reason = AgentTerminationReason.MODEL_REFUSAL
                    break
                if response.finish_reason is NormalizedFinishReason.CONTENT_FILTER:
                    outcome_status = RunStatus.REFUSED
                    outcome_reason = AgentTerminationReason.MODEL_REFUSAL
                    break
                if response.finish_reason is NormalizedFinishReason.CANCELLED:
                    outcome_status = RunStatus.CANCELLED
                    outcome_reason = AgentTerminationReason.USER_INTERRUPTION
                    break
                if response.finish_reason is NormalizedFinishReason.PROVIDER_ERROR:
                    outcome_status = RunStatus.PROVIDER_ERROR
                    outcome_reason = AgentTerminationReason.REPEATED_PROVIDER_FAILURE
                    outcome_error = ModelError(
                        ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                        "the provider returned a failed response",
                        provider_metadata=response.provider_metadata,
                    )
                    break

                duplicate_id = self._duplicate_call_id(response.tool_calls)
                if duplicate_id is not None:
                    outcome_error = ModelError(
                        ModelErrorCode.DUPLICATE_TOOL_CALL_ID,
                        "the provider returned duplicate tool-call identifiers",
                    )
                    outcome_status = RunStatus.PROVIDER_ERROR
                    outcome_reason = AgentTerminationReason.REPEATED_PROVIDER_FAILURE
                    break

                limit_reason = self._token_limit_reason()
                if limit_reason is not None:
                    for call in response.tool_calls:
                        dispatcher.reject(
                            call,
                            ToolErrorCode.CALL_LIMIT_EXCEEDED,
                            "a cumulative model-token limit was reached",
                        )
                    outcome_status = RunStatus.LIMIT_EXCEEDED
                    outcome_reason = limit_reason
                    break

                if response.finish_reason is NormalizedFinishReason.LENGTH:
                    for call in response.tool_calls:
                        dispatcher.reject(
                            call,
                            ToolErrorCode.CALL_LIMIT_EXCEEDED,
                            "the provider output-token limit was reached",
                        )
                    outcome_status = RunStatus.LIMIT_EXCEEDED
                    outcome_reason = AgentTerminationReason.OUTPUT_TOKEN_LIMIT_REACHED
                    break

                results: list[ToolResult] = []
                if response.tool_calls:
                    self._state_machine.transition(AgentRunStatus.EXECUTING_TOOLS)
                wall_time_reached_during_tools = False
                for call in response.tool_calls:
                    if wall_time_reached_during_tools or self._wall_time_exceeded():
                        wall_time_reached_during_tools = True
                        results.append(
                            dispatcher.reject(
                                call,
                                ToolErrorCode.CALL_LIMIT_EXCEEDED,
                                "the agent wall-time limit was reached",
                            )
                        )
                        continue
                    results.append(dispatcher.dispatch(call))
                    if self._wall_time_exceeded():
                        wall_time_reached_during_tools = True
                previous_results = tuple(results)

                if wall_time_reached_during_tools:
                    outcome_status = RunStatus.LIMIT_EXCEEDED
                    outcome_reason = AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
                    break

                if response.tool_calls and not session.is_active:
                    outcome_status = RunStatus.SANDBOX_ERROR
                    outcome_reason = AgentTerminationReason.TOOL_FAILURE
                    outcome_error = SandboxError("the sandbox became unavailable during tool use")
                    break

                if dispatcher.submission_state.submitted:
                    self._state_machine.transition(AgentRunStatus.SUBMITTED)
                    self._state_machine.transition(AgentRunStatus.COMPLETED)
                    outcome_status = RunStatus.SUCCEEDED
                    outcome_reason = AgentTerminationReason.SUCCESSFUL_SUBMISSION
                    should_collect_submission = True
                    break

                if (
                    self.agent_spec.required_submission
                    and not dispatcher.required_submission_is_reachable
                ):
                    outcome_status = RunStatus.INCOMPLETE
                    outcome_reason = AgentTerminationReason.TOOL_FAILURE
                    break

                if not response.tool_calls:
                    outcome_reason = AgentTerminationReason.FINAL_TEXT_WITHOUT_SUBMISSION
                    if (
                        not self.agent_spec.required_submission
                        and self.agent_spec.final_text_without_submission
                        is FinalTextBehavior.SUCCEED
                    ):
                        outcome_status = RunStatus.SUCCEEDED
                        self._state_machine.transition(AgentRunStatus.COMPLETED)
                    else:
                        outcome_status = RunStatus.INCOMPLETE
                        self._state_machine.transition(AgentRunStatus.INCOMPLETE)
                    break

                if turn_number == self.agent_spec.maximum_model_turns:
                    outcome_status = RunStatus.LIMIT_EXCEEDED
                    outcome_reason = AgentTerminationReason.MAXIMUM_TURNS_REACHED
                    break
                self._state_machine.transition(AgentRunStatus.READY_FOR_NEXT_TURN)
            else:  # pragma: no cover - the bounded range always terminates in-loop
                outcome_status = RunStatus.LIMIT_EXCEEDED
                outcome_reason = AgentTerminationReason.MAXIMUM_TURNS_REACHED
        except KeyboardInterrupt as exc:
            outcome_status = RunStatus.CANCELLED
            outcome_reason = AgentTerminationReason.USER_INTERRUPTION
            outcome_error = exc
        except ModelError as exc:
            outcome_status = (
                RunStatus.REFUSED
                if exc.model_code is ModelErrorCode.MODEL_REFUSAL
                else RunStatus.PROVIDER_ERROR
            )
            outcome_reason = (
                AgentTerminationReason.MODEL_REFUSAL
                if exc.model_code is ModelErrorCode.MODEL_REFUSAL
                else AgentTerminationReason.REPEATED_PROVIDER_FAILURE
            )
            outcome_error = exc
        except SandboxError as exc:
            outcome_status = RunStatus.SANDBOX_ERROR
            outcome_reason = AgentTerminationReason.SANDBOX_FAILURE
            outcome_error = exc
        except Exception as exc:
            outcome_status = RunStatus.INTERNAL_ERROR
            outcome_reason = AgentTerminationReason.INTERNAL_ORCHESTRATION_ERROR
            outcome_error = exc
        finally:
            close_error: ModelError | None = None
            try:
                self.model_adapter.close()
            except ModelError as exc:
                close_error = exc
            except Exception:
                close_error = ModelError(
                    ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                    "the model adapter failed while closing",
                )
            if (
                close_error is not None
                and adapter_started
                and outcome_status
                in {
                    RunStatus.SUCCEEDED,
                    RunStatus.INCOMPLETE,
                    RunStatus.REFUSED,
                }
            ):
                outcome_status = RunStatus.PROVIDER_ERROR
                outcome_reason = AgentTerminationReason.REPEATED_PROVIDER_FAILURE
                outcome_error = close_error
                should_collect_submission = False

            if session._record is not None:  # the record exists before container creation
                if session.record.model_configuration is None:
                    self._initialize_record(session.record)
                self._apply_outcome(
                    session.record,
                    outcome_status,
                    outcome_reason,
                    outcome_error,
                )
                if should_collect_submission and outcome_status is RunStatus.SUCCEEDED:
                    try:
                        session.collect_submitted_artifacts()
                    except SandboxError as exc:
                        outcome_status = RunStatus.SANDBOX_ERROR
                        outcome_reason = AgentTerminationReason.SANDBOX_FAILURE
                        outcome_error = exc
                        self._apply_outcome(
                            session.record,
                            outcome_status,
                            outcome_reason,
                            outcome_error,
                        )
                session.record.metrics = RunMetrics.aggregate(
                    session.record.model_events,
                    session.record.tool_events,
                    total_wall_time_seconds=max(0.0, self.clock() - self._run_started),
                )
            try:
                session.stop()
            except SandboxError as exc:
                if session._record is not None:
                    self._apply_outcome(
                        session.record,
                        RunStatus.SANDBOX_ERROR,
                        AgentTerminationReason.SANDBOX_FAILURE,
                        exc,
                    )
                    session.record.metrics = RunMetrics.aggregate(
                        session.record.model_events,
                        session.record.tool_events,
                        total_wall_time_seconds=max(0.0, self.clock() - self._run_started),
                    )
                    session.stop()

        return session.record

    def _prepare_inputs(self) -> None:
        self._instruction = self._read_task_file(
            self.agent_spec.instruction_file,
            label="agent instruction",
            maximum_bytes=_MAXIMUM_INSTRUCTION_BYTES,
        )
        if self.agent_spec.system_prompt_file is not None:
            self._system_prompt = self._read_task_file(
                self.agent_spec.system_prompt_file,
                label="system prompt",
                maximum_bytes=_MAXIMUM_PROMPT_BYTES,
            )
        elif self.agent_spec.system_prompt_version == "standard_agent_v1":
            prompt_path = Path(__file__).with_name("prompts") / "standard_agent_v1.txt"
            try:
                self._system_prompt = prompt_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigurationError("the standard system prompt is unavailable") from exc
        else:
            raise ConfigurationError("the requested built-in system prompt version is unavailable")
        if not self._instruction.strip():
            raise ConfigurationError("agent instruction file may not be empty")
        if not self._system_prompt.strip():
            raise ConfigurationError("system prompt may not be empty")
        self._instruction_hash = self._sha256_text(self._instruction)
        self._prompt_hash = self._sha256_text(self._system_prompt)
        self._tool_definitions = self._available_tool_definitions()
        if self.agent_spec.required_submission and not any(
            definition.name == "submit_result" for definition in self._tool_definitions
        ):
            raise ConfigurationError("required submission needs the submit_result tool")
        schemas = [definition.model_dump(mode="json") for definition in self._tool_definitions]
        self._tool_definitions_hash = hashlib.sha256(self._canonical_json(schemas)).hexdigest()

    def _read_task_file(self, relative_name: str, *, label: str, maximum_bytes: int) -> str:
        root = self.task_directory.resolve(strict=True)
        candidate = root.joinpath(*Path(relative_name).parts)
        current = root
        try:
            for part in Path(relative_name).parts:
                current = current / part
                if current.is_symlink():
                    raise ConfigurationError(f"{label} may not be a symbolic link")
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file():
                raise ConfigurationError(f"{label} must be a regular file")
            data = resolved.read_bytes()
        except ConfigurationError:
            raise
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot safely read {label}") from exc
        if len(data) > maximum_bytes:
            raise ConfigurationError(f"{label} exceeds its size limit")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigurationError(f"{label} must be UTF-8 text") from exc

    def _available_tool_definitions(self) -> tuple[ToolDefinition, ...]:
        policy = self.task.tool_policy
        definitions: list[ToolDefinition] = []
        for definition in self.tool_registry.definitions():
            if definition.name not in policy.allowed_tool_names:
                continue
            if definition.name == "execute_shell" and not policy.shell_execution_allowed:
                continue
            if definition.name == "execute_python" and not policy.python_execution_allowed:
                continue
            definitions.append(definition)
        return tuple(definitions)

    def _initialize_record(self, record: RunRecord) -> None:
        record.model_configuration = self.requested_model_config
        record.effective_model_settings = self.model_config.requested_settings()
        record.inference_transport = self.model_config.transport
        record.provenance = self.model_config.provenance
        record.logical_provider = self.model_config.provider
        record.requested_model = self.model_config.model
        record.api_host = self.model_config.api_host
        record.official_route = (
            self.model_config.provenance.value == "official_provider"
            if self.model_config.provenance is not None
            else False
        )
        record.upstream_provider_verifiable = self.model_config.upstream_provider_verifiable
        record.experiment_namespace = (
            "gateway_unverified"
            if self.model_config.provider.value == "airforce"
            else "official"
            if record.official_route
            else "scripted_test"
        )
        record.system_prompt_version = self.agent_spec.system_prompt_version
        record.system_prompt_sha256 = self._prompt_hash
        record.system_prompt_content = self._bounded_utf8_text(self._system_prompt, 64_000)
        record.task_instruction_sha256 = self._instruction_hash

    def _request(
        self, turn_number: int, previous_results: tuple[ToolResult, ...]
    ) -> ModelTurnRequest:
        remaining = self._remaining_wall_time()
        request_timeout = min(
            self.model_config.model_call_timeout_seconds,
            remaining if remaining > 0 else 0.001,
        )
        return ModelTurnRequest(
            turn_number=turn_number,
            system_prompt=self._system_prompt,
            initial_task_instruction=self._instruction,
            tool_definitions=self._tool_definitions,
            tool_results=previous_results,
            run_config=self.model_config,
            request_timeout_seconds=request_timeout,
        )

    def _generate_with_retries(
        self, request: ModelTurnRequest
    ) -> tuple[ModelTurnResponse, tuple[ModelAttempt, ...], datetime, datetime]:
        attempts: list[ModelAttempt] = []
        request_started_at = self.wall_clock()
        last_error: ModelError | None = None
        maximum_attempts = self.model_config.maximum_retry_attempts + 1

        for attempt_number in range(1, maximum_attempts + 1):
            if attempts and self._wall_time_exceeded():
                assert last_error is not None
                raise _AttemptFailure(
                    last_error,
                    tuple(attempts),
                    request_started_at,
                    self.wall_clock(),
                )
            if self._provider_attempts >= self.agent_spec.maximum_provider_requests:
                if last_error is None:
                    last_error = ModelError(
                        ModelErrorCode.PROVIDER_UNAVAILABLE,
                        "the provider request limit was reached",
                    )
                raise _AttemptFailure(
                    last_error,
                    tuple(attempts),
                    request_started_at,
                    self.wall_clock(),
                    provider_request_limit_prevented_retry=True,
                )
            remaining = self._remaining_wall_time()
            effective_timeout = min(
                self.model_config.model_call_timeout_seconds,
                remaining if remaining > 0 else 0.001,
            )
            attempt_request = request.model_copy(
                update={"request_timeout_seconds": effective_timeout}
            )
            self._provider_attempts += 1
            attempt_started_at = self.wall_clock()
            attempt_started = self.clock()
            try:
                response = self.model_adapter.generate_next_turn(attempt_request)
                self._validate_response_identity(response)
            except ModelError as exc:
                last_error = exc
                latency = max(0.0, self.clock() - attempt_started)
                transient = exc.retryable and exc.model_code in _TRANSIENT_MODEL_ERRORS
                another_configured_attempt = attempt_number < maximum_attempts
                provider_quota_available = (
                    self._provider_attempts < self.agent_spec.maximum_provider_requests
                )
                retryable = (
                    transient
                    and another_configured_attempt
                    and provider_quota_available
                    and not self._wall_time_exceeded()
                )
                provider_request_limit_prevented_retry = (
                    transient and another_configured_attempt and not provider_quota_available
                )
                delay: float | None = None
                if retryable:
                    base_delay = min(
                        _MAXIMUM_RETRY_DELAY_SECONDS,
                        self.model_config.initial_retry_delay_seconds * (2 ** (attempt_number - 1)),
                    )
                    proposed_delay = min(
                        _MAXIMUM_RETRY_DELAY_SECONDS,
                        base_delay + max(0.0, self.jitter(base_delay)),
                    )
                    remaining = self._remaining_wall_time()
                    if remaining <= 0:
                        retryable = False
                    else:
                        delay = min(proposed_delay, remaining)
                attempts.append(
                    ModelAttempt(
                        attempt_number=attempt_number,
                        start_timestamp=attempt_started_at,
                        end_timestamp=self.wall_clock(),
                        latency_seconds=latency,
                        succeeded=False,
                        error_code=exc.model_code,
                        retryable=retryable,
                        retry_delay_seconds=delay,
                        request_timeout_seconds=effective_timeout,
                        provider_metadata=exc.provider_metadata,
                    )
                )
                if not retryable:
                    raise _AttemptFailure(
                        exc,
                        tuple(attempts),
                        request_started_at,
                        self.wall_clock(),
                        provider_request_limit_prevented_retry=(
                            provider_request_limit_prevented_retry
                        ),
                    ) from None
                assert delay is not None
                self.sleeper(delay)
                continue
            except KeyboardInterrupt:
                error = ModelError(
                    ModelErrorCode.CANCELLED,
                    "the model request was interrupted",
                )
                attempts.append(
                    ModelAttempt(
                        attempt_number=attempt_number,
                        start_timestamp=attempt_started_at,
                        end_timestamp=self.wall_clock(),
                        latency_seconds=max(0.0, self.clock() - attempt_started),
                        succeeded=False,
                        error_code=error.model_code,
                        request_timeout_seconds=effective_timeout,
                    )
                )
                raise _AttemptFailure(
                    error,
                    tuple(attempts),
                    request_started_at,
                    self.wall_clock(),
                ) from None
            except Exception:
                error = ModelError(
                    ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                    "the model adapter failed internally",
                )
                attempts.append(
                    ModelAttempt(
                        attempt_number=attempt_number,
                        start_timestamp=attempt_started_at,
                        end_timestamp=self.wall_clock(),
                        latency_seconds=max(0.0, self.clock() - attempt_started),
                        succeeded=False,
                        error_code=error.model_code,
                        request_timeout_seconds=effective_timeout,
                    )
                )
                raise _AttemptFailure(
                    error,
                    tuple(attempts),
                    request_started_at,
                    self.wall_clock(),
                ) from None

            observed_latency = max(0.0, self.clock() - attempt_started)
            latency = max(observed_latency, response.latency_seconds)
            attempts.append(
                ModelAttempt(
                    attempt_number=attempt_number,
                    start_timestamp=attempt_started_at,
                    end_timestamp=self.wall_clock(),
                    latency_seconds=latency,
                    succeeded=True,
                    request_timeout_seconds=effective_timeout,
                )
            )
            normalized = response.model_copy(
                update={
                    "attempts": tuple(attempts),
                    "attempt_count": len(attempts),
                    "latency_seconds": sum(attempt.latency_seconds for attempt in attempts),
                }
            )
            return normalized, tuple(attempts), request_started_at, self.wall_clock()

        assert last_error is not None  # pragma: no cover
        raise _AttemptFailure(
            last_error,
            tuple(attempts),
            request_started_at,
            self.wall_clock(),
        )

    def _validate_response_identity(self, response: ModelTurnResponse) -> None:
        if response.provider is not self.model_config.provider:
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "the adapter returned a different provider identifier",
            )
        if response.requested_model != self.model_config.model:
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "the adapter returned a different requested model identifier",
            )
        expected_official = (
            self.model_config.provenance is ProvenanceClassification.OFFICIAL_PROVIDER
        )
        if (
            response.transport is not self.model_config.transport
            or response.provenance is not self.model_config.provenance
            or response.api_host != self.model_config.api_host
            or response.official_route is not expected_official
            or response.upstream_provider_verifiable
            is not self.model_config.upstream_provider_verifiable
        ):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "the adapter returned a response from a different inference route",
            )

    def _normalize_response_indexes(
        self, response: ModelTurnResponse, attempts: tuple[ModelAttempt, ...]
    ) -> ModelTurnResponse:
        calls = tuple(
            call
            if call.original_index == index
            else call.model_copy(update={"original_index": index})
            for index, call in enumerate(response.tool_calls)
        )
        return response.model_copy(
            update={"tool_calls": calls, "attempts": attempts, "attempt_count": len(attempts)}
        )

    def _record_model_event(
        self,
        turn_number: int,
        response: ModelTurnResponse,
        request_start: datetime,
        request_end: datetime,
        *,
        extra_warning: str | None,
        request_timeout_seconds: float | None,
    ) -> None:
        assert self.session is not None
        warnings = response.warnings + ((extra_warning,) if extra_warning else ())
        event = ModelEvent(
            sequence_number=len(self.session.record.model_events) + 1,
            turn_number=turn_number,
            provider=response.provider,
            requested_model=response.requested_model,
            returned_model=response.returned_model,
            request_start_timestamp=request_start,
            request_end_timestamp=request_end,
            latency_seconds=response.latency_seconds,
            attempt_count=response.attempt_count,
            attempts=response.attempts,
            finish_reason=response.finish_reason,
            text=response.text,
            tool_calls=tuple(self._tool_call_trace(call) for call in response.tool_calls),
            usage=response.usage,
            requested_settings=self.requested_model_config.requested_settings(),
            effective_settings=self._effective_settings(request_timeout_seconds),
            tool_definitions_sha256=self._tool_definitions_hash,
            prompt_sha256=self._prompt_hash,
            response_id=response.response_id,
            warnings=warnings,
            provider_metadata=response.provider_metadata,
            transport=response.transport or self.model_config.transport,
            api_host=response.api_host or self.model_config.api_host or "scripted.local",
            provenance=response.provenance or self.model_config.provenance,
            official_route=bool(response.official_route),
            upstream_provider_verifiable=bool(response.upstream_provider_verifiable),
            batch_job_id=response.batch_job_id,
            batch_request_id=response.batch_request_id,
        )
        self.session.record.model_events.append(event)
        if (
            event.returned_model is not None
            and event.returned_model not in self.session.record.returned_models
        ):
            self.session.record.returned_models.append(event.returned_model)
        if (
            event.batch_job_id is not None
            and event.batch_job_id not in self.session.record.batch_job_ids
        ):
            self.session.record.batch_job_ids.append(event.batch_job_id)
        if (
            event.batch_request_id is not None
            and event.batch_request_id not in self.session.record.batch_request_ids
        ):
            self.session.record.batch_request_ids.append(event.batch_request_id)
        self.session.record.metrics = RunMetrics.aggregate(
            self.session.record.model_events,
            self.session.record.tool_events,
            total_wall_time_seconds=max(0.0, self.clock() - self._run_started),
        )

    def _record_failed_model_event(
        self,
        turn_number: int,
        failure: _AttemptFailure,
        *,
        request_timeout_seconds: float | None,
    ) -> None:
        assert self.session is not None
        event = ModelEvent(
            sequence_number=len(self.session.record.model_events) + 1,
            turn_number=turn_number,
            provider=self.model_config.provider,
            requested_model=self.model_config.model,
            request_start_timestamp=failure.request_started_at,
            request_end_timestamp=failure.request_ended_at,
            latency_seconds=sum(attempt.latency_seconds for attempt in failure.attempts),
            attempt_count=len(failure.attempts),
            attempts=failure.attempts,
            finish_reason=(
                NormalizedFinishReason.CANCELLED
                if failure.error.model_code is ModelErrorCode.CANCELLED
                else NormalizedFinishReason.PROVIDER_ERROR
            ),
            requested_settings=self.requested_model_config.requested_settings(),
            effective_settings=self._effective_settings(request_timeout_seconds),
            tool_definitions_sha256=self._tool_definitions_hash,
            prompt_sha256=self._prompt_hash,
            error=failure.error.error,
            provider_metadata=failure.error.provider_metadata,
            transport=self.model_config.transport,
            api_host=self.model_config.api_host or "scripted.local",
            provenance=self.model_config.provenance,
            official_route=self.model_config.provenance.value == "official_provider",
            upstream_provider_verifiable=bool(self.model_config.upstream_provider_verifiable),
        )
        self.session.record.model_events.append(event)

    def _tool_call_trace(self, call: ToolCall) -> ModelToolCallTrace:
        redacted = redact_arguments(call.tool_name, call.arguments)
        argument_limit = min(4_096, self.task.tool_policy.max_tool_result_size_bytes)
        arguments, digest, truncated = bounded_json_value(redacted, argument_limit)
        return ModelToolCallTrace(
            call_id=call.call_id,
            tool_name=call.tool_name,
            original_index=call.original_index or 0,
            arguments=arguments,
            arguments_sha256=digest,
            arguments_truncated=truncated,
        )

    def _effective_settings(self, request_timeout_seconds: float | None) -> dict[str, Any]:
        settings = self.model_config.requested_settings()
        settings["effective_request_timeout_seconds"] = request_timeout_seconds
        return settings

    def _usage_enforcement_warning(self, response: ModelTurnResponse) -> str | None:
        usage = response.usage
        missing = (
            (self.agent_spec.maximum_total_input_tokens is not None and usage.input_tokens is None)
            or (
                self.agent_spec.maximum_total_output_tokens is not None
                and usage.output_tokens is None
            )
            or (self.agent_spec.maximum_total_tokens is not None and usage.total_tokens is None)
        )
        if missing:
            return "provider usage was unavailable; exact token-limit enforcement was not possible"
        return None

    def _token_limit_reason(self) -> AgentTerminationReason | None:
        assert self.session is not None
        events = self.session.record.model_events

        def total(field_name: str) -> int | None:
            values = [getattr(event.usage, field_name) for event in events]
            if not values or any(value is None for value in values):
                return None
            return sum(value for value in values if value is not None)

        input_tokens = total("input_tokens")
        output_tokens = total("output_tokens")
        all_tokens = total("total_tokens")
        if (
            self.agent_spec.maximum_total_input_tokens is not None
            and input_tokens is not None
            and input_tokens > self.agent_spec.maximum_total_input_tokens
        ):
            return AgentTerminationReason.INPUT_TOKEN_LIMIT_REACHED
        if (
            self.agent_spec.maximum_total_output_tokens is not None
            and output_tokens is not None
            and output_tokens > self.agent_spec.maximum_total_output_tokens
        ):
            return AgentTerminationReason.OUTPUT_TOKEN_LIMIT_REACHED
        if (
            self.agent_spec.maximum_total_tokens is not None
            and all_tokens is not None
            and all_tokens > self.agent_spec.maximum_total_tokens
        ):
            return AgentTerminationReason.TOTAL_TOKEN_LIMIT_REACHED
        return None

    def _duplicate_call_id(self, calls: tuple[ToolCall, ...]) -> str | None:
        response_ids: set[str] = set()
        for call in calls:
            if call.call_id in response_ids or call.call_id in self._seen_tool_call_ids:
                return call.call_id
            response_ids.add(call.call_id)
        self._seen_tool_call_ids.update(response_ids)
        return None

    def _apply_outcome(
        self,
        record: RunRecord,
        status: RunStatus,
        reason: AgentTerminationReason,
        error: BaseException | None,
    ) -> None:
        record.final_status = status
        record.termination_reason = reason
        if error is None:
            record.error = None
            return
        if isinstance(error, ModelError):
            code = error.model_code.value
            message = error.message
        elif isinstance(error, SandboxError):
            code = error.code
            message = "sandbox execution failed"
        elif isinstance(error, KeyboardInterrupt):
            code = "cancelled"
            message = "agent run was interrupted"
        else:
            code = "internal_orchestration_error"
            message = "agent orchestration failed internally"
        record.error = ErrorInformation(type=type(error).__name__, code=code, message=message)

    def _wall_time_exceeded(self) -> bool:
        return self.clock() - self._run_started >= self.agent_spec.overall_wall_time_seconds

    def _remaining_wall_time(self) -> float:
        return max(
            0.0,
            self.agent_spec.overall_wall_time_seconds - (self.clock() - self._run_started),
        )

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _bounded_utf8_text(value: str, maximum_bytes: int) -> str:
        return value.encode("utf-8")[:maximum_bytes].decode("utf-8", errors="ignore")

    @staticmethod
    def _canonical_json(value: Any) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
