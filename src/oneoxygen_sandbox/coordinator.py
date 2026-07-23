"""Turn-level durable agent coordinator for direct-compatible batch execution."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from oneoxygen_sandbox.batching.backends import BatchBackend, OpenAIBatchBackend
from oneoxygen_sandbox.batching.models import BatchItemResult, BatchJob, BatchRequest
from oneoxygen_sandbox.batching.store import BatchArtifactStore
from oneoxygen_sandbox.checkpoints import WorkspaceCheckpoint
from oneoxygen_sandbox.errors import ConfigurationError, LifecycleError
from oneoxygen_sandbox.filesystem import copy_input_assets
from oneoxygen_sandbox.model_adapters.openai import (
    apply_openai_batch_response_state,
    compile_openai_batch_turn,
)
from oneoxygen_sandbox.models import (
    AgentTerminationReason,
    InferenceTransport,
    ModelProvider,
    ModelRunConfig,
    ModelTurnRequest,
    ModelTurnResponse,
    ModelUsage,
    SandboxTask,
    ToolDefinition,
    ToolResult,
)
from oneoxygen_sandbox.orchestration import (
    AgentRunState,
    AgentRunStatus,
    BatchRequestReference,
    ReadyModelTurn,
    SQLiteAgentStateStore,
)
from oneoxygen_sandbox.session import SandboxSession
from oneoxygen_sandbox.tools import ToolDispatcher, ToolRegistry, default_tool_registry


class DurableAgentCoordinator:
    """Advance persisted runs one provider turn and one fresh sandbox at a time."""

    def __init__(
        self,
        states: SQLiteAgentStateStore,
        artifacts: BatchArtifactStore,
        runs_directory: Path,
        *,
        sandbox_adapter: Any | None = None,
        tool_registry: ToolRegistry | None = None,
        maximum_checkpoint_bytes: int = 100 * 1024 * 1024,
        maximum_batch_retries: int = 2,
    ) -> None:
        self.states = states
        self.artifacts = artifacts
        self.runs_directory = runs_directory.resolve()
        self.runs_directory.mkdir(parents=True, exist_ok=True)
        self.sandbox_adapter = sandbox_adapter
        self.tool_registry = tool_registry or default_tool_registry()
        self.maximum_checkpoint_bytes = maximum_checkpoint_bytes
        if maximum_batch_retries < 0 or maximum_batch_retries > 98:
            raise ValueError("maximum batch retries must be between 0 and 98")
        self.maximum_batch_retries = maximum_batch_retries

    def enqueue(
        self,
        task: SandboxTask,
        task_directory: Path,
        model_config: ModelRunConfig,
        *,
        run_id: str | None = None,
    ) -> AgentRunState:
        if task.agent is None:
            raise ConfigurationError("durable agent runs require an agent task")
        if model_config.transport is not InferenceTransport.PROVIDER_BATCH:
            raise ConfigurationError("durable batch enqueue requires provider_batch transport")
        if model_config.provider is ModelProvider.AIRFORCE:
            raise ConfigurationError("Api.Airforce does not document a Batch endpoint")
        identifier = run_id or uuid.uuid4().hex
        task_root = task_directory.resolve(strict=True)
        instruction = self._read_task_file(task_root, task.agent.instruction_file, 1_000_000)
        system_prompt = self._system_prompt(task, task_root)
        definitions = self._tool_definitions(task)
        prompt_hash = _sha256_text(system_prompt)
        tool_hash = _sha256_json([definition.model_dump(mode="json") for definition in definitions])
        checkpoint = self._checkpoint(identifier)
        temporary = Path(tempfile.mkdtemp(prefix=f"oneoxygen-enqueue-{identifier[:8]}-"))
        try:
            copy_input_assets(task_root, temporary, task.sandbox.input_assets)
            temporary.joinpath(*task.sandbox.working_relative_path.parts).mkdir(
                parents=True, exist_ok=True
            )
            checkpoint.capture(temporary, 0)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        state = AgentRunState(
            run_id=identifier,
            task_id=task.sandbox.task_id,
            task_version=task.sandbox.task_version,
            model_configuration=model_config,
            transport=model_config.transport,
            provenance=model_config.provenance,
            prompt_sha256=prompt_hash,
            tool_schema_sha256=tool_hash,
            workspace_checkpoint_reference="checkpoints",
            workspace_checkpoint_generation=0,
            system_prompt=system_prompt,
            initial_task_instruction=instruction,
            tool_definitions=definitions,
            system_prompt_version=task.agent.system_prompt_version,
            task_configuration=task.model_dump(mode="json"),
            task_directory_reference=str(task_root),
        )
        self.states.create(state)
        return self.states.transition(identifier, AgentRunStatus.READY_FOR_MODEL)

    def ready_turn(self, run_id: str, *, attempt_number: int | None = None) -> ReadyModelTurn:
        state = self.states.load(run_id)
        if state.status is not AgentRunStatus.READY_FOR_MODEL:
            raise LifecycleError("agent run is not ready for a model turn")
        derived_attempt = state.retry_counts.get(str(state.current_turn), 0) + 1
        if attempt_number is not None and attempt_number != derived_attempt:
            raise LifecycleError("model-turn attempt does not match durable retry state")
        results = tuple(ToolResult.model_validate(result) for result in state.pending_tool_results)
        request = ModelTurnRequest(
            turn_number=state.current_turn,
            system_prompt=state.system_prompt,
            initial_task_instruction=state.initial_task_instruction,
            tool_definitions=state.tool_definitions,
            tool_results=results,
            run_config=state.model_configuration,
            request_timeout_seconds=state.model_configuration.model_call_timeout_seconds,
        )
        classification = None
        task = SandboxTask.model_validate(state.task_configuration)
        if task.agent is not None and task.agent.data_classification is not None:
            classification = task.agent.data_classification.value
        return ReadyModelTurn(
            run_id=run_id,
            turn_number=state.current_turn,
            attempt_number=derived_attempt,
            request=request,
            provider_conversation_state=state.provider_conversation_state,
            prompt_sha256=state.prompt_sha256,
            tool_schema_sha256=state.tool_schema_sha256,
            system_prompt_version=state.system_prompt_version,
            data_policy_class=classification,
        )

    def materialize_batch_request(self, ready: ReadyModelTurn) -> BatchRequest:
        state = self.states.load(ready.run_id)
        if state.status is not AgentRunStatus.READY_FOR_MODEL:
            raise LifecycleError("agent run is no longer ready for batching")
        prior_reference = max(
            (
                reference
                for reference in state.batch_request_references
                if reference.turn_number == ready.turn_number
            ),
            key=lambda item: item.attempt_number,
            default=None,
        )
        if ready.attempt_number > 1:
            if (
                prior_reference is None
                or prior_reference.compiled_request_body_sha256 is None
                or prior_reference.compiled_request_body_reference is None
            ):
                raise LifecycleError("retry is missing its original compiled request")
            digest = prior_reference.compiled_request_body_sha256
            reference = prior_reference.compiled_request_body_reference
            pending_provider_state = state.provider_conversation_state
            endpoint = "/v1/responses"
        elif state.model_configuration.provider is ModelProvider.OPENAI:
            body, pending_provider_state = compile_openai_batch_turn(
                state.model_configuration,
                ready.request,
                state.provider_conversation_state,
            )
            endpoint = "/v1/responses"
        else:
            body = {
                "model": state.model_configuration.model,
                "turn": ready.turn_number,
                "request": ready.request.model_dump(mode="json"),
            }
            pending_provider_state = state.provider_conversation_state
            endpoint = "/v1/responses"
        if ready.attempt_number == 1:
            body_id = _sha256_text(f"{ready.run_id}:{ready.turn_number}:{ready.attempt_number}")[
                :32
            ]
            reference, digest = self.artifacts.write_json("requests", body_id, body)
        request_id = (
            "req_" + _sha256_text(f"{ready.run_id}:{ready.turn_number}:{ready.attempt_number}")[:32]
        )
        generation_settings = state.model_configuration.requested_settings()
        request = BatchRequest(
            internal_request_id=request_id,
            run_id=ready.run_id,
            turn_number=ready.turn_number,
            attempt_number=ready.attempt_number,
            provider=state.model_configuration.provider,
            model=state.model_configuration.model,
            endpoint=endpoint,
            compiled_request_body_sha256=digest,
            compiled_request_body_reference=reference,
            tool_schema_sha256=state.tool_schema_sha256,
            system_prompt_version=ready.system_prompt_version,
            schema_mode=state.model_configuration.tool_schema_mode.value,
            effective_generation_settings_sha256=_sha256_json(generation_settings),
            data_policy_class=ready.data_policy_class,
        )
        batch_reference = BatchRequestReference(
            internal_request_id=request_id,
            compiled_request_body_sha256=digest,
            compiled_request_body_reference=reference,
            turn_number=ready.turn_number,
            attempt_number=ready.attempt_number,
        )
        self.states.transition(
            ready.run_id,
            AgentRunStatus.BATCH_DRAFTED,
            updates={
                "provider_conversation_state": pending_provider_state,
                "batch_request_references": (
                    *state.batch_request_references,
                    batch_reference,
                ),
            },
            expected_revision=state.revision,
        )
        return request

    def mark_batch_submitted(self, job: BatchJob) -> tuple[AgentRunState, ...]:
        updated: list[AgentRunState] = []
        for request_id in job.request_ids:
            request_run = self._run_for_request(request_id)
            state = self.states.load(request_run)
            if state.status is not AgentRunStatus.BATCH_DRAFTED:
                raise LifecycleError("run is not in the batch-drafted state")
            references = tuple(
                reference.model_copy(
                    update={
                        "internal_batch_id": job.internal_batch_id,
                        "provider_batch_id": job.provider_batch_id,
                    }
                )
                if reference.internal_request_id == request_id
                else reference
                for reference in state.batch_request_references
            )
            submitted = self.states.transition(
                state.run_id,
                AgentRunStatus.BATCH_SUBMITTED,
                updates={"batch_request_references": references},
                expected_revision=state.revision,
            )
            updated.append(
                self.states.transition(
                    state.run_id,
                    AgentRunStatus.WAITING_FOR_MODEL,
                    expected_revision=submitted.revision,
                )
            )
        return tuple(updated)

    def apply_result(
        self,
        result: BatchItemResult,
        backend: BatchBackend,
    ) -> AgentRunState:
        run_id = self._run_for_request(result.request_id)
        state = self.states.load(run_id)
        if state.status is not AgentRunStatus.WAITING_FOR_MODEL:
            if any(event.get("request_id") == result.request_id for event in state.model_trace):
                return state
            raise LifecycleError("agent run is not waiting for a model result")
        batch_reference = next(
            reference
            for reference in state.batch_request_references
            if reference.internal_request_id == result.request_id
        )
        if not result.success:
            retry_key = str(state.current_turn)
            retries_used = state.retry_counts.get(retry_key, 0)
            trace = (
                *state.model_trace,
                {
                    "request_id": result.request_id,
                    "provider_custom_id": result.provider_custom_id,
                    "batch_id": batch_reference.provider_batch_id,
                    "turn": state.current_turn,
                    "success": False,
                    "retryable": result.retryable,
                    "error": result.normalized_error,
                },
            )
            if result.retryable and retries_used < self.maximum_batch_retries:
                return self.states.transition(
                    run_id,
                    AgentRunStatus.READY_FOR_MODEL,
                    updates={
                        "model_trace": trace,
                        "retry_counts": {
                            **state.retry_counts,
                            retry_key: retries_used + 1,
                        },
                        "termination_message": None,
                    },
                    expected_revision=state.revision,
                )
            return self.states.transition(
                run_id,
                AgentRunStatus.FAILED,
                updates={
                    "model_trace": trace,
                    "termination_reason": AgentTerminationReason.REPEATED_PROVIDER_FAILURE,
                    "termination_message": (
                        str(result.normalized_error.get("code"))
                        if result.normalized_error
                        else "batch item failed"
                    ),
                },
                expected_revision=state.revision,
            )
        request = self._batch_request(result.request_id, backend)
        if isinstance(backend, OpenAIBatchBackend):
            response, serialized_output = backend.normalize_model_response(request, result)
            provider_state = apply_openai_batch_response_state(
                state.provider_conversation_state,
                serialized_output,
                state.current_turn,
            )
        else:
            if result.response is None:
                raise LifecycleError("successful batch result is missing its response")
            response = ModelTurnResponse.model_validate(result.response)
            provider_state = state.provider_conversation_state
        response = ModelTurnResponse.model_validate(
            response.model_copy(
                update={
                    "batch_job_id": batch_reference.provider_batch_id,
                    "batch_request_id": result.request_id,
                    "transport": state.transport,
                    "api_host": state.api_host,
                    "provenance": state.provenance,
                    "official_route": state.official_route,
                    "upstream_provider_verifiable": (state.upstream_provider_verifiable),
                }
            ).model_dump(mode="python")
        )
        ready = self.states.transition(
            run_id,
            AgentRunStatus.MODEL_RESULT_READY,
            updates={
                "provider_conversation_state": provider_state,
                "normalized_conversation_history": (
                    *state.normalized_conversation_history,
                    {
                        "turn": state.current_turn,
                        "response": response.model_dump(mode="json"),
                    },
                ),
                "model_trace": (
                    *state.model_trace,
                    {
                        "request_id": result.request_id,
                        "provider_custom_id": result.provider_custom_id,
                        "batch_id": batch_reference.provider_batch_id,
                        "turn": state.current_turn,
                        "success": True,
                        "response": response.model_dump(mode="json"),
                        "accounting": (
                            result.accounting.model_dump(mode="json")
                            if result.accounting is not None
                            else None
                        ),
                    },
                ),
                "usage_totals": _add_usage(state.usage_totals, response.usage),
            },
            expected_revision=state.revision,
        )
        if not response.tool_calls:
            return self.states.transition(
                run_id,
                AgentRunStatus.INCOMPLETE,
                updates={
                    "termination_reason": AgentTerminationReason.FINAL_TEXT_WITHOUT_SUBMISSION,
                    "termination_message": "model returned final text without submit_result",
                },
                expected_revision=ready.revision,
            )
        return self._execute_tools(ready, response)

    def expire_waiting_run(self, run_id: str) -> AgentRunState:
        state = self.states.load(run_id)
        return self.states.transition(
            run_id,
            AgentRunStatus.EXPIRED,
            updates={"termination_message": "provider batch expired"},
            expected_revision=state.revision,
        )

    def cancel_run(self, run_id: str) -> AgentRunState:
        state = self.states.load(run_id)
        return self.states.transition(
            run_id,
            AgentRunStatus.CANCELLED,
            updates={
                "termination_reason": AgentTerminationReason.USER_INTERRUPTION,
                "termination_message": "agent run was cancelled",
            },
            expected_revision=state.revision,
        )

    def _execute_tools(self, state: AgentRunState, response: ModelTurnResponse) -> AgentRunState:
        executing = self.states.transition(
            state.run_id,
            AgentRunStatus.EXECUTING_TOOLS,
            expected_revision=state.revision,
        )
        task = SandboxTask.model_validate(executing.task_configuration)
        if executing.task_directory_reference is None:
            raise LifecycleError("durable task directory reference is missing")
        generation = executing.workspace_checkpoint_generation
        if generation is None:
            raise LifecycleError("durable workspace checkpoint is missing")
        checkpoint = self._checkpoint(executing.run_id)
        session = SandboxSession(
            task,
            Path(executing.task_directory_reference),
            self.runs_directory / executing.run_id / "turn-executions",
            self.sandbox_adapter,
            checkpoint=checkpoint,
            restore_generation=generation,
        )
        results: list[ToolResult] = []
        dispatcher: ToolDispatcher | None = None
        try:
            session.start()
            dispatcher = ToolDispatcher(
                session,
                registry=self.tool_registry,
                initial_total_count=executing.total_tool_calls,
                initial_tool_counts=executing.per_tool_call_totals,
                initial_sequence_number=executing.total_tool_calls,
            )
            for call in response.tool_calls:
                results.append(dispatcher.dispatch(call))
            if session.workspace_path is None:
                raise LifecycleError("sandbox workspace disappeared during tool execution")
            next_generation = generation + 1
            checkpoint.capture(session.workspace_path, next_generation)
            checkpoint.cleanup()
            total, per_tool, _sequence = dispatcher.counter_snapshot()
            common_updates: dict[str, Any] = {
                "workspace_checkpoint_generation": next_generation,
                "pending_tool_results": tuple(result.model_dump(mode="json") for result in results),
                "total_tool_calls": total,
                "per_tool_call_totals": per_tool,
                "tool_trace": (
                    *executing.tool_trace,
                    *(event.model_dump(mode="json") for event in session.record.tool_events),
                ),
            }
            if dispatcher.submission_state.submitted:
                session.collect_submitted_artifacts()
                submitted = self.states.transition(
                    executing.run_id,
                    AgentRunStatus.SUBMITTED,
                    updates={
                        **common_updates,
                        "termination_reason": (AgentTerminationReason.SUCCESSFUL_SUBMISSION),
                    },
                    expected_revision=executing.revision,
                )
                return self.states.transition(
                    executing.run_id,
                    AgentRunStatus.COMPLETED,
                    expected_revision=submitted.revision,
                )
            next_turn = self.states.transition(
                executing.run_id,
                AgentRunStatus.READY_FOR_NEXT_TURN,
                updates={
                    **common_updates,
                    "current_turn": executing.current_turn + 1,
                },
                expected_revision=executing.revision,
            )
            return self.states.transition(
                executing.run_id,
                AgentRunStatus.READY_FOR_MODEL,
                expected_revision=next_turn.revision,
            )
        finally:
            if session._record is not None:
                session.stop()

    def _batch_request(self, request_id: str, backend: BatchBackend) -> BatchRequest:
        store = getattr(backend, "store", None)
        if store is None:
            raise LifecycleError("batch backend does not expose its durable request store")
        return store.load_request(request_id)

    def _run_for_request(self, request_id: str) -> str:
        for state in self.states.list():
            if any(
                reference.internal_request_id == request_id
                for reference in state.batch_request_references
            ):
                return state.run_id
        raise LifecycleError("batch request is not mapped to a durable run")

    def _checkpoint(self, run_id: str) -> WorkspaceCheckpoint:
        return WorkspaceCheckpoint(
            self.runs_directory / run_id / "checkpoints",
            run_id,
            maximum_size_bytes=self.maximum_checkpoint_bytes,
        )

    def _tool_definitions(self, task: SandboxTask) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = []
        for definition in self.tool_registry.definitions():
            if definition.name not in task.tool_policy.allowed_tool_names:
                continue
            if definition.name == "execute_shell" and not task.tool_policy.shell_execution_allowed:
                continue
            if (
                definition.name == "execute_python"
                and not task.tool_policy.python_execution_allowed
            ):
                continue
            definitions.append(definition)
        return tuple(definitions)

    @staticmethod
    def _read_task_file(root: Path, relative: str, maximum_bytes: int) -> str:
        candidate = root.joinpath(*Path(relative).parts)
        current = root
        for part in Path(relative).parts:
            current = current / part
            if current.is_symlink():
                raise ConfigurationError("durable agent input cannot be a symbolic link")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ConfigurationError("durable agent input escapes the task directory") from exc
        if not resolved.is_file():
            raise ConfigurationError("durable agent input must be a regular file")
        data = resolved.read_bytes()
        if len(data) > maximum_bytes:
            raise ConfigurationError("durable agent input exceeds its size limit")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigurationError("durable agent input must be UTF-8") from exc

    def _system_prompt(self, task: SandboxTask, task_root: Path) -> str:
        assert task.agent is not None
        if task.agent.system_prompt_file is not None:
            return self._read_task_file(task_root, task.agent.system_prompt_file, 256_000)
        if task.agent.system_prompt_version != "standard_agent_v1":
            raise ConfigurationError("unsupported built-in system prompt version")
        return (Path(__file__).with_name("prompts") / "standard_agent_v1.txt").read_text(
            encoding="utf-8"
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    if all(
        getattr(left, field) is None
        for field in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "total_tokens",
        )
    ):
        return right

    def add(field: str) -> int | None:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None or right_value is None:
            return None
        return left_value + right_value

    return ModelUsage(
        input_tokens=add("input_tokens"),
        output_tokens=add("output_tokens"),
        reasoning_tokens=add("reasoning_tokens"),
        cached_input_tokens=add("cached_input_tokens"),
        total_tokens=add("total_tokens"),
    )
