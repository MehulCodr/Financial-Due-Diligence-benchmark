from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from oneoxygen_sandbox.batching import (
    BatchArtifactStore,
    MockBatchBackend,
    SQLiteBatchStore,
)
from oneoxygen_sandbox.checkpoints import WorkspaceCheckpoint
from oneoxygen_sandbox.config import load_task
from oneoxygen_sandbox.coordinator import DurableAgentCoordinator
from oneoxygen_sandbox.models import (
    InferenceTransport,
    ModelProvider,
    ModelRunConfig,
    ModelTurnResponse,
    NormalizedFinishReason,
    ToolCall,
)
from oneoxygen_sandbox.orchestration import AgentRunStatus, SQLiteAgentStateStore


class FakeSandboxAdapter:
    sandbox_user = "10001:10001"

    def resolve_image(self, image: str) -> str:
        return f"resolved:{image}"

    def create_container(self, *_args, **_kwargs):
        return SimpleNamespace(status="created")

    def start_container(self, container) -> None:
        container.status = "running"

    def stop_container(self, container) -> None:
        container.status = "exited"

    def remove_container(self, container) -> None:
        container.status = "removed"


def _coordinator(
    root: Path, adapter: FakeSandboxAdapter
) -> tuple[DurableAgentCoordinator, BatchArtifactStore, SQLiteBatchStore]:
    states = SQLiteAgentStateStore(str(root / "agents.sqlite3"))
    artifacts = BatchArtifactStore(root / "batch-files")
    batches = SQLiteBatchStore(root / "batches.sqlite3")
    return (
        DurableAgentCoordinator(
            states,
            artifacts,
            root / "runs",
            sandbox_adapter=adapter,
        ),
        artifacts,
        batches,
    )


def test_offline_multiturn_batch_restart_checkpoint_and_submission(tmp_path: Path) -> None:
    task_path = Path("examples/agent_demo/task.yaml").resolve()
    task = load_task(task_path)
    config = ModelRunConfig(
        provider=ModelProvider.SCRIPTED,
        model="offline-script",
        transport=InferenceTransport.PROVIDER_BATCH,
    )
    adapter = FakeSandboxAdapter()
    coordinator, artifacts, batches = _coordinator(tmp_path, adapter)
    state = coordinator.enqueue(task, task_path.parent, config, run_id="durable-run")

    first_request = coordinator.materialize_batch_request(coordinator.ready_turn(state.run_id))
    first_response = ModelTurnResponse(
        provider=ModelProvider.SCRIPTED,
        requested_model=config.model,
        returned_model=config.model,
        finish_reason=NormalizedFinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                call_id="write-1",
                tool_name="write_text_file",
                arguments={
                    "path": "output/report.txt",
                    "content": "checkpointed\n",
                    "create_parents": True,
                },
            ),
        ),
    )
    backend = MockBatchBackend(
        artifacts,
        batches,
        responses={first_request.internal_request_id: first_response.model_dump(mode="json")},
    )
    first_job = backend.submit_batch(backend.build_batch((first_request,)))
    coordinator.mark_batch_submitted(first_job)
    first_result = backend.retrieve_results(backend.get_status(first_job))[0]
    after_first = coordinator.apply_result(first_result, backend)
    assert after_first.status is AgentRunStatus.READY_FOR_MODEL
    assert after_first.workspace_checkpoint_generation == 1

    # Recreate every coordinator component to simulate a terminated process.
    coordinator, artifacts, batches = _coordinator(tmp_path, adapter)
    second_request = coordinator.materialize_batch_request(coordinator.ready_turn("durable-run"))
    second_response = ModelTurnResponse(
        provider=ModelProvider.SCRIPTED,
        requested_model=config.model,
        returned_model=config.model,
        finish_reason=NormalizedFinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                call_id="submit-2",
                tool_name="submit_result",
                arguments={
                    "summary": "offline complete",
                    "artifact_paths": ["output/report.txt"],
                },
            ),
        ),
    )
    backend = MockBatchBackend(
        artifacts,
        batches,
        responses={second_request.internal_request_id: second_response.model_dump(mode="json")},
    )
    second_job = backend.submit_batch(backend.build_batch((second_request,)))
    coordinator.mark_batch_submitted(second_job)
    second_result = backend.retrieve_results(backend.get_status(second_job))[0]
    completed = coordinator.apply_result(second_result, backend)

    assert completed.status is AgentRunStatus.COMPLETED
    assert completed.workspace_checkpoint_generation == 2
    assert [event["request_id"] for event in completed.model_trace] == [
        first_request.internal_request_id,
        second_request.internal_request_id,
    ]
    assert [event["tool_name"] for event in completed.tool_trace] == [
        "write_text_file",
        "submit_result",
    ]
    assert coordinator.apply_result(second_result, backend) == completed


def test_offline_several_runs_are_batched_out_of_order_and_resumed(
    tmp_path: Path,
) -> None:
    task_path = Path("examples/agent_demo/task.yaml").resolve()
    task = load_task(task_path)
    config = ModelRunConfig(
        provider=ModelProvider.SCRIPTED,
        model="offline-multi-run",
        transport=InferenceTransport.PROVIDER_BATCH,
    )
    adapter = FakeSandboxAdapter()
    coordinator, artifacts, batches = _coordinator(tmp_path, adapter)
    run_ids = tuple(f"multi-run-{index}" for index in range(3))
    for run_id in run_ids:
        coordinator.enqueue(task, task_path.parent, config, run_id=run_id)

    first_requests = tuple(
        coordinator.materialize_batch_request(coordinator.ready_turn(run_id)) for run_id in run_ids
    )
    first_responses = {
        request.internal_request_id: ModelTurnResponse(
            provider=ModelProvider.SCRIPTED,
            requested_model=config.model,
            returned_model=config.model,
            finish_reason=NormalizedFinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    call_id=f"write-{index}",
                    tool_name="write_text_file",
                    arguments={
                        "path": f"output/report-{index}.txt",
                        "content": f"run {index}\n",
                        "create_parents": True,
                    },
                ),
            ),
        ).model_dump(mode="json")
        for index, request in enumerate(first_requests)
    }
    backend = MockBatchBackend(artifacts, batches, responses=first_responses)
    first_job = backend.submit_batch(backend.build_batch(first_requests))
    coordinator.mark_batch_submitted(first_job)
    first_results = backend.retrieve_results(backend.get_status(first_job))
    assert {result.request_id for result in first_results} == {
        request.internal_request_id for request in first_requests
    }

    # Apply one result, recreate all durable components, then finish the batch.
    coordinator.apply_result(first_results[-1], backend)
    coordinator, artifacts, batches = _coordinator(tmp_path, adapter)
    backend = MockBatchBackend(artifacts, batches, responses=first_responses)
    for result in first_results[:-1]:
        assert coordinator.apply_result(result, backend).status is AgentRunStatus.READY_FOR_MODEL

    second_requests = tuple(
        coordinator.materialize_batch_request(coordinator.ready_turn(run_id)) for run_id in run_ids
    )
    second_responses = {
        request.internal_request_id: ModelTurnResponse(
            provider=ModelProvider.SCRIPTED,
            requested_model=config.model,
            returned_model=config.model,
            finish_reason=NormalizedFinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    call_id=f"submit-{index}",
                    tool_name="submit_result",
                    arguments={
                        "summary": f"completed run {index}",
                        "artifact_paths": [f"output/report-{index}.txt"],
                    },
                ),
            ),
        ).model_dump(mode="json")
        for index, request in enumerate(second_requests)
    }
    backend = MockBatchBackend(artifacts, batches, responses=second_responses)
    second_job = backend.submit_batch(backend.build_batch(second_requests))
    coordinator.mark_batch_submitted(second_job)
    second_results = backend.retrieve_results(backend.get_status(second_job))
    for result in reversed(second_results):
        assert coordinator.apply_result(result, backend).status is AgentRunStatus.COMPLETED

    for index, run_id in enumerate(run_ids):
        state = coordinator.states.load(run_id)
        assert state.workspace_checkpoint_generation == 2
        assert len(state.model_trace) == 2
        assert [event["tool_name"] for event in state.tool_trace] == [
            "write_text_file",
            "submit_result",
        ]
        restored = tmp_path / f"restored-{index}"
        restored.mkdir()
        WorkspaceCheckpoint(
            tmp_path / "runs" / run_id / "checkpoints",
            run_id,
        ).restore(2, restored)
        assert (restored / "output" / f"report-{index}.txt").read_text(
            encoding="utf-8"
        ) == f"run {index}\n"


def test_retryable_batch_item_reuses_body_with_a_new_attempt(tmp_path: Path) -> None:
    task_path = Path("examples/agent_demo/task.yaml").resolve()
    task = load_task(task_path)
    config = ModelRunConfig(
        provider=ModelProvider.SCRIPTED,
        model="offline-retry",
        transport=InferenceTransport.PROVIDER_BATCH,
    )
    coordinator, artifacts, batches = _coordinator(tmp_path, FakeSandboxAdapter())
    coordinator.enqueue(task, task_path.parent, config, run_id="retry-run")
    original = coordinator.materialize_batch_request(coordinator.ready_turn("retry-run"))
    failing = MockBatchBackend(
        artifacts,
        batches,
        errors={
            original.internal_request_id: {
                "code": "server_error",
                "message": "try again",
                "retryable": True,
            }
        },
    )
    job = failing.submit_batch(failing.build_batch((original,)))
    coordinator.mark_batch_submitted(job)
    retry_ready = coordinator.apply_result(
        failing.retrieve_results(failing.get_status(job))[0],
        failing,
    )

    assert retry_ready.status is AgentRunStatus.READY_FOR_MODEL
    retry_turn = coordinator.ready_turn("retry-run")
    assert retry_turn.attempt_number == 2
    retry = coordinator.materialize_batch_request(retry_turn)
    assert retry.internal_request_id != original.internal_request_id
    assert retry.compiled_request_body_sha256 == original.compiled_request_body_sha256
    assert retry.compiled_request_body_reference == original.compiled_request_body_reference
