from __future__ import annotations

from pathlib import Path

import pytest

docker = pytest.importorskip("docker")

from oneoxygen_sandbox.batching import (  # noqa: E402
    BatchArtifactStore,
    MockBatchBackend,
    SQLiteBatchStore,
)
from oneoxygen_sandbox.coordinator import DurableAgentCoordinator  # noqa: E402
from oneoxygen_sandbox.docker_adapter import DockerSDKAdapter  # noqa: E402
from oneoxygen_sandbox.models import (  # noqa: E402
    AgentTaskSpec,
    InferenceTransport,
    ModelProvider,
    ModelRunConfig,
    ModelTurnResponse,
    NormalizedFinishReason,
    SandboxSpec,
    SandboxTask,
    ToolCall,
    ToolPolicy,
)
from oneoxygen_sandbox.orchestration import (  # noqa: E402
    AgentRunStatus,
    SQLiteAgentStateStore,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def docker_adapter(project_root: Path) -> DockerSDKAdapter:
    try:
        adapter = DockerSDKAdapter()
        adapter.check_available()
    except Exception as exc:
        pytest.skip(f"Docker Linux engine is unavailable: {exc}")
    adapter.build_image(project_root / "docker", "oneoxygen-sandbox:phase1")
    return adapter


def test_batch_turns_restore_workspace_and_leave_no_containers(
    tmp_path: Path, docker_adapter: DockerSDKAdapter
) -> None:
    task_directory = tmp_path / "task"
    task_directory.mkdir()
    (task_directory / "task.md").write_text("Use only synthetic data.", encoding="utf-8")
    task = SandboxTask(
        sandbox=SandboxSpec(
            image="oneoxygen-sandbox:phase1",
            task_id="batch-checkpoint-docker",
            task_version="1",
            overall_timeout_seconds=30,
        ),
        tool_policy=ToolPolicy(
            allowed_tool_names=("write_text_file", "submit_result"),
            max_total_tool_calls=4,
        ),
        agent=AgentTaskSpec(
            instruction_file="task.md",
            maximum_model_turns=2,
            maximum_provider_requests=2,
            overall_wall_time_seconds=30,
            data_classification="synthetic",
        ),
    )
    root = tmp_path / "state"
    states = SQLiteAgentStateStore(str(root / "agents.sqlite3"))
    artifacts = BatchArtifactStore(root / "files")
    batches = SQLiteBatchStore(root / "batches.sqlite3")
    coordinator = DurableAgentCoordinator(
        states,
        artifacts,
        root / "runs",
        sandbox_adapter=docker_adapter,
    )
    config = ModelRunConfig(
        provider=ModelProvider.SCRIPTED,
        model="scripted-docker-batch",
        transport=InferenceTransport.PROVIDER_BATCH,
    )
    coordinator.enqueue(task, task_directory, config, run_id="docker-run")

    first = coordinator.materialize_batch_request(coordinator.ready_turn("docker-run"))
    response = ModelTurnResponse(
        provider=ModelProvider.SCRIPTED,
        requested_model=config.model,
        returned_model=config.model,
        finish_reason=NormalizedFinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                call_id="write",
                tool_name="write_text_file",
                arguments={
                    "path": "output/result.txt",
                    "content": "survives\n",
                    "create_parents": True,
                },
            ),
        ),
    )
    backend = MockBatchBackend(
        artifacts,
        batches,
        responses={first.internal_request_id: response.model_dump(mode="json")},
    )
    job = backend.submit_batch(backend.build_batch((first,)))
    coordinator.mark_batch_submitted(job)
    assert not docker_adapter.client.containers.list(
        all=True, filters={"label": "com.oneoxygen.sandbox=true"}
    )
    state = coordinator.apply_result(backend.retrieve_results(backend.get_status(job))[0], backend)
    assert state.status is AgentRunStatus.READY_FOR_MODEL
    assert not docker_adapter.client.containers.list(
        all=True, filters={"label": "com.oneoxygen.sandbox=true"}
    )

    second = coordinator.materialize_batch_request(coordinator.ready_turn("docker-run"))
    final_response = ModelTurnResponse(
        provider=ModelProvider.SCRIPTED,
        requested_model=config.model,
        returned_model=config.model,
        finish_reason=NormalizedFinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                call_id="submit",
                tool_name="submit_result",
                arguments={
                    "summary": "done",
                    "artifact_paths": ["output/result.txt"],
                },
            ),
        ),
    )
    backend = MockBatchBackend(
        artifacts,
        batches,
        responses={second.internal_request_id: final_response.model_dump(mode="json")},
    )
    job = backend.submit_batch(backend.build_batch((second,)))
    coordinator.mark_batch_submitted(job)
    completed = coordinator.apply_result(
        backend.retrieve_results(backend.get_status(job))[0], backend
    )
    assert completed.status is AgentRunStatus.COMPLETED
    assert len(completed.model_trace) == 2
    assert [event["tool_name"] for event in completed.tool_trace] == [
        "write_text_file",
        "submit_result",
    ]
    assert any(
        path.read_text(encoding="utf-8") == "survives\n"
        for path in (root / "runs" / "docker-run" / "turn-executions").rglob("artifacts/result.txt")
    )
    assert not docker_adapter.client.containers.list(
        all=True, filters={"label": "com.oneoxygen.sandbox=true"}
    )

    coordinator.enqueue(task, task_directory, config, run_id="expired-run")
    expired_request = coordinator.materialize_batch_request(coordinator.ready_turn("expired-run"))
    expiring = MockBatchBackend(artifacts, batches, expire=True)
    expired_job = expiring.submit_batch(expiring.build_batch((expired_request,)))
    coordinator.mark_batch_submitted(expired_job)
    assert expiring.get_status(expired_job).state.value == "expired"
    assert coordinator.expire_waiting_run("expired-run").status is AgentRunStatus.EXPIRED
    assert not docker_adapter.client.containers.list(
        all=True, filters={"label": "com.oneoxygen.sandbox=true"}
    )

    for run_id in ("partial-success", "partial-failure"):
        coordinator.enqueue(task, task_directory, config, run_id=run_id)
    partial_requests = tuple(
        coordinator.materialize_batch_request(coordinator.ready_turn(run_id))
        for run_id in ("partial-success", "partial-failure")
    )
    partial_response = ModelTurnResponse(
        provider=ModelProvider.SCRIPTED,
        requested_model=config.model,
        returned_model=config.model,
        finish_reason=NormalizedFinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                call_id="partial-write",
                tool_name="write_text_file",
                arguments={
                    "path": "output/partial.txt",
                    "content": "partial\n",
                    "create_parents": True,
                },
            ),
        ),
    )
    partial_backend = MockBatchBackend(
        artifacts,
        batches,
        responses={
            partial_requests[0].internal_request_id: partial_response.model_dump(mode="json")
        },
        errors={
            partial_requests[1].internal_request_id: {
                "code": "invalid_request",
                "message": "permanent item failure",
                "retryable": False,
            }
        },
    )
    partial_job = partial_backend.submit_batch(partial_backend.build_batch(partial_requests))
    coordinator.mark_batch_submitted(partial_job)
    partial_results = partial_backend.retrieve_results(partial_backend.get_status(partial_job))
    partial_states = {}
    for result in partial_results:
        applied = coordinator.apply_result(result, partial_backend)
        partial_states[applied.run_id] = applied
    assert partial_states["partial-success"].status is AgentRunStatus.READY_FOR_MODEL
    assert partial_states["partial-failure"].status is AgentRunStatus.FAILED
    assert not docker_adapter.client.containers.list(
        all=True, filters={"label": "com.oneoxygen.sandbox=true"}
    )
