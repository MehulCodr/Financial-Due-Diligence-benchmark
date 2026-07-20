from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

docker = pytest.importorskip("docker")

from docker.errors import NotFound  # noqa: E402

from oneoxygen_sandbox.config import load_task  # noqa: E402
from oneoxygen_sandbox.docker_adapter import DockerSDKAdapter  # noqa: E402
from oneoxygen_sandbox.models import (  # noqa: E402
    RunStatus,
    SandboxSpec,
    SandboxTask,
    ToolCall,
    ToolErrorCode,
    ToolPolicy,
)
from oneoxygen_sandbox.session import SandboxSession  # noqa: E402
from oneoxygen_sandbox.tools import ToolDispatcher  # noqa: E402

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


def tool_task(policy: ToolPolicy) -> SandboxTask:
    spec = SandboxSpec(
        image="oneoxygen-sandbox:phase1",
        task_id="tool-integration",
        task_version="1",
        memory_limit_bytes=134217728,
        cpu_limit=0.5,
        pid_limit=32,
        overall_timeout_seconds=60,
    )
    return SandboxTask(sandbox=spec, tool_policy=policy)


def call(call_id: str, tool_name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments)


def test_tools_execute_inside_hardened_container_and_submit(
    tmp_path: Path, docker_adapter: DockerSDKAdapter
) -> None:
    policy = ToolPolicy(
        allowed_tool_names=(
            "execute_shell",
            "execute_python",
            "read_text_file",
            "write_text_file",
            "submit_result",
        ),
        shell_execution_allowed=True,
        python_execution_allowed=True,
        shell_timeout_seconds=5,
        python_timeout_seconds=5,
    )
    session = SandboxSession(tool_task(policy), tmp_path, tmp_path / "runs", docker_adapter)

    with session:
        assert session.container is not None
        container_id = session.container.id
        dispatcher = ToolDispatcher(session)
        uid = dispatcher.dispatch(call("call-1", "execute_shell", {"command": "id -u"}))
        state = dispatcher.dispatch(
            call("call-2", "execute_shell", {"command": "printf persistent > state.txt"})
        )
        python = dispatcher.dispatch(
            call(
                "call-3",
                "execute_python",
                {
                    "source_code": (
                        "from pathlib import Path\n"
                        "state = Path('/workspace/state.txt').read_text().strip()\n"
                        "Path('/workspace/output').mkdir(exist_ok=True)\n"
                        "Path('/workspace/output/findings.md').write_text("
                        "f'# Finding\\n\\nState: {state}\\n', encoding='utf-8')\n"
                        "print(state)\n"
                    )
                },
            )
        )
        network = dispatcher.dispatch(
            call(
                "call-4",
                "execute_shell",
                {
                    "command": (
                        'python -c "import socket; '
                        "s=socket.socket(); s.settimeout(1); s.connect(('1.1.1.1', 53))\""
                    )
                },
            )
        )
        root_write = dispatcher.dispatch(
            call("call-5", "execute_shell", {"command": "touch /root-denied"})
        )
        readback = dispatcher.dispatch(
            call("call-6", "read_text_file", {"path": "output/findings.md"})
        )
        submitted = dispatcher.dispatch(
            call(
                "call-7",
                "submit_result",
                {"summary": "done", "artifact_paths": ["output/findings.md"]},
            )
        )
        artifacts = session.collect_artifacts()

    assert uid.success is True
    assert uid.content["stdout"].strip() != "0"
    assert state.success is True
    assert python.success is True
    assert "persistent" in python.content["stdout"]
    assert network.error is not None
    assert network.error.code is ToolErrorCode.EXECUTION_FAILED
    assert root_write.error is not None
    assert root_write.error.code is ToolErrorCode.EXECUTION_FAILED
    assert readback.success is True
    assert submitted.success is True
    assert artifacts[0].relative_path == "findings.md"
    assert session.record.submission is not None
    assert len(session.record.tool_events) == 7
    assert session.record.final_status is RunStatus.FAILED
    with pytest.raises(NotFound):
        docker_adapter.client.containers.get(container_id)


def test_scripted_tool_demo_succeeds_and_records_trace(
    project_root: Path, tmp_path: Path, docker_adapter: DockerSDKAdapter
) -> None:
    task_path = project_root / "examples" / "tool_demo" / "task.yaml"
    calls_data = yaml.safe_load(
        (project_root / "examples" / "tool_demo" / "scripted_calls.yaml").read_text(
            encoding="utf-8"
        )
    )["calls"]
    task = load_task(task_path)
    session = SandboxSession(task, task_path.parent, tmp_path / "runs", docker_adapter)

    with session:
        dispatcher = ToolDispatcher(session)
        results = [
            dispatcher.dispatch(
                ToolCall(
                    call_id=item["call_id"],
                    tool_name=item["tool_name"],
                    arguments=item["arguments"],
                )
            )
            for item in calls_data
        ]
        artifacts = session.collect_artifacts()

    assert all(result.success for result in results)
    assert session.record.final_status is RunStatus.SUCCEEDED
    assert session.record.submission is not None
    assert artifacts[0].relative_path == "findings.md"
    assert "Revenue grew 20.0%" in (session.run_directory / "artifacts" / "findings.md").read_text(
        encoding="utf-8"
    )
    assert len(session.record.tool_events) == len(calls_data)


def test_tool_failure_and_timeout_cleanup(tmp_path: Path, docker_adapter: DockerSDKAdapter) -> None:
    failure_policy = ToolPolicy(allowed_tool_names=("list_files",))
    failure_session = SandboxSession(
        tool_task(failure_policy), tmp_path, tmp_path / "failure-runs", docker_adapter
    )
    with failure_session:
        assert failure_session.container is not None
        failure_container_id = failure_session.container.id
        dispatcher = ToolDispatcher(failure_session)
        result = dispatcher.dispatch(call("call-1", "execute_shell", {"command": "true"}))
        assert result.error is not None
        assert result.error.code is ToolErrorCode.TOOL_NOT_ALLOWED

    timeout_policy = ToolPolicy(
        allowed_tool_names=("execute_shell",),
        shell_execution_allowed=True,
        shell_timeout_seconds=0.5,
    )
    timeout_session = SandboxSession(
        tool_task(timeout_policy), tmp_path, tmp_path / "timeout-runs", docker_adapter
    )
    with timeout_session:
        assert timeout_session.container is not None
        timeout_container_id = timeout_session.container.id
        dispatcher = ToolDispatcher(timeout_session)
        result = dispatcher.dispatch(call("call-1", "execute_shell", {"command": "sleep 5"}))
        assert result.error is not None
        assert result.error.code is ToolErrorCode.EXECUTION_TIMEOUT

    with pytest.raises(NotFound):
        docker_adapter.client.containers.get(failure_container_id)
    with pytest.raises(NotFound):
        docker_adapter.client.containers.get(timeout_container_id)
