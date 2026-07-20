from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from oneoxygen_sandbox.errors import ToolFailure
from oneoxygen_sandbox.models import (
    ExecResult,
    InputAsset,
    RunRecord,
    SandboxSpec,
    SandboxTask,
    ToolCall,
    ToolErrorCode,
    ToolPolicy,
)
from oneoxygen_sandbox.session import SandboxSession
from oneoxygen_sandbox.tools import ToolDispatcher, ToolRegistry, default_tool_registry
from oneoxygen_sandbox.tools.base import BaseTool, ToolContext


class FakeContainer:
    id = "fake-container"


class FakeDockerAdapter:
    def __init__(self, *, delay_seconds: float = 0) -> None:
        self.container = FakeContainer()
        self.workspace: Path | None = None
        self.delay_seconds = delay_seconds

    def check_available(self) -> dict[str, Any]:
        return {"OSType": "linux"}

    def resolve_image(self, image: str) -> str:
        return "sha256:resolved"

    def create_container(
        self, spec: SandboxSpec, workspace: Path, environment: dict[str, str], run_id: str
    ) -> FakeContainer:
        self.workspace = workspace
        return self.container

    def start_container(self, container: FakeContainer) -> None:
        return None

    def execute(
        self,
        container: FakeContainer,
        command: str,
        working_directory: str,
        timeout_seconds: float,
        maximum_output_bytes: int,
    ) -> ExecResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        now = datetime.now(UTC)
        return ExecResult(
            command=command,
            stdout="ok",
            stderr="",
            exit_code=0,
            start_timestamp=now,
            end_timestamp=now,
            duration_seconds=0,
        )

    def stop_container(self, container: FakeContainer) -> None:
        return None

    def remove_container(self, container: FakeContainer) -> None:
        return None


def make_session(tmp_path: Path, *, policy: ToolPolicy | None = None) -> SandboxSession:
    task_directory = tmp_path / "task"
    task_directory.mkdir()
    spec = SandboxSpec(
        image="image:tag",
        task_id="tool-task",
        task_version="1",
        input_assets=(InputAsset(source="seed.txt", destination="seed.txt"),),
    )
    (task_directory / "seed.txt").write_text("seed\n", encoding="utf-8")
    task = SandboxTask(sandbox=spec, tool_policy=policy or ToolPolicy())
    return SandboxSession(task, task_directory, tmp_path / "runs", FakeDockerAdapter())


def dispatch(session: SandboxSession, tool_name: str, arguments: dict[str, Any]) -> Any:
    dispatcher = ToolDispatcher(session)
    return dispatcher.dispatch(
        ToolCall(
            call_id=f"call-{len(session.record.tool_events) + 1}",
            tool_name=tool_name,
            arguments=arguments,
        )
    )


def test_tool_schema_generation_is_deterministic() -> None:
    registry = default_tool_registry()
    first = json.dumps(registry.provider_schemas(), sort_keys=True)
    second = json.dumps(registry.provider_schemas(), sort_keys=True)

    assert first == second
    assert registry.names() == tuple(sorted(registry.names()))
    assert "execute_python" in first
    assert "source_code" in first


def test_duplicate_tool_registration_is_rejected() -> None:
    registry = ToolRegistry()
    tool = default_tool_registry().resolve("list_files")
    registry.register(tool)

    with pytest.raises(ToolFailure, match="duplicate tool"):
        registry.register(tool)


def test_unknown_tool_returns_structured_error(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        result = dispatch(session, "missing_tool", {})

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.UNKNOWN_TOOL


def test_malformed_arguments_return_invalid_arguments(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        result = dispatch(
            session, "read_text_file", {"path": "seed.txt", "start_line": 5, "end_line": 2}
        )

    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS


def test_dangerous_tools_are_not_allowed_by_default(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        result = dispatch(session, "execute_python", {"source_code": "print('nope')"})

    assert result.error is not None
    assert result.error.code is ToolErrorCode.TOOL_NOT_ALLOWED


def test_total_and_per_tool_call_limits(tmp_path: Path) -> None:
    policy = ToolPolicy(max_total_tool_calls=1, per_tool_call_limits={"list_files": 1})
    with make_session(tmp_path, policy=policy) as session:
        dispatcher = ToolDispatcher(session)
        first = dispatcher.dispatch(
            ToolCall(call_id="call-1", tool_name="list_files", arguments={})
        )
        second = dispatcher.dispatch(
            ToolCall(call_id="call-2", tool_name="list_files", arguments={})
        )

    assert first.success is True
    assert second.error is not None
    assert second.error.code is ToolErrorCode.CALL_LIMIT_EXCEEDED


def test_calls_after_submission_are_rejected(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        output = session.workspace_path / "output"
        output.mkdir()
        (output / "findings.md").write_text("done", encoding="utf-8")
        dispatcher = ToolDispatcher(session)
        submitted = dispatcher.dispatch(
            ToolCall(
                call_id="call-1",
                tool_name="submit_result",
                arguments={"summary": "done", "artifact_paths": ["output/findings.md"]},
            )
        )
        rejected = dispatcher.dispatch(
            ToolCall(call_id="call-2", tool_name="list_files", arguments={})
        )

    assert submitted.success is True
    assert rejected.error is not None
    assert rejected.error.code is ToolErrorCode.ALREADY_SUBMITTED


def test_submission_state_survives_new_dispatcher(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        output = session.workspace_path / "output"
        output.mkdir()
        (output / "findings.md").write_text("done", encoding="utf-8")
        first_dispatcher = ToolDispatcher(session)
        submitted = first_dispatcher.dispatch(
            ToolCall(
                call_id="call-1",
                tool_name="submit_result",
                arguments={"summary": "done", "artifact_paths": ["output/findings.md"]},
            )
        )
        second_dispatcher = ToolDispatcher(session)
        rejected = second_dispatcher.dispatch(
            ToolCall(call_id="call-2", tool_name="list_files", arguments={})
        )

    assert submitted.success is True
    assert rejected.error is not None
    assert rejected.error.code is ToolErrorCode.ALREADY_SUBMITTED


@pytest.mark.parametrize(
    "path", ["../seed.txt", "/workspace/seed.txt", "C:\\seed.txt", ".oneoxygen/state"]
)
def test_model_facing_paths_reject_unsafe_inputs(tmp_path: Path, path: str) -> None:
    with make_session(tmp_path) as session:
        result = dispatch(session, "read_text_file", {"path": path})

    assert result.error is not None
    assert result.error.code is ToolErrorCode.PATH_NOT_ALLOWED


def test_binary_files_are_rejected(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        assert session.workspace_path is not None
        (session.workspace_path / "binary.dat").write_bytes(b"abc\x00def")
        binary_result = dispatch(session, "read_text_file", {"path": "binary.dat"})

    assert binary_result.error is not None
    assert binary_result.error.code is ToolErrorCode.BINARY_FILE


def test_symlink_files_are_rejected(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        assert session.workspace_path is not None
        target = session.workspace_path / "seed.txt"
        link = session.workspace_path / "linked.txt"
        try:
            link.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"symbolic links unavailable: {exc}")
        link_result = dispatch(session, "read_text_file", {"path": "linked.txt"})

    assert link_result.error is not None
    assert link_result.error.code is ToolErrorCode.PATH_NOT_ALLOWED


def test_read_truncation_and_result_truncation(tmp_path: Path) -> None:
    policy = ToolPolicy(max_read_size_bytes=20, max_tool_result_size_bytes=256)
    with make_session(tmp_path, policy=policy) as session:
        assert session.workspace_path is not None
        (session.workspace_path / "large.txt").write_text("line\n" * 100, encoding="utf-8")
        result = dispatch(session, "read_text_file", {"path": "large.txt"})

    assert result.success is True
    assert result.truncated is True


def test_write_size_enforcement(tmp_path: Path) -> None:
    policy = ToolPolicy(max_write_size_bytes=5)
    with make_session(tmp_path, policy=policy) as session:
        result = dispatch(
            session,
            "write_text_file",
            {"path": "out.txt", "content": "too large", "overwrite": True},
        )

    assert result.error is not None
    assert result.error.code is ToolErrorCode.SIZE_LIMIT_EXCEEDED


def test_write_text_file_creates_parent_and_returns_hash(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        result = dispatch(
            session,
            "write_text_file",
            {
                "path": "notes/finding.md",
                "content": "finding\n",
                "overwrite": False,
                "create_parents": True,
            },
        )

    assert result.success is True
    assert result.content["relative_path"] == "notes/finding.md"
    assert result.content["size_bytes"] == 8
    assert len(result.content["sha256"]) == 64


def test_replace_requires_exact_count_and_preserves_file(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        assert session.workspace_path is not None
        path = session.workspace_path / "replace.txt"
        path.write_text("alpha alpha\n", encoding="utf-8")
        result = dispatch(
            session,
            "replace_text",
            {
                "path": "replace.txt",
                "old_text": "alpha",
                "replacement_text": "beta",
                "expected_replacements": 1,
            },
        )
        after = path.read_text(encoding="utf-8")

    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS
    assert after == "alpha alpha\n"


def test_trace_serialization_and_error_sanitization(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        result = dispatch(session, "read_text_file", {"path": "missing.txt"})
        record_json = session.record.model_dump_json()
        restored = RunRecord.model_validate_json(record_json)

    assert result.error is not None
    assert result.error.code is ToolErrorCode.FILE_NOT_FOUND
    assert str(tmp_path) not in result.error.message
    assert restored.tool_events[0].error_code is ToolErrorCode.FILE_NOT_FOUND


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExplodingTool(BaseTool):
    name = "explode"
    description = "Raise an internal exception for sanitization tests."
    argument_model = EmptyArgs

    def __init__(self, leaked_path: Path) -> None:
        self.leaked_path = leaked_path

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        raise RuntimeError(f"internal path leaked here: {self.leaked_path}")


def test_internal_errors_are_sanitized(tmp_path: Path) -> None:
    policy = ToolPolicy(allowed_tool_names=("explode",))
    registry = ToolRegistry()
    registry.register(ExplodingTool(tmp_path))
    with make_session(tmp_path, policy=policy) as session:
        dispatcher = ToolDispatcher(session, registry=registry)
        result = dispatcher.dispatch(ToolCall(call_id="call-1", tool_name="explode", arguments={}))

    assert result.error is not None
    assert result.error.code is ToolErrorCode.INTERNAL_TOOL_ERROR
    assert str(tmp_path) not in result.error.message


def test_submission_validation_and_duplicate_artifacts(tmp_path: Path) -> None:
    with make_session(tmp_path) as session:
        assert session.workspace_path is not None
        output = session.workspace_path / "output"
        output.mkdir()
        (output / "findings.md").write_text("done", encoding="utf-8")
        duplicate = dispatch(
            session,
            "submit_result",
            {
                "summary": "done",
                "artifact_paths": ["output/findings.md", "output/findings.md"],
            },
        )

    assert duplicate.error is not None
    assert duplicate.error.code is ToolErrorCode.INVALID_ARGUMENTS
