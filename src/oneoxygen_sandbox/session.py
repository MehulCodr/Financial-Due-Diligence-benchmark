"""Sandbox lifecycle orchestration independent of Docker SDK details."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType

from oneoxygen_sandbox.config import configuration_hash
from oneoxygen_sandbox.docker_adapter import Container, DockerAdapter, DockerSDKAdapter
from oneoxygen_sandbox.errors import (
    CleanupError,
    LifecycleError,
    RecordPersistenceError,
    SandboxError,
    SandboxTimeoutError,
)
from oneoxygen_sandbox.filesystem import collect_output_artifacts, copy_input_assets
from oneoxygen_sandbox.models import (
    ArtifactMetadata,
    ErrorInformation,
    ExecResult,
    RunRecord,
    RunStatus,
    SandboxPolicy,
    SandboxTask,
    ToolEventStatus,
)


class _SessionState(StrEnum):
    NEW = "new"
    ACTIVE = "active"
    TERMINATED = "terminated"
    COLLECTING = "collecting"
    COLLECTED = "collected"
    STOPPED = "stopped"


class SandboxSession:
    """One isolated container and workspace with guaranteed cleanup."""

    def __init__(
        self,
        task: SandboxTask,
        task_directory: Path,
        runs_directory: Path = Path("runs"),
        adapter: DockerAdapter | None = None,
    ) -> None:
        self.task = task
        self.spec = task.sandbox
        self.task_directory = task_directory
        self.runs_directory = runs_directory
        self.adapter = adapter or DockerSDKAdapter()
        self.run_id = uuid.uuid4().hex
        self.run_directory = self.runs_directory / self.run_id
        self.workspace_path: Path | None = None
        self.container: Container | None = None
        self._record: RunRecord | None = None
        self._state = _SessionState.NEW
        self._container_stopped = False
        self._timer: threading.Timer | None = None
        self._deadline = 0.0
        self._overall_timed_out = threading.Event()

    @property
    def record(self) -> RunRecord:
        if self._record is None:
            raise LifecycleError("run record is unavailable before the session starts")
        return self._record

    @property
    def record_path(self) -> Path:
        return self.run_directory / "run.json"

    def start(self) -> SandboxSession:
        if self._state is not _SessionState.NEW:
            raise LifecycleError(f"cannot start a session in state {self._state}")
        self._record = RunRecord(
            run_id=self.run_id,
            task_id=self.spec.task_id,
            task_version=self.spec.task_version,
            requested_image=self.spec.image,
            task_configuration_hash=configuration_hash(self.task),
            start_timestamp=datetime.now(UTC),
            sandbox_policy=SandboxPolicy.from_spec(
                self.spec,
                non_root_user=str(getattr(self.adapter, "sandbox_user", "10001:10001")),
            ),
            tool_policy=self.task.tool_policy,
        )
        try:
            self.run_directory.mkdir(parents=True, exist_ok=False)
            self.workspace_path = Path(
                tempfile.mkdtemp(prefix=f"oneoxygen-{self.spec.task_id}-")
            ).resolve()
            copy_input_assets(self.task_directory, self.workspace_path, self.spec.input_assets)
            self.workspace_path.joinpath(*self.spec.working_relative_path.parts).mkdir(
                parents=True, exist_ok=True
            )
            self._prepare_workspace_ownership()
            environment = {
                name: os.environ[name]
                for name in self.spec.environment_allowlist
                if name in os.environ
            }
            self.record.resolved_image = self.adapter.resolve_image(self.spec.image)
            self.container = self.adapter.create_container(
                self.spec, self.workspace_path, environment, self.run_id
            )
            self.adapter.start_container(self.container)
            self._state = _SessionState.ACTIVE
            self._deadline = time.monotonic() + self.spec.overall_timeout_seconds
            self._timer = threading.Timer(
                self.spec.overall_timeout_seconds, self._expire_overall_timeout
            )
            self._timer.daemon = True
            self._timer.start()
            return self
        except BaseException as exc:
            normalized = self._normalize_error(exc, "could not start sandbox")
            self._set_error(normalized)
            with suppress(SandboxError):
                self.stop()
            raise normalized from exc

    def _prepare_workspace_ownership(self) -> None:
        """Keep Linux workspaces private while making them writable by the container UID."""
        sandbox_user = getattr(self.adapter, "sandbox_user", None)
        if os.name != "posix" or sandbox_user is None or self.workspace_path is None:
            return
        user_id_text, _, group_id_text = str(sandbox_user).partition(":")
        user_id = int(user_id_text)
        group_id = int(group_id_text or user_id_text)
        if user_id == 0:
            raise LifecycleError("sandbox adapter requested the root user")
        if os.geteuid() != 0:
            if user_id != os.geteuid():
                raise LifecycleError(
                    "sandbox UID cannot access the private workspace; use the calling UID"
                )
            return
        for directory, child_directories, filenames in os.walk(self.workspace_path):
            os.chown(directory, user_id, group_id, follow_symlinks=False)
            for name in (*child_directories, *filenames):
                os.chown(Path(directory) / name, user_id, group_id, follow_symlinks=False)

    def _expire_overall_timeout(self) -> None:
        if self._state is not _SessionState.ACTIVE:
            return
        self._overall_timed_out.set()
        if self.container is not None and not self._container_stopped:
            try:
                self.adapter.stop_container(self.container)
                self._container_stopped = True
                self._state = _SessionState.TERMINATED
            except SandboxError:
                # The foreground lifecycle retries cleanup and records any failure.
                pass

    @property
    def is_active(self) -> bool:
        return self._state is _SessionState.ACTIVE and not self._overall_timed_out.is_set()

    def execute(self, command: str) -> ExecResult:
        result = self._execute_container_command(
            command=command,
            working_directory=self.spec.working_directory,
            timeout_seconds=self.spec.command_timeout_seconds,
        )
        self.record.command_results.append(result)
        return result

    def execute_tool_command(
        self, command: str, working_directory: str, timeout_seconds: float
    ) -> ExecResult:
        return self._execute_container_command(command, working_directory, timeout_seconds)

    def _execute_container_command(
        self, command: str, working_directory: str, timeout_seconds: float
    ) -> ExecResult:
        if self._state is not _SessionState.ACTIVE:
            raise LifecycleError(f"cannot execute a command in state {self._state}")
        if self._overall_timed_out.is_set():
            raise SandboxTimeoutError("overall sandbox timeout expired")
        if not command.strip() or "\x00" in command:
            raise LifecycleError("command must be non-empty and contain no null bytes")
        if self.container is None:
            raise LifecycleError("sandbox container is missing")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._expire_overall_timeout()
            raise SandboxTimeoutError("overall sandbox timeout expired")
        timeout = min(timeout_seconds, remaining)
        execution_started_at = datetime.now(UTC)
        execution_started = time.monotonic()
        try:
            result = self.adapter.execute(
                self.container,
                command,
                working_directory,
                timeout,
                self.spec.maximum_output_size_bytes,
            )
        except SandboxError as exc:
            if not self._overall_timed_out.is_set():
                raise
            result = ExecResult(
                command=command,
                stdout="",
                stderr=str(exc),
                exit_code=124,
                start_timestamp=execution_started_at,
                end_timestamp=datetime.now(UTC),
                duration_seconds=time.monotonic() - execution_started,
                timed_out=True,
            )
        if self._overall_timed_out.is_set() and not result.timed_out:
            result = result.model_copy(update={"timed_out": True, "exit_code": 124})
        if result.timed_out:
            self._container_stopped = True
            self._state = _SessionState.TERMINATED
        return result

    def collect_artifacts(self) -> list[ArtifactMetadata]:
        if self._state not in {_SessionState.ACTIVE, _SessionState.TERMINATED}:
            raise LifecycleError(f"cannot collect artifacts in state {self._state}")
        if self.container is None or self.workspace_path is None:
            raise LifecycleError("sandbox resources are missing")
        self._state = _SessionState.COLLECTING
        if self._timer is not None:
            self._timer.cancel()
        if not self._container_stopped:
            self.adapter.stop_container(self.container)
            self._container_stopped = True
        output_root = self.workspace_path.joinpath(*self.spec.output_relative_path.parts)
        artifacts = collect_output_artifacts(
            output_root,
            self.run_directory / "artifacts",
            self.spec.maximum_output_size_bytes,
        )
        self.record.artifacts = artifacts
        self._state = _SessionState.COLLECTED
        return artifacts

    def stop(self) -> None:
        if self._state is _SessionState.STOPPED:
            self._finish_record()
            self._write_record()
            return
        if self._timer is not None:
            self._timer.cancel()
        failures: list[str] = []
        if self.container is not None:
            if not self._container_stopped:
                try:
                    self.adapter.stop_container(self.container)
                    self._container_stopped = True
                except SandboxError as exc:
                    failures.append(str(exc))
            try:
                self.adapter.remove_container(self.container)
            except SandboxError as exc:
                failures.append(str(exc))
        if self.workspace_path is not None and self.workspace_path.exists():
            try:
                shutil.rmtree(self.workspace_path)
            except OSError as exc:
                failures.append(f"cannot remove temporary workspace: {exc}")
        self._state = _SessionState.STOPPED
        cleanup_error = CleanupError("; ".join(failures)) if failures else None
        if cleanup_error is not None:
            self._append_cleanup_error(cleanup_error)
        self._finish_record()
        try:
            self._write_record()
        except OSError as exc:
            raise RecordPersistenceError(f"cannot write run record: {exc}") from exc
        if cleanup_error is not None:
            raise cleanup_error

    def _normalize_error(self, exc: BaseException, prefix: str) -> SandboxError:
        if isinstance(exc, SandboxError):
            return exc
        return LifecycleError(f"{prefix}: {type(exc).__name__}: {exc}")

    def _set_error(self, exc: BaseException) -> None:
        if self._record is None:
            return
        code = exc.code if isinstance(exc, SandboxError) else "unexpected_error"
        self.record.error = ErrorInformation(type=type(exc).__name__, code=code, message=str(exc))

    def _append_cleanup_error(self, exc: SandboxError) -> None:
        if self._record is None:
            return
        if self.record.error is None:
            self._set_error(exc)
            return
        self.record.error = self.record.error.model_copy(
            update={"message": f"{self.record.error.message}; cleanup: {exc}"}
        )

    def _finish_record(self) -> None:
        if self._record is None or self.record.end_timestamp is not None:
            return
        self.record.end_timestamp = datetime.now(UTC)
        if self._overall_timed_out.is_set() and self.record.error is None:
            self._set_error(SandboxTimeoutError("overall sandbox timeout expired"))
        if self.record.error is not None:
            self.record.final_status = (
                RunStatus.TIMED_OUT
                if self.record.error.code == SandboxTimeoutError.code
                else RunStatus.ERROR
            )
        elif any(result.timed_out for result in self.record.command_results):
            self.record.final_status = RunStatus.TIMED_OUT
        elif any(result.exit_code != 0 for result in self.record.command_results) or any(
            event.status is ToolEventStatus.FAILED for event in self.record.tool_events
        ):
            self.record.final_status = RunStatus.FAILED
        else:
            self.record.final_status = RunStatus.SUCCEEDED

    def _write_record(self) -> None:
        if self._record is None:
            return
        self.run_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = self.run_directory / ".run.json.tmp"
        serialized = json.dumps(
            self.record.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        temporary_path.write_text(f"{serialized}\n", encoding="utf-8")
        temporary_path.replace(self.record_path)

    def __enter__(self) -> SandboxSession:
        return self.start()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exception is not None:
            self._set_error(exception)
        cleanup_error: SandboxError | None = None
        try:
            self.stop()
        except SandboxError as exc:
            cleanup_error = exc
        if exception is None and cleanup_error is not None:
            raise cleanup_error
        return False
