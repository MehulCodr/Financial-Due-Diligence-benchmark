"""Standard provider-neutral tools for active sandbox sessions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oneoxygen_sandbox.errors import ToolFailure
from oneoxygen_sandbox.models import ExecResult, ToolErrorCode
from oneoxygen_sandbox.tools.base import BaseTool, ToolContext


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListFilesArgs(ToolArgs):
    path: str = "."
    max_depth: int = Field(default=3, ge=0, le=20)


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files and directories inside the sandbox workspace without host paths."
    argument_model = ListFilesArgs

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        args = ListFilesArgs.model_validate(arguments)
        entries, truncated = context.workspace.list_files(
            args.path, args.max_depth, context.policy.max_file_list_entries
        )
        return {"path": args.path, "entries": entries, "truncated": truncated}


class ReadTextFileArgs(ToolArgs):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> ReadTextFileArgs:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ReadTextFileTool(BaseTool):
    name = "read_text_file"
    description = "Read a bounded UTF-8 text file from the sandbox workspace."
    argument_model = ReadTextFileArgs

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        args = ReadTextFileArgs.model_validate(arguments)
        result = context.workspace.read_text_file(
            args.path,
            start_line=args.start_line,
            end_line=args.end_line,
            max_read_size=context.policy.max_read_size_bytes,
        )
        return {
            "path": result.path,
            "requested_start_line": result.requested_start_line,
            "requested_end_line": result.requested_end_line,
            "lines": result.lines,
            "total_line_count": result.total_line_count,
            "truncated": result.truncated,
        }


class WriteTextFileArgs(ToolArgs):
    path: str
    content: str
    overwrite: bool = False
    create_parents: bool = False


class WriteTextFileTool(BaseTool):
    name = "write_text_file"
    description = "Atomically write UTF-8 text into an allowed workspace file."
    argument_model = WriteTextFileArgs

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        args = WriteTextFileArgs.model_validate(arguments)
        metadata = context.workspace.write_text_file(
            args.path,
            args.content,
            overwrite=args.overwrite,
            create_parents=args.create_parents,
            max_write_size=context.policy.max_write_size_bytes,
        )
        return metadata.model_dump(mode="json")


class ReplaceTextArgs(ToolArgs):
    path: str
    old_text: str = Field(min_length=1)
    replacement_text: str
    expected_replacements: int = Field(ge=1)


class ReplaceTextTool(BaseTool):
    name = "replace_text"
    description = "Atomically replace exact text only when the expected count matches."
    argument_model = ReplaceTextArgs

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        args = ReplaceTextArgs.model_validate(arguments)
        count, metadata = context.workspace.replace_text(
            args.path,
            args.old_text,
            args.replacement_text,
            expected_replacements=args.expected_replacements,
            max_read_size=context.policy.max_read_size_bytes,
            max_write_size=context.policy.max_write_size_bytes,
        )
        return {"replacement_count": count, "file": metadata.model_dump(mode="json")}


class ExecuteShellArgs(ToolArgs):
    command: str = Field(min_length=1)
    working_directory: str = "."
    timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("command may not contain null bytes")
        return value


def _exec_content(result: ExecResult) -> dict[str, Any]:
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "timed_out": result.timed_out,
        "output_truncated": result.output_truncated,
    }


class ExecuteShellTool(BaseTool):
    name = "execute_shell"
    description = "Execute a shell command inside the active sandbox container."
    argument_model = ExecuteShellArgs

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        args = ExecuteShellArgs.model_validate(arguments)
        working_directory = context.workspace.container_working_directory(args.working_directory)
        timeout = min(
            args.timeout_seconds or context.policy.shell_timeout_seconds,
            context.policy.shell_timeout_seconds,
        )
        result = context.session.execute_tool_command(
            args.command,
            working_directory,
            timeout,
        )
        content = _exec_content(result)
        if result.timed_out:
            raise ToolFailure(
                ToolErrorCode.EXECUTION_TIMEOUT.value,
                "shell command timed out",
                content=content,
                truncated=result.output_truncated,
            )
        if result.exit_code != 0:
            raise ToolFailure(
                ToolErrorCode.EXECUTION_FAILED.value,
                "shell command exited with a non-zero status",
                content=content,
                truncated=result.output_truncated,
            )
        return content


class ExecutePythonArgs(ToolArgs):
    source_code: str = Field(min_length=1)
    working_directory: str = "."
    timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("source_code")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("source code may not contain null bytes")
        return value


class ExecutePythonTool(BaseTool):
    name = "execute_python"
    description = "Execute a Python source snippet inside the active sandbox container."
    argument_model = ExecutePythonArgs

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        args = ExecutePythonArgs.model_validate(arguments)
        working_directory = context.workspace.container_working_directory(args.working_directory)
        timeout = min(
            args.timeout_seconds or context.policy.python_timeout_seconds,
            context.policy.python_timeout_seconds,
        )
        runtime_file = context.workspace.create_runtime_file(
            args.source_code,
            context.policy.max_write_size_bytes,
        )
        try:
            result = context.session.execute_tool_command(
                f"python {context.workspace.container_path(runtime_file)}",
                working_directory,
                timeout,
            )
        finally:
            context.workspace.remove_runtime_file(runtime_file)
        content = _exec_content(result)
        if result.timed_out:
            raise ToolFailure(
                ToolErrorCode.EXECUTION_TIMEOUT.value,
                "Python execution timed out",
                content=content,
                truncated=result.output_truncated,
            )
        if result.exit_code != 0:
            raise ToolFailure(
                ToolErrorCode.EXECUTION_FAILED.value,
                "Python exited with a non-zero status",
                content=content,
                truncated=result.output_truncated,
            )
        return content


class SubmitResultArgs(ToolArgs):
    summary: str = Field(min_length=1, max_length=4_000)
    artifact_paths: tuple[str, ...] = Field(min_length=1, max_length=100)
    findings: dict[str, Any] | None = None


class SubmitResultTool(BaseTool):
    name = "submit_result"
    description = "Submit final findings and approved workspace output artifacts."
    argument_model = SubmitResultArgs

    def execute(self, arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
        args = SubmitResultArgs.model_validate(arguments)
        if context.submission_state.submitted:
            raise ToolFailure(
                ToolErrorCode.ALREADY_SUBMITTED.value,
                "a result has already been submitted",
            )
        artifacts = context.workspace.validate_submitted_artifacts(
            args.artifact_paths,
            context.session.spec.maximum_output_size_bytes,
        )
        context.submission_state.submitted = True
        return {
            "summary": args.summary,
            "artifact_paths": list(args.artifact_paths),
            "findings": args.findings,
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            "submitted": True,
        }
