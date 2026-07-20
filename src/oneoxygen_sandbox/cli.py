"""Command-line interface for the One Oxygen sandbox."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from oneoxygen_sandbox.config import load_task
from oneoxygen_sandbox.docker_adapter import DockerSDKAdapter
from oneoxygen_sandbox.errors import ConfigurationError, SandboxError
from oneoxygen_sandbox.models import RunStatus, ToolCall
from oneoxygen_sandbox.session import SandboxSession
from oneoxygen_sandbox.tools import ToolDispatcher, default_tool_registry

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Secure local Docker sandbox runner for One Oxygen.",
)
tools_app = typer.Typer(help="Inspect provider-independent tool definitions.")
app.add_typer(tools_app, name="tools")


def _print_captured_output(value: str, *, error: bool = False) -> None:
    maximum_display_characters = 4_000
    displayed = value.rstrip()
    if len(displayed) > maximum_display_characters:
        displayed = f"{displayed[:maximum_display_characters]}\n... CLI display truncated ..."
    if displayed:
        typer.echo(displayed, err=error)


@app.command()
def doctor() -> None:
    """Check that a compatible Docker engine is available."""
    try:
        info = DockerSDKAdapter().check_available()
    except SandboxError as exc:
        typer.secho(f"Docker unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(
        f"Docker ready: {info.get('ServerVersion', 'unknown')} "
        f"({info.get('OSType', 'unknown')}/{info.get('Architecture', 'unknown')})",
        fg=typer.colors.GREEN,
    )


@app.command()
def build(
    tag: Annotated[str, typer.Option(help="Docker tag to create.")] = "oneoxygen-sandbox:phase1",
) -> None:
    """Build and smoke-test the minimal Python sandbox image."""
    context = Path(__file__).resolve().parents[2] / "docker"
    try:
        image_id = DockerSDKAdapter().build_image(context, tag)
    except SandboxError as exc:
        typer.secho(f"Build failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    typer.secho(f"Built {tag} ({image_id}) and passed smoke test", fg=typer.colors.GREEN)


@tools_app.command("list")
def list_tools(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit provider-independent JSON tool definitions."),
    ] = False,
) -> None:
    """List available sandbox tools."""
    registry = default_tool_registry()
    if json_output:
        typer.echo(json.dumps(registry.provider_schemas(), indent=2, sort_keys=True))
        return
    for definition in registry.definitions():
        typer.echo(f"{definition.name}: {definition.description}")


@app.command("run")
def run_task(
    task_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    runs_directory: Annotated[
        Path, typer.Option(help="Directory for run records and approved artifacts.")
    ] = Path("runs"),
) -> None:
    """Execute all commands from a task YAML in one fresh sandbox."""
    session: SandboxSession | None = None
    try:
        task_path = task_file.resolve()
        task = load_task(task_path)
        session = SandboxSession(task, task_path.parent, runs_directory.resolve())
        typer.echo(
            f"Run {session.run_id} | task {task.sandbox.task_id}@{task.sandbox.task_version}"
        )
        with session:
            for index, command in enumerate(task.commands, start=1):
                result = session.execute(command)
                label = "timeout" if result.timed_out else f"exit {result.exit_code}"
                typer.echo(f"  [{index}/{len(task.commands)}] {label} | {command}")
                _print_captured_output(result.stdout)
                _print_captured_output(result.stderr, error=True)
                if result.exit_code != 0:
                    break
            artifacts = session.collect_artifacts()
            typer.echo(f"  collected {len(artifacts)} artifact(s)")
    except SandboxError as exc:
        typer.secho(f"Sandbox failed: {exc}", fg=typer.colors.RED, err=True)
        if session is not None:
            typer.echo(f"Run record: {session.record_path}", err=True)
        raise typer.Exit(1) from exc

    if session is None:
        raise typer.Exit(1)
    typer.echo(f"Run record: {session.record_path}")
    if session.record.final_status is not RunStatus.SUCCEEDED:
        raise typer.Exit(1)
    typer.secho("Run succeeded", fg=typer.colors.GREEN)


@app.command("tool-demo")
def tool_demo(
    task_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    calls_file: Annotated[
        Path | None,
        typer.Option("--calls", help="Scripted ToolCall YAML file."),
    ] = None,
    runs_directory: Annotated[
        Path, typer.Option("--runs-dir", help="Directory for run records and approved artifacts.")
    ] = Path("runs"),
) -> None:
    """Run a scripted demonstration through the Phase 2 tool dispatcher."""
    session: SandboxSession | None = None
    failed = False
    try:
        task_path = task_file.resolve()
        task = load_task(task_path)
        calls_path = (
            calls_file.resolve() if calls_file else task_path.parent / "scripted_calls.yaml"
        )
        calls = _load_scripted_calls(calls_path)
        session = SandboxSession(task, task_path.parent, runs_directory.resolve())
        typer.echo(
            f"Tool demo {session.run_id} | task {task.sandbox.task_id}@{task.sandbox.task_version}"
        )
        with session:
            dispatcher = ToolDispatcher(session)
            for index, call in enumerate(calls, start=1):
                result = dispatcher.dispatch(call)
                status = "ok" if result.success else f"error {result.error.code.value}"
                typer.echo(f"  [{index}/{len(calls)}] {status} | {call.tool_name}")
                if not result.success:
                    failed = True
                    if result.error is not None:
                        typer.echo(f"    {result.error.message}", err=True)
                    break
            if not dispatcher.submission_state.submitted:
                failed = True
                typer.echo("  submission missing", err=True)
            artifacts = session.collect_artifacts()
            typer.echo(f"  collected {len(artifacts)} artifact(s)")
    except SandboxError as exc:
        typer.secho(f"Tool demo failed: {exc}", fg=typer.colors.RED, err=True)
        if session is not None:
            typer.echo(f"Run record: {session.record_path}", err=True)
        raise typer.Exit(1) from exc

    if session is None:
        raise typer.Exit(1)
    typer.echo(f"Run record: {session.record_path}")
    if failed or session.record.final_status is not RunStatus.SUCCEEDED:
        raise typer.Exit(1)
    typer.secho("Tool demo succeeded", fg=typer.colors.GREEN)


def _load_scripted_calls(path: Path) -> tuple[ToolCall, ...]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read scripted tool calls: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid scripted tool-call YAML: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("calls")
    if not isinstance(data, list):
        raise ConfigurationError("scripted tool calls must be a YAML list or a mapping with calls")
    calls: list[ToolCall] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ConfigurationError("each scripted tool call must be a mapping")
        call_data = {
            "call_id": item.get("call_id", f"call-{index}"),
            "tool_name": item.get("tool_name"),
            "arguments": item.get("arguments", {}),
        }
        try:
            calls.append(ToolCall.model_validate(call_data))
        except ValueError as exc:
            raise ConfigurationError(f"invalid scripted tool call {index}: {exc}") from exc
    return tuple(calls)
