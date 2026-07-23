"""Command-line interface for the One Oxygen sandbox."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from oneoxygen_sandbox.agent import AgentRunner
from oneoxygen_sandbox.batching import (
    BatchArtifactStore,
    BatchItemResult,
    MockBatchBackend,
    OpenAIBatchBackend,
    SQLiteBatchStore,
    group_compatible_requests,
)
from oneoxygen_sandbox.config import load_task
from oneoxygen_sandbox.coordinator import DurableAgentCoordinator
from oneoxygen_sandbox.docker_adapter import DockerSDKAdapter
from oneoxygen_sandbox.errors import (
    ConfigurationError,
    LifecycleError,
    ModelError,
    SandboxError,
)
from oneoxygen_sandbox.model_adapters import default_model_adapter_registry
from oneoxygen_sandbox.models import (
    DataClassification,
    InferenceTransport,
    ModelErrorCode,
    ModelProvider,
    ModelRunConfig,
    RunStatus,
    ToolCall,
    ToolSchemaMode,
)
from oneoxygen_sandbox.orchestration import AgentRunStatus, SQLiteAgentStateStore
from oneoxygen_sandbox.session import SandboxSession
from oneoxygen_sandbox.tools import ToolDispatcher, default_tool_registry

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Secure local Docker sandbox runner for One Oxygen.",
)
tools_app = typer.Typer(help="Inspect provider-independent tool definitions.")
models_app = typer.Typer(help="Inspect and diagnose model adapters without network access.")
eval_app = typer.Typer(help="Enqueue and resume durable agent evaluations.")
batch_app = typer.Typer(help="Build, submit, inspect, collect, and cancel batches.")
app.add_typer(tools_app, name="tools")
app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")
app.add_typer(batch_app, name="batch")

_EXIT_INVALID_TASK = 2
_EXIT_UNAVAILABLE_PROVIDER = 3
_EXIT_MISSING_DEPENDENCY = 4
_EXIT_MISSING_API_KEY = 5
_EXIT_PROVIDER_ERROR = 6
_EXIT_INCOMPLETE = 7
_EXIT_LIMIT_EXCEEDED = 8
_EXIT_SANDBOX_ERROR = 9
_EXIT_INTERNAL_ERROR = 10


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


@models_app.command("list")
def list_models(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable adapter availability."),
    ] = False,
) -> None:
    """List registered adapters without importing SDKs or contacting providers."""
    descriptions = default_model_adapter_registry().descriptions()
    rows = [
        {
            "provider": item.provider.value,
            "optional_dependency": item.optional_dependency,
            "dependency_available": item.dependency_available,
        }
        for item in descriptions
    ]
    if json_output:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    for item in descriptions:
        availability = "available" if item.dependency_available else "missing dependency"
        typer.echo(f"{item.provider.value}: {availability}")


@models_app.command("doctor")
def model_doctor(
    provider: Annotated[
        ModelProvider,
        typer.Option("--provider", help="Provider to check without making an API request."),
    ],
) -> None:
    """Check local dependency and credential configuration without a paid request."""
    registry = default_model_adapter_registry()
    info = next((item for item in registry.descriptions() if item.provider is provider), None)
    if info is None:
        typer.secho("Model provider is not configured", fg=typer.colors.RED, err=True)
        raise typer.Exit(_EXIT_UNAVAILABLE_PROVIDER)
    if not info.dependency_available:
        typer.secho(
            f"{provider.value} dependency is not installed",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(_EXIT_MISSING_DEPENDENCY)
    key_name = (
        "OPENAI_API_KEY"
        if provider is ModelProvider.OPENAI
        else "AIRFORCE_API_KEY"
        if provider is ModelProvider.AIRFORCE
        else None
    )
    if key_name is not None and not os.environ.get(key_name, "").strip():
        typer.secho(f"{key_name} is not configured", fg=typer.colors.RED, err=True)
        raise typer.Exit(_EXIT_MISSING_API_KEY)
    typer.secho(
        f"{provider.value} is locally configured; no provider request was made",
        fg=typer.colors.GREEN,
    )


def _durable_components(
    state_directory: Path,
) -> tuple[
    SQLiteAgentStateStore,
    BatchArtifactStore,
    SQLiteBatchStore,
    DurableAgentCoordinator,
]:
    root = state_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    states = SQLiteAgentStateStore(str(root / "agent-state.sqlite3"))
    artifacts = BatchArtifactStore(root / "batch-files")
    batches = SQLiteBatchStore(root / "batch-state.sqlite3")
    coordinator = DurableAgentCoordinator(states, artifacts, root / "runs")
    return states, artifacts, batches, coordinator


def _batch_backend(
    provider: ModelProvider,
    artifacts: BatchArtifactStore,
    batches: SQLiteBatchStore,
) -> MockBatchBackend | OpenAIBatchBackend:
    if provider is ModelProvider.OPENAI:
        return OpenAIBatchBackend(artifacts, batches)
    if provider is ModelProvider.SCRIPTED:
        return MockBatchBackend(artifacts, batches)
    raise ConfigurationError("Api.Airforce has no documented Batch endpoint")


@models_app.command("discover")
def discover_models(
    provider: Annotated[
        ModelProvider,
        typer.Option("--provider", help="Model catalog provider."),
    ],
    free: Annotated[
        bool,
        typer.Option("--free", help="Require explicit zero input and output prices."),
    ] = False,
    tools: Annotated[
        bool,
        typer.Option("--tools", help="Require reported tool/function calling support."),
    ] = False,
    operational: Annotated[
        bool,
        typer.Option("--operational", help="Require operational live status."),
    ] = False,
    allow_third_party_gateway: Annotated[
        bool,
        typer.Option(
            "--allow-third-party-gateway",
            help="Acknowledge that Api.Airforce is an unverified third-party route.",
        ),
    ] = False,
) -> None:
    """Discover the live Api.Airforce catalog without hard-coded model IDs."""
    if provider is not ModelProvider.AIRFORCE:
        raise typer.BadParameter("live discovery is implemented only for airforce")
    config = ModelRunConfig(
        provider=provider,
        model="catalog-discovery",
        transport=InferenceTransport.GATEWAY_DIRECT,
    )
    try:
        adapter = default_model_adapter_registry().create(
            provider,
            config,
            data_classification=DataClassification.SYNTHETIC,
            allow_third_party_gateway=allow_third_party_gateway,
        )
        rows = list(adapter.discover_models())  # type: ignore[attr-defined]
        adapter.close()
    except ModelError as exc:
        typer.secho(exc.message, fg=typer.colors.RED, err=True)
        raise typer.Exit(_EXIT_PROVIDER_ERROR) from exc
    if free:
        rows = [
            row
            for row in rows
            if row["input_price_cents_per_million"] == 0
            and row["output_price_cents_per_million"] == 0
        ]
    if tools:
        rows = [row for row in rows if row["supports_tools"] is True]
    if operational:
        rows = [row for row in rows if row["status"] == "operational"]
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


@eval_app.command("enqueue")
def enqueue_agent_runs(
    task_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    provider: Annotated[ModelProvider, typer.Option("--provider")],
    model: Annotated[str, typer.Option("--model")],
    transport: Annotated[
        InferenceTransport, typer.Option("--transport")
    ] = InferenceTransport.PROVIDER_BATCH,
    count: Annotated[int, typer.Option("--count", min=1, max=10_000)] = 1,
    state_directory: Annotated[
        Path, typer.Option("--state-dir", help="Durable local coordinator directory.")
    ] = Path(".oneoxygen"),
) -> None:
    """Create durable runs and stop with each first turn ready for batching."""
    if transport is not InferenceTransport.PROVIDER_BATCH:
        raise typer.BadParameter("enqueue currently requires provider_batch transport")
    task_path = task_file.resolve()
    try:
        task = load_task(task_path)
        _states, _artifacts, _batches, coordinator = _durable_components(state_directory)
        config = ModelRunConfig(
            provider=provider,
            model=model,
            transport=transport,
        )
        created = [
            coordinator.enqueue(task, task_path.parent, config).run_id for _index in range(count)
        ]
    except (ConfigurationError, ValidationError, LifecycleError) as exc:
        typer.secho("Could not enqueue durable runs", fg=typer.colors.RED, err=True)
        raise typer.Exit(_EXIT_INVALID_TASK) from exc
    typer.echo(json.dumps({"enqueued": created}, indent=2, sort_keys=True))


@batch_app.command("ready")
def list_ready_turns(
    provider: Annotated[ModelProvider | None, typer.Option("--provider")] = None,
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """List ready model turns without modifying them."""
    states, _artifacts, _batches, _coordinator = _durable_components(state_directory)
    ready = [
        {
            "run_id": state.run_id,
            "turn": state.current_turn,
            "provider": state.model_configuration.provider.value,
            "model": state.model_configuration.model,
        }
        for state in states.list(AgentRunStatus.READY_FOR_MODEL)
        if provider is None or state.model_configuration.provider is provider
    ]
    typer.echo(json.dumps(ready, indent=2, sort_keys=True))


@batch_app.command("build")
def build_local_batches(
    provider: Annotated[ModelProvider, typer.Option("--provider")],
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """Compile compatible ready turns into local JSONL without network access."""
    states, artifacts, batches, coordinator = _durable_components(state_directory)
    ready_states = [
        state
        for state in states.list(AgentRunStatus.READY_FOR_MODEL)
        if state.model_configuration.provider is provider
    ]
    requests = [
        coordinator.materialize_batch_request(coordinator.ready_turn(state.run_id))
        for state in ready_states
    ]
    backend = _batch_backend(provider, artifacts, batches)
    jobs = [backend.build_batch(group) for group in group_compatible_requests(requests)]
    typer.echo(
        json.dumps(
            [job.model_dump(mode="json") for job in jobs],
            indent=2,
            sort_keys=True,
        )
    )


@batch_app.command("validate")
def validate_local_batch(
    local_batch_id: Annotated[str, typer.Argument()],
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """Revalidate every request and the one-model-per-file rule."""
    _states, artifacts, batches, _coordinator = _durable_components(state_directory)
    job = batches.load_job(local_batch_id)
    backend = _batch_backend(job.provider, artifacts, batches)
    requests = tuple(batches.load_request(request_id) for request_id in job.request_ids)
    backend.validate_requests(requests)
    typer.secho("Batch JSONL is valid", fg=typer.colors.GREEN)


@batch_app.command("submit")
def submit_local_batch(
    local_batch_id: Annotated[str, typer.Argument()],
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """Upload and submit one already-built batch."""
    _states, artifacts, batches, coordinator = _durable_components(state_directory)
    job = batches.load_job(local_batch_id)
    backend = _batch_backend(job.provider, artifacts, batches)
    submitted = backend.submit_batch(job)
    coordinator.mark_batch_submitted(submitted)
    typer.echo(json.dumps(submitted.model_dump(mode="json"), indent=2, sort_keys=True))


@batch_app.command("status")
def batch_status(
    local_batch_id: Annotated[str, typer.Argument()],
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """Poll a provider exactly once."""
    _states, artifacts, batches, _coordinator = _durable_components(state_directory)
    job = batches.load_job(local_batch_id)
    backend = _batch_backend(job.provider, artifacts, batches)
    updated = backend.get_status(job)
    typer.echo(json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True))


@batch_app.command("collect")
def collect_batch(
    local_batch_id: Annotated[str, typer.Argument()],
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """Download and correlate provider output/error files without executing tools."""
    _states, artifacts, batches, _coordinator = _durable_components(state_directory)
    job = batches.load_job(local_batch_id)
    backend = _batch_backend(job.provider, artifacts, batches)
    results = backend.retrieve_results(job)
    artifacts.write_json(
        "normalized-results",
        local_batch_id,
        [result.model_dump(mode="json") for result in results],
    )
    typer.echo(json.dumps({"collected": len(results)}, sort_keys=True))


@eval_app.command("resume-ready")
def resume_ready_runs(
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """Apply collected results, run tools in fresh sandboxes, and checkpoint again."""
    _states, artifacts, batches, coordinator = _durable_components(state_directory)
    resumed: list[str] = []
    for job in batches.list_jobs():
        reference = f"normalized-results/{job.internal_batch_id}.json"
        if not artifacts.resolve(reference).is_file():
            continue
        backend = _batch_backend(job.provider, artifacts, batches)
        raw_results = artifacts.read_json(reference)
        for raw in raw_results:
            result = BatchItemResult.model_validate(raw)
            state = coordinator.apply_result(result, backend)
            resumed.append(state.run_id)
    typer.echo(json.dumps({"resumed": sorted(set(resumed))}, sort_keys=True))


@batch_app.command("cancel")
def cancel_batch(
    local_batch_id: Annotated[str, typer.Argument()],
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """Request cancellation without waiting indefinitely."""
    _states, artifacts, batches, _coordinator = _durable_components(state_directory)
    job = batches.load_job(local_batch_id)
    backend = _batch_backend(job.provider, artifacts, batches)
    cancelled = backend.cancel_batch(job)
    typer.echo(json.dumps(cancelled.model_dump(mode="json"), indent=2, sort_keys=True))


@batch_app.command("list")
def list_batches(
    unfinished: Annotated[bool, typer.Option("--unfinished")] = False,
    state_directory: Annotated[Path, typer.Option("--state-dir")] = Path(".oneoxygen"),
) -> None:
    """List durable local batch jobs."""
    _states, _artifacts, batches, _coordinator = _durable_components(state_directory)
    typer.echo(
        json.dumps(
            [job.model_dump(mode="json") for job in batches.list_jobs(unfinished=unfinished)],
            indent=2,
            sort_keys=True,
        )
    )


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


@app.command("agent-run")
def agent_run(
    task_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    provider: Annotated[
        ModelProvider,
        typer.Option("--provider", help="Model adapter provider."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Required provider model identifier."),
    ],
    script: Annotated[
        Path | None,
        typer.Option("--script", help="Scripted-provider YAML or JSON turn script."),
    ] = None,
    runs_directory: Annotated[
        Path,
        typer.Option("--runs-dir", help="Directory for run records and submitted artifacts."),
    ] = Path("runs"),
    maximum_output_tokens: Annotated[
        int,
        typer.Option("--max-output-tokens", help="Maximum provider output tokens per turn."),
    ] = 4_096,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="Optional provider sampling temperature."),
    ] = None,
    model_call_timeout_seconds: Annotated[
        float,
        typer.Option("--model-call-timeout", help="Timeout for each provider attempt."),
    ] = 60.0,
    maximum_retry_attempts: Annotated[
        int,
        typer.Option("--max-retries", help="Maximum central retries after an initial failure."),
    ] = 2,
    initial_retry_delay_seconds: Annotated[
        float,
        typer.Option("--initial-retry-delay", help="Initial retry backoff delay."),
    ] = 1.0,
    tool_schema_mode: Annotated[
        ToolSchemaMode,
        typer.Option("--tool-schema-mode", help="Portable or explicitly native strict schemas."),
    ] = ToolSchemaMode.PORTABLE,
    store_provider_response: Annotated[
        bool,
        typer.Option(
            "--store-provider-response",
            help="Request provider-side response storage when the adapter supports it.",
        ),
    ] = False,
    transport: Annotated[
        InferenceTransport,
        typer.Option("--transport", help="Direct official or acknowledged gateway route."),
    ] = InferenceTransport.DIRECT,
    allow_third_party_gateway: Annotated[
        bool,
        typer.Option(
            "--allow-third-party-gateway",
            help="Acknowledge the experimental, unverified Api.Airforce route.",
        ),
    ] = False,
) -> None:
    """Execute a task through the direct provider-neutral agent loop."""
    runner: AgentRunner | None = None
    try:
        task_path = task_file.resolve()
        task = load_task(task_path)
        if task.agent is None:
            raise ConfigurationError("task does not define an agent section")
        effective_transport = (
            InferenceTransport.GATEWAY_DIRECT if provider is ModelProvider.AIRFORCE else transport
        )
        if effective_transport is InferenceTransport.PROVIDER_BATCH:
            raise ConfigurationError("use 'eval enqueue' for provider_batch execution")
        if provider is ModelProvider.OPENAI and not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ModelError(
                ModelErrorCode.MISSING_API_KEY,
                "OPENAI_API_KEY is required for the OpenAI provider",
            )
        if (
            provider is ModelProvider.AIRFORCE
            and not os.environ.get("AIRFORCE_API_KEY", "").strip()
        ):
            raise ModelError(
                ModelErrorCode.MISSING_API_KEY,
                "AIRFORCE_API_KEY is required for the Api.Airforce gateway",
            )
        config = ModelRunConfig(
            provider=provider,
            model=model,
            maximum_output_tokens=maximum_output_tokens,
            temperature=temperature,
            model_call_timeout_seconds=model_call_timeout_seconds,
            maximum_retry_attempts=maximum_retry_attempts,
            initial_retry_delay_seconds=initial_retry_delay_seconds,
            tool_schema_mode=tool_schema_mode,
            store_provider_response=store_provider_response,
            transport=effective_transport,
        )
        registry = default_model_adapter_registry()
        adapter_arguments: dict[str, object] = {}
        if provider is ModelProvider.SCRIPTED:
            adapter_arguments["script_path"] = (
                script.resolve() if script is not None else task_path.parent / "model_script.yaml"
            )
        elif script is not None:
            raise ConfigurationError("--script is only valid with the scripted provider")
        if provider is ModelProvider.AIRFORCE:
            adapter_arguments.update(
                {
                    "data_classification": task.agent.data_classification,
                    "allow_third_party_gateway": allow_third_party_gateway,
                }
            )
        adapter = registry.create(provider, config, **adapter_arguments)
        runner = AgentRunner(
            task,
            task_path.parent,
            config,
            adapter,
            runs_directory=runs_directory.resolve(),
        )
        typer.echo(
            f"Agent run | task {task.sandbox.task_id}@{task.sandbox.task_version} "
            f"| provider {provider.value} | model {model}"
        )
        record = runner.run()
    except (ConfigurationError, ValidationError) as exc:
        typer.secho("Invalid agent task or configuration", fg=typer.colors.RED, err=True)
        raise typer.Exit(_EXIT_INVALID_TASK) from exc
    except ModelError as exc:
        exit_code = (
            _EXIT_MISSING_API_KEY
            if exc.model_code is ModelErrorCode.MISSING_API_KEY
            else _EXIT_MISSING_DEPENDENCY
            if exc.model_code is ModelErrorCode.MISSING_DEPENDENCY
            else _EXIT_UNAVAILABLE_PROVIDER
            if exc.model_code is ModelErrorCode.PROVIDER_NOT_CONFIGURED
            else _EXIT_PROVIDER_ERROR
        )
        typer.secho(f"Model adapter failed: {exc.message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(exit_code) from exc
    except SandboxError as exc:
        typer.secho(
            "Sandbox failed before the agent run could start", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(_EXIT_SANDBOX_ERROR) from exc

    if runner.record_path is not None:
        typer.echo(f"Run record: {runner.record_path}")
    termination = record.termination_reason.value if record.termination_reason else "unknown"
    typer.echo(f"Termination: {termination}")
    if record.final_status is RunStatus.SUCCEEDED:
        typer.secho("Agent run succeeded", fg=typer.colors.GREEN)
        return
    typer.secho(f"Agent run ended with status {record.final_status.value}", fg=typer.colors.RED)
    raise typer.Exit(_agent_exit_code(record.final_status))


def _agent_exit_code(status: RunStatus) -> int:
    if status is RunStatus.INCOMPLETE:
        return _EXIT_INCOMPLETE
    if status is RunStatus.LIMIT_EXCEEDED:
        return _EXIT_LIMIT_EXCEEDED
    if status is RunStatus.PROVIDER_ERROR or status is RunStatus.REFUSED:
        return _EXIT_PROVIDER_ERROR
    if status is RunStatus.SANDBOX_ERROR:
        return _EXIT_SANDBOX_ERROR
    if status is RunStatus.CANCELLED:
        return 130
    return _EXIT_INTERNAL_ERROR


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
