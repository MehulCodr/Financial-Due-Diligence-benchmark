from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from oneoxygen_sandbox.agent import AgentRunner
from oneoxygen_sandbox.errors import DockerOperationError, ModelError
from oneoxygen_sandbox.model_adapters.scripted import ScriptedModelAdapter
from oneoxygen_sandbox.models import (
    AgentTaskSpec,
    AgentTerminationReason,
    BrowserConfig,
    BrowserSourceProfile,
    ExecResult,
    InputAsset,
    ModelCapabilities,
    ModelErrorCode,
    ModelProvider,
    ModelRunConfig,
    ModelTurnRequest,
    ModelTurnResponse,
    ModelUsage,
    NormalizedFinishReason,
    RunRecord,
    RunStatus,
    SandboxSpec,
    SandboxTask,
    ToolErrorCode,
    ToolPolicy,
)


class FakeContainer:
    id = "fake-agent-container"


class FakeDockerAdapter:
    sandbox_user = "10001:10001"

    def __init__(self) -> None:
        self.container = FakeContainer()
        self.workspace: Path | None = None
        self.environments: list[dict[str, str]] = []
        self.started = 0
        self.stopped = 0
        self.removed = 0
        self.executed_commands: list[str] = []

    def check_available(self) -> dict[str, Any]:
        return {"OSType": "linux"}

    def resolve_image(self, image: str) -> str:
        return "sha256:agent-runner-test"

    def create_container(
        self,
        spec: SandboxSpec,
        workspace: Path,
        environment: dict[str, str],
        run_id: str,
    ) -> FakeContainer:
        self.workspace = workspace
        self.environments.append(dict(environment))
        return self.container

    def start_container(self, container: FakeContainer) -> None:
        self.started += 1

    def execute(
        self,
        container: FakeContainer,
        command: str,
        working_directory: str,
        timeout_seconds: float,
        maximum_output_bytes: int,
    ) -> ExecResult:
        self.executed_commands.append(command)
        now = datetime.now(UTC)
        return ExecResult(
            command=command,
            stdout="",
            stderr="",
            exit_code=0,
            start_timestamp=now,
            end_timestamp=now,
            duration_seconds=0,
        )

    def stop_container(self, container: FakeContainer) -> None:
        self.stopped += 1

    def remove_container(self, container: FakeContainer) -> None:
        self.removed += 1


class SequenceAdapter:
    """Small adapter for runner-only failure, redaction, and clock tests."""

    provider = ModelProvider.SCRIPTED
    capabilities = ModelCapabilities(tool_calling=True, multiple_tool_calls_per_turn=True)

    def __init__(
        self,
        config: ModelRunConfig,
        outcomes: list[ModelTurnResponse | ModelError],
    ) -> None:
        self.config = config
        self.outcomes = list(outcomes)
        self.requests: list[ModelTurnRequest] = []
        self.started = False
        self.closed = False

    def validate_config(self, config: ModelRunConfig) -> ModelRunConfig:
        return config

    def start_conversation(self, request: ModelTurnRequest) -> None:
        self.started = True

    def generate_next_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("test adapter ran out of outcomes")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, ModelError):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


class JumpClock:
    def __init__(self, first: float, later: float) -> None:
        self.first = first
        self.later = later
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.first if self.calls == 1 else self.later


class MutableClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class AdvancingAdapter(SequenceAdapter):
    def __init__(
        self,
        config: ModelRunConfig,
        outcomes: list[ModelTurnResponse | ModelError],
        clock: MutableClock,
    ) -> None:
        super().__init__(config, outcomes)
        self.clock = clock

    def generate_next_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        self.clock.value = 2.0
        return super().generate_next_turn(request)


def model_config(**updates: object) -> ModelRunConfig:
    values: dict[str, object] = {
        "provider": ModelProvider.SCRIPTED,
        "model": "scripted-runner-test",
        "maximum_retry_attempts": 0,
        "initial_retry_delay_seconds": 0,
    }
    values.update(updates)
    return ModelRunConfig.model_validate(values)


def make_task(
    root: Path,
    *,
    agent_updates: dict[str, object] | None = None,
    policy: ToolPolicy | None = None,
    browser: BrowserConfig | None = None,
) -> tuple[SandboxTask, Path]:
    task_directory = root / "task"
    task_directory.mkdir(parents=True)
    (task_directory / "task.md").write_text(
        "Inspect the supplied evidence and submit a supported finding.\n",
        encoding="utf-8",
    )
    assets = task_directory / "assets"
    assets.mkdir()
    (assets / "seed.txt").write_text("evidence\n", encoding="utf-8")
    agent_values: dict[str, object] = {
        "instruction_file": "task.md",
        "maximum_model_turns": 8,
        "maximum_provider_requests": 20,
        "overall_wall_time_seconds": 60,
    }
    agent_values.update(agent_updates or {})
    task = SandboxTask(
        sandbox=SandboxSpec(
            image="image:agent-test",
            task_id="agent-runner-test",
            task_version="1",
            input_assets=(InputAsset(source="assets/seed.txt", destination="input/seed.txt"),),
            overall_timeout_seconds=60,
        ),
        tool_policy=policy or ToolPolicy(),
        agent=AgentTaskSpec.model_validate(agent_values),
        browser=browser,
    )
    return task, task_directory


def scripted_run(
    root: Path,
    turns: list[dict[str, Any]],
    *,
    task: SandboxTask | None = None,
    task_directory: Path | None = None,
    config: ModelRunConfig | None = None,
    docker: FakeDockerAdapter | None = None,
    sleeper: Callable[[float], None] = lambda _delay: None,
    jitter: Callable[[float], float] = lambda _delay: 0,
    clock: Callable[[], float] | None = None,
) -> tuple[RunRecord, AgentRunner, FakeDockerAdapter]:
    if task is None or task_directory is None:
        task, task_directory = make_task(root)
    run_config = config or model_config()
    adapter = ScriptedModelAdapter(run_config, {"schema_version": 1, "turns": turns})
    fake_docker = docker or FakeDockerAdapter()
    runner_kwargs: dict[str, Any] = {}
    if clock is not None:
        runner_kwargs["clock"] = clock
    runner = AgentRunner(
        task,
        task_directory,
        run_config,
        adapter,
        runs_directory=root / "runs",
        sandbox_adapter=fake_docker,
        sleeper=sleeper,
        jitter=jitter,
        **runner_kwargs,
    )
    return runner.run(), runner, fake_docker


def persisted_record(runner: AgentRunner) -> RunRecord:
    assert runner.record_path is not None
    return RunRecord.model_validate_json(runner.record_path.read_text(encoding="utf-8"))


def test_successful_multiturn_run_records_tools_and_collects_only_submission(
    tmp_path: Path,
) -> None:
    turns = [
        {
            "tool_calls": [
                {"call_id": "call-list", "tool_name": "list_files", "arguments": {}},
                {
                    "call_id": "call-read",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "input/seed.txt"},
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        },
        {
            "expected_previous_tool_result_call_ids": ["call-list", "call-read"],
            "tool_calls": [
                {
                    "call_id": "call-write-main",
                    "tool_name": "write_text_file",
                    "arguments": {
                        "path": "output/findings.md",
                        "content": "# Finding\n\nEvidence inspected.\n",
                        "create_parents": True,
                    },
                },
                {
                    "call_id": "call-write-extra",
                    "tool_name": "write_text_file",
                    "arguments": {
                        "path": "output/not-submitted.txt",
                        "content": "must not be retained\n",
                    },
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
        },
        {
            "expected_previous_tool_result_call_ids": [
                "call-write-main",
                "call-write-extra",
            ],
            "tool_calls": [
                {
                    "call_id": "call-submit",
                    "tool_name": "submit_result",
                    "arguments": {
                        "summary": "Evidence-based finding completed.",
                        "artifact_paths": ["output/findings.md"],
                    },
                }
            ],
            "usage": {"input_tokens": 30, "output_tokens": 6, "total_tokens": 36},
        },
    ]

    record, runner, docker = scripted_run(tmp_path, turns)
    restored = persisted_record(runner)

    assert record.final_status is RunStatus.SUCCEEDED
    assert record.termination_reason is AgentTerminationReason.SUCCESSFUL_SUBMISSION
    assert [event.tool_name for event in record.tool_events] == [
        "list_files",
        "read_text_file",
        "write_text_file",
        "write_text_file",
        "submit_result",
    ]
    assert record.submission is not None
    assert [artifact.relative_path for artifact in record.artifacts] == ["findings.md"]
    assert (runner.session.run_directory / "artifacts" / "findings.md").is_file()
    assert not (runner.session.run_directory / "artifacts" / "not-submitted.txt").exists()
    assert record.metrics.model_turns == 3
    assert record.metrics.provider_attempts == 3
    assert record.metrics.successful_tool_calls == 5
    assert record.metrics.total_input_tokens == 60
    assert record.metrics.total_output_tokens == 18
    assert record.metrics.total_tokens == 78
    assert restored == record
    assert docker.started == docker.stopped == docker.removed == 1
    assert docker.workspace is not None and not docker.workspace.exists()


def test_multiple_calls_keep_order_and_reject_calls_after_submission(tmp_path: Path) -> None:
    turns = [
        {
            "tool_calls": [
                {
                    "call_id": "call-write",
                    "tool_name": "write_text_file",
                    "arguments": {
                        "path": "output/findings.md",
                        "content": "done\n",
                        "create_parents": True,
                    },
                },
                {
                    "call_id": "call-submit",
                    "tool_name": "submit_result",
                    "arguments": {
                        "summary": "done",
                        "artifact_paths": ["output/findings.md"],
                    },
                },
                {
                    "call_id": "call-after",
                    "tool_name": "write_text_file",
                    "arguments": {"path": "output/after.txt", "content": "late"},
                },
            ]
        }
    ]

    record, runner, _docker = scripted_run(tmp_path, turns)

    assert record.final_status is RunStatus.SUCCEEDED
    assert [event.call_id for event in record.tool_events] == [
        "call-write",
        "call-submit",
        "call-after",
    ]
    assert [call.original_index for call in record.model_events[0].tool_calls] == [0, 1, 2]
    assert record.tool_events[-1].error_code is ToolErrorCode.ALREADY_SUBMITTED
    assert record.metrics.successful_tool_calls == 2
    assert record.metrics.failed_tool_calls == 1
    assert not (runner.session.run_directory / "artifacts" / "after.txt").exists()


def test_duplicate_tool_call_ids_stop_before_dispatch(tmp_path: Path) -> None:
    turns = [
        {
            "tool_calls": [
                {"call_id": "duplicate", "tool_name": "list_files"},
                {"call_id": "duplicate", "tool_name": "list_files"},
            ]
        }
    ]

    record, runner, _docker = scripted_run(tmp_path, turns)

    assert record.final_status is RunStatus.PROVIDER_ERROR
    assert record.error is not None
    assert record.error.code == ModelErrorCode.DUPLICATE_TOOL_CALL_ID.value
    assert record.tool_events == []
    assert len(record.model_events) == 1
    assert persisted_record(runner).error == record.error


def test_tool_call_ids_must_be_unique_across_turns(tmp_path: Path) -> None:
    turns = [
        {"tool_calls": [{"call_id": "reused", "tool_name": "list_files"}]},
        {
            "expected_previous_call_ids": ["reused"],
            "tool_calls": [{"call_id": "reused", "tool_name": "list_files"}],
        },
    ]

    record, _runner, _docker = scripted_run(tmp_path, turns)

    assert record.final_status is RunStatus.PROVIDER_ERROR
    assert record.error is not None
    assert record.error.code == ModelErrorCode.DUPLICATE_TOOL_CALL_ID.value
    assert len(record.tool_events) == 1
    assert len(record.model_events) == 2


def test_invalid_tool_arguments_are_returned_and_a_later_turn_can_submit(
    tmp_path: Path,
) -> None:
    turns = [
        {
            "tool_calls": [
                {
                    "call_id": "call-invalid",
                    "tool_name": "read_text_file",
                    "arguments": {},
                }
            ]
        },
        {
            "expected_previous_tool_results": [
                {
                    "call_id": "call-invalid",
                    "success": False,
                    "error_code": "invalid_arguments",
                }
            ],
            "tool_calls": [
                {
                    "call_id": "call-write",
                    "tool_name": "write_text_file",
                    "arguments": {
                        "path": "output/findings.md",
                        "content": "recovered\n",
                        "create_parents": True,
                    },
                },
                {
                    "call_id": "call-submit",
                    "tool_name": "submit_result",
                    "arguments": {
                        "summary": "recovered",
                        "artifact_paths": ["output/findings.md"],
                    },
                },
            ],
        },
    ]

    record, _runner, _docker = scripted_run(tmp_path, turns)

    assert record.final_status is RunStatus.SUCCEEDED
    assert record.tool_events[0].error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert record.metrics.failed_tool_calls == 1
    assert record.metrics.successful_tool_calls == 2


def test_final_text_without_required_submission_is_incomplete(tmp_path: Path) -> None:
    record, runner, _docker = scripted_run(
        tmp_path,
        [{"text": "I am done.", "finish_reason": "completed"}],
    )

    assert record.final_status is RunStatus.INCOMPLETE
    assert record.termination_reason is AgentTerminationReason.FINAL_TEXT_WITHOUT_SUBMISSION
    assert record.submission is None
    assert record.artifacts == []
    assert persisted_record(runner).final_status is RunStatus.INCOMPLETE


def test_model_refusal_is_a_stable_terminal_state(tmp_path: Path) -> None:
    record, _runner, _docker = scripted_run(
        tmp_path,
        [{"text": "Cannot comply.", "finish_reason": "refused"}],
    )

    assert record.final_status is RunStatus.REFUSED
    assert record.termination_reason is AgentTerminationReason.MODEL_REFUSAL
    assert record.model_events[0].finish_reason is NormalizedFinishReason.REFUSED


def test_maximum_turns_is_enforced_after_the_last_allowed_turn(tmp_path: Path) -> None:
    task, task_directory = make_task(tmp_path, agent_updates={"maximum_model_turns": 1})
    record, _runner, _docker = scripted_run(
        tmp_path,
        [{"tool_calls": [{"call_id": "call-list", "tool_name": "list_files"}]}],
        task=task,
        task_directory=task_directory,
    )

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is AgentTerminationReason.MAXIMUM_TURNS_REACHED
    assert len(record.model_events) == 1
    assert len(record.tool_events) == 1


@pytest.mark.parametrize(
    ("agent_field", "usage", "expected_reason"),
    [
        (
            "maximum_total_input_tokens",
            {"input_tokens": 11, "output_tokens": 1, "total_tokens": 12},
            AgentTerminationReason.INPUT_TOKEN_LIMIT_REACHED,
        ),
        (
            "maximum_total_output_tokens",
            {"input_tokens": 1, "output_tokens": 11, "total_tokens": 12},
            AgentTerminationReason.OUTPUT_TOKEN_LIMIT_REACHED,
        ),
        (
            "maximum_total_tokens",
            {"input_tokens": 6, "output_tokens": 5, "total_tokens": 11},
            AgentTerminationReason.TOTAL_TOKEN_LIMIT_REACHED,
        ),
    ],
)
def test_token_limits_stop_after_recording_the_response(
    tmp_path: Path,
    agent_field: str,
    usage: dict[str, int],
    expected_reason: AgentTerminationReason,
) -> None:
    task, task_directory = make_task(tmp_path, agent_updates={agent_field: 10})
    record, _runner, _docker = scripted_run(
        tmp_path,
        [{"text": "over limit", "usage": usage}],
        task=task,
        task_directory=task_directory,
    )

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is expected_reason
    assert len(record.model_events) == 1
    assert record.model_events[0].usage == ModelUsage.model_validate(usage)


def test_transient_failure_retries_then_records_one_successful_turn(tmp_path: Path) -> None:
    delays: list[float] = []
    config = model_config(maximum_retry_attempts=1, initial_retry_delay_seconds=0.25)
    turns = [
        {
            "error": {
                "code": "rate_limited",
                "message": "slow down",
                "retryable": True,
                "provider_metadata": {"request_id": "retry-1"},
            }
        },
        {
            "text": "finished without submission",
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    ]

    record, _runner, _docker = scripted_run(
        tmp_path,
        turns,
        config=config,
        sleeper=delays.append,
    )

    assert record.final_status is RunStatus.INCOMPLETE
    assert len(record.model_events) == 1
    event = record.model_events[0]
    assert event.attempt_count == 2
    assert [attempt.succeeded for attempt in event.attempts] == [False, True]
    assert event.attempts[0].error_code is ModelErrorCode.RATE_LIMITED
    assert event.attempts[0].retryable is True
    assert delays == [0.25]
    assert record.metrics.provider_attempts == 2


def test_transient_retry_exhaustion_records_all_attempts_and_persists(
    tmp_path: Path,
) -> None:
    delays: list[float] = []
    config = model_config(maximum_retry_attempts=2, initial_retry_delay_seconds=0.1)
    turns = [
        {
            "error": {
                "code": "provider_unavailable",
                "message": f"temporary failure {number}",
                "retryable": True,
            }
        }
        for number in range(3)
    ]

    record, runner, docker = scripted_run(
        tmp_path,
        turns,
        config=config,
        sleeper=delays.append,
    )
    restored = persisted_record(runner)

    assert record.final_status is RunStatus.PROVIDER_ERROR
    assert record.termination_reason is AgentTerminationReason.REPEATED_PROVIDER_FAILURE
    assert record.error is not None
    assert record.error.code == ModelErrorCode.PROVIDER_UNAVAILABLE.value
    assert len(record.model_events) == 1
    assert record.model_events[0].attempt_count == 3
    assert [attempt.retryable for attempt in record.model_events[0].attempts] == [
        True,
        True,
        False,
    ]
    assert delays == pytest.approx([0.1, 0.2])
    assert restored == record
    assert docker.removed == 1


@pytest.mark.parametrize(
    "code",
    [ModelErrorCode.AUTHENTICATION_FAILED, ModelErrorCode.INVALID_REQUEST],
)
def test_nonretryable_provider_errors_are_attempted_once(
    tmp_path: Path,
    code: ModelErrorCode,
) -> None:
    delays: list[float] = []
    config = model_config(maximum_retry_attempts=3, initial_retry_delay_seconds=1)
    turns = [{"error": {"code": code.value, "message": "permanent failure"}}]

    record, _runner, _docker = scripted_run(
        tmp_path,
        turns,
        config=config,
        sleeper=delays.append,
    )

    assert record.final_status is RunStatus.PROVIDER_ERROR
    assert record.model_events[0].attempt_count == 1
    assert record.model_events[0].attempts[0].error_code is code
    assert delays == []


def test_model_call_timeout_is_normalized_and_persisted(tmp_path: Path) -> None:
    record, runner, _docker = scripted_run(tmp_path, [{"simulate_timeout": True}])

    assert record.final_status is RunStatus.PROVIDER_ERROR
    assert record.error is not None
    assert record.error.code == ModelErrorCode.REQUEST_TIMEOUT.value
    assert record.model_events[0].attempts[0].error_code is ModelErrorCode.REQUEST_TIMEOUT
    assert persisted_record(runner).error == record.error


def test_prompt_instruction_and_tool_hashes_are_stable(tmp_path: Path) -> None:
    task, task_directory = make_task(tmp_path)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, _first_runner, _docker = scripted_run(
        first_root,
        [{"text": "done"}],
        task=task,
        task_directory=task_directory,
    )
    second, _second_runner, _docker = scripted_run(
        second_root,
        [{"text": "done"}],
        task=task,
        task_directory=task_directory,
    )

    instruction = (task_directory / "task.md").read_bytes()
    assert first.task_instruction_sha256 == hashlib.sha256(instruction).hexdigest()
    assert first.system_prompt_sha256 == second.system_prompt_sha256
    assert first.model_events[0].prompt_sha256 == first.system_prompt_sha256
    assert (
        first.model_events[0].tool_definitions_sha256
        == second.model_events[0].tool_definitions_sha256
    )
    assert len(first.model_events[0].tool_definitions_sha256) == 64


def test_browser_tools_and_exact_hosts_are_exposed_provider_neutrally(tmp_path: Path) -> None:
    browser = BrowserConfig(
        source_profiles=(BrowserSourceProfile.SEC_EDGAR,),
        user_agent="OneOxygen-Test/1.0 test@example.com",
    )
    task, task_directory = make_task(
        tmp_path,
        browser=browser,
        policy=ToolPolicy(
            allowed_tool_names=("browser_sources", "browser_open"),
            max_total_tool_calls=4,
        ),
        agent_updates={
            "data_classification": "public",
            "required_submission": False,
            "final_text_without_submission": "succeed",
        },
    )

    docker = FakeDockerAdapter()
    if os.name == "posix" and os.geteuid() != 0:
        docker.sandbox_user = f"{os.geteuid()}:{os.getegid()}"
    record, runner, docker = scripted_run(
        tmp_path,
        [{"text": "Browser tools are available.", "finish_reason": "completed"}],
        task=task,
        task_directory=task_directory,
        docker=docker,
    )

    assert record.final_status is RunStatus.SUCCEEDED
    assert {definition.name for definition in runner._tool_definitions} == {
        "browser_open",
        "browser_sources",
    }
    assert record.system_prompt_content is not None
    assert "Exact allowed hosts:" in record.system_prompt_content
    assert "www.sec.gov" in record.system_prompt_content
    assert record.browser_configuration == browser
    assert record.browser_policy_sha256 is not None
    assert docker.environments == [{}]


def test_provider_error_redacts_secrets_and_host_paths_from_run_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "runner-test-secret-value-123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    task, task_directory = make_task(tmp_path)
    config = model_config()
    error = ModelError(
        ModelErrorCode.AUTHENTICATION_FAILED,
        f"Authorization: Bearer {secret}; failed while reading {task_directory.resolve()}",
    )
    adapter = SequenceAdapter(config, [error])
    docker = FakeDockerAdapter()
    runner = AgentRunner(
        task,
        task_directory,
        config,
        adapter,
        runs_directory=tmp_path / "runs",
        sandbox_adapter=docker,
    )

    record = runner.run()
    serialized = json.dumps(record.model_dump(mode="json"), sort_keys=True)

    assert record.final_status is RunStatus.PROVIDER_ERROR
    assert secret not in serialized
    assert str(task_directory.resolve()) not in serialized
    assert docker.environments == [{}]
    assert secret not in runner.record_path.read_text(encoding="utf-8")


def test_wall_time_limit_can_stop_before_the_first_provider_request(tmp_path: Path) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"overall_wall_time_seconds": 1},
    )
    config = model_config()
    adapter = SequenceAdapter(
        config,
        [
            ModelTurnResponse(
                provider=ModelProvider.SCRIPTED,
                requested_model=config.model,
                text="must not be requested",
                finish_reason=NormalizedFinishReason.COMPLETED,
            )
        ],
    )
    docker = FakeDockerAdapter()
    runner = AgentRunner(
        task,
        task_directory,
        config,
        adapter,
        runs_directory=tmp_path / "runs",
        sandbox_adapter=docker,
        clock=JumpClock(0, 2),
    )

    record = runner.run()

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
    assert record.model_events == []
    assert adapter.requests == []
    assert docker.removed == 1


def test_wall_time_is_rechecked_after_a_successful_provider_response(tmp_path: Path) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"overall_wall_time_seconds": 1},
    )
    config = model_config()
    clock = MutableClock()
    adapter = AdvancingAdapter(
        config,
        [
            ModelTurnResponse(
                provider=ModelProvider.SCRIPTED,
                requested_model=config.model,
                text="returned too late",
                finish_reason=NormalizedFinishReason.COMPLETED,
            )
        ],
        clock,
    )
    runner = AgentRunner(
        task,
        task_directory,
        config,
        adapter,
        runs_directory=tmp_path / "runs",
        sandbox_adapter=FakeDockerAdapter(),
        clock=clock,
    )

    record = runner.run()

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
    assert len(record.model_events) == 1


def test_nonretryable_error_on_last_provider_request_remains_provider_error(
    tmp_path: Path,
) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"maximum_provider_requests": 1},
    )
    config = model_config(maximum_retry_attempts=3)

    record, _runner, _docker = scripted_run(
        tmp_path,
        [
            {
                "error": {
                    "code": "authentication_failed",
                    "message": "bad credentials",
                    "retryable": False,
                }
            }
        ],
        task=task,
        task_directory=task_directory,
        config=config,
    )

    assert record.final_status is RunStatus.PROVIDER_ERROR
    assert record.termination_reason is AgentTerminationReason.REPEATED_PROVIDER_FAILURE
    assert record.error is not None
    assert record.error.code == ModelErrorCode.AUTHENTICATION_FAILED.value


def test_provider_request_limit_is_reported_only_when_it_prevents_a_retry(
    tmp_path: Path,
) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"maximum_provider_requests": 1},
    )
    config = model_config(maximum_retry_attempts=3)

    record, _runner, _docker = scripted_run(
        tmp_path,
        [
            {
                "error": {
                    "code": "provider_unavailable",
                    "message": "temporary",
                    "retryable": True,
                }
            },
            {"text": "must not be requested"},
        ],
        task=task,
        task_directory=task_directory,
        config=config,
    )

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is AgentTerminationReason.MAXIMUM_PROVIDER_REQUESTS_REACHED
    assert record.metrics.provider_attempts == 1


def test_retry_backoff_reaching_wall_deadline_does_not_send_another_request(
    tmp_path: Path,
) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"overall_wall_time_seconds": 1},
    )
    config = model_config(maximum_retry_attempts=1, initial_retry_delay_seconds=2)
    clock = MutableClock()

    def advance(seconds: float) -> None:
        clock.value += seconds

    record, _runner, _docker = scripted_run(
        tmp_path,
        [
            {
                "error": {
                    "code": "provider_unavailable",
                    "message": "temporary",
                    "retryable": True,
                }
            },
            {"text": "must not be requested"},
        ],
        task=task,
        task_directory=task_directory,
        config=config,
        sleeper=advance,
        clock=clock,
    )

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
    assert record.metrics.provider_attempts == 1


def test_wall_deadline_rejects_later_calls_from_the_same_response(tmp_path: Path) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"overall_wall_time_seconds": 1},
        policy=ToolPolicy(
            allowed_tool_names=("execute_shell", "write_text_file", "submit_result"),
            shell_execution_allowed=True,
            max_total_tool_calls=5,
        ),
    )
    clock = MutableClock()

    class AdvancingDocker(FakeDockerAdapter):
        def execute(
            self,
            container: FakeContainer,
            command: str,
            working_directory: str,
            timeout_seconds: float,
            maximum_output_bytes: int,
        ) -> ExecResult:
            result = super().execute(
                container,
                command,
                working_directory,
                timeout_seconds,
                maximum_output_bytes,
            )
            clock.value = 2
            return result

    record, _runner, docker = scripted_run(
        tmp_path,
        [
            {
                "tool_calls": [
                    {
                        "call_id": "call-slow",
                        "tool_name": "execute_shell",
                        "arguments": {"command": "printf done"},
                    },
                    {
                        "call_id": "call-too-late",
                        "tool_name": "write_text_file",
                        "arguments": {"path": "output/late.txt", "content": "late"},
                    },
                ]
            }
        ],
        task=task,
        task_directory=task_directory,
        docker=AdvancingDocker(),
        clock=clock,
    )

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is AgentTerminationReason.OVERALL_WALL_TIME_LIMIT_REACHED
    assert [event.status.value for event in record.tool_events] == ["succeeded", "failed"]
    assert record.tool_events[1].error_code is ToolErrorCode.CALL_LIMIT_EXCEEDED
    assert docker.workspace is not None
    assert not (docker.workspace / "output" / "late.txt").exists()


def test_exhausted_tool_budget_stops_when_required_submission_is_impossible(
    tmp_path: Path,
) -> None:
    task, task_directory = make_task(
        tmp_path,
        policy=ToolPolicy(
            allowed_tool_names=("write_text_file", "submit_result"),
            max_total_tool_calls=1,
        ),
    )

    record, _runner, _docker = scripted_run(
        tmp_path,
        [
            {
                "tool_calls": [
                    {
                        "call_id": "call-only-budget",
                        "tool_name": "write_text_file",
                        "arguments": {"path": "output/draft.txt", "content": "draft"},
                    }
                ]
            }
        ],
        task=task,
        task_directory=task_directory,
    )

    assert record.final_status is RunStatus.INCOMPLETE
    assert record.termination_reason is AgentTerminationReason.TOOL_FAILURE
    assert record.metrics.provider_attempts == 1
    assert record.artifacts == []


def test_sandbox_start_failure_persists_complete_phase3_record(tmp_path: Path) -> None:
    task, task_directory = make_task(tmp_path)
    config = model_config()
    adapter = ScriptedModelAdapter(config, {"schema_version": 1, "turns": [{"text": "unused"}]})

    class StartFailingDocker(FakeDockerAdapter):
        def resolve_image(self, image: str) -> str:
            raise DockerOperationError("simulated image resolution failure")

    runner = AgentRunner(
        task,
        task_directory,
        config,
        adapter,
        runs_directory=tmp_path / "runs",
        sandbox_adapter=StartFailingDocker(),
    )

    record = runner.run()
    restored = persisted_record(runner)

    assert record.final_status is RunStatus.SANDBOX_ERROR
    assert record.termination_reason is AgentTerminationReason.SANDBOX_FAILURE
    assert record.model_configuration == config
    assert record.effective_model_settings == config.requested_settings()
    assert record.system_prompt_sha256 is not None
    assert record.system_prompt_content
    assert record.task_instruction_sha256 is not None
    assert restored == record


def test_provider_context_limit_has_a_distinct_termination_reason(tmp_path: Path) -> None:
    record, _runner, _docker = scripted_run(
        tmp_path,
        [
            {
                "error": {
                    "code": "context_limit_exceeded",
                    "message": "context is full",
                    "retryable": False,
                }
            }
        ],
    )

    assert record.final_status is RunStatus.LIMIT_EXCEEDED
    assert record.termination_reason is AgentTerminationReason.CONTEXT_LIMIT_EXCEEDED


def test_interrupt_during_provider_request_records_the_attempt(tmp_path: Path) -> None:
    task, task_directory = make_task(tmp_path)
    config = model_config()

    class InterruptingAdapter(SequenceAdapter):
        def generate_next_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
            self.requests.append(request)
            raise KeyboardInterrupt

    adapter = InterruptingAdapter(config, [])
    runner = AgentRunner(
        task,
        task_directory,
        config,
        adapter,
        runs_directory=tmp_path / "runs",
        sandbox_adapter=FakeDockerAdapter(),
    )

    record = runner.run()

    assert record.final_status is RunStatus.CANCELLED
    assert record.termination_reason is AgentTerminationReason.USER_INTERRUPTION
    assert len(record.model_events) == 1
    assert record.model_events[0].finish_reason is NormalizedFinishReason.CANCELLED
    assert record.model_events[0].attempts[0].error_code is ModelErrorCode.CANCELLED
    assert persisted_record(runner) == record


def test_provider_request_timeout_is_bounded_by_remaining_agent_wall_time(
    tmp_path: Path,
) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"overall_wall_time_seconds": 1},
    )
    config = model_config(model_call_timeout_seconds=10)
    clock = MutableClock()

    class StartAdvancingDocker(FakeDockerAdapter):
        def start_container(self, container: FakeContainer) -> None:
            super().start_container(container)
            clock.value = 0.75

    adapter = SequenceAdapter(
        config,
        [
            ModelTurnResponse(
                provider=ModelProvider.SCRIPTED,
                requested_model=config.model,
                text="finished",
                finish_reason=NormalizedFinishReason.COMPLETED,
            )
        ],
    )
    runner = AgentRunner(
        task,
        task_directory,
        config,
        adapter,
        runs_directory=tmp_path / "runs",
        sandbox_adapter=StartAdvancingDocker(),
        clock=clock,
    )

    record = runner.run()

    assert adapter.requests[0].request_timeout_seconds == pytest.approx(0.25)
    assert record.model_events[0].effective_settings[
        "effective_request_timeout_seconds"
    ] == pytest.approx(0.25)


def test_each_retry_recomputes_and_traces_its_remaining_wall_timeout(tmp_path: Path) -> None:
    task, task_directory = make_task(
        tmp_path,
        agent_updates={"overall_wall_time_seconds": 10},
    )
    config = model_config(
        model_call_timeout_seconds=60,
        maximum_retry_attempts=1,
        initial_retry_delay_seconds=1,
    )
    clock = MutableClock()

    def advance(seconds: float) -> None:
        clock.value += seconds

    record, _runner, _docker = scripted_run(
        tmp_path,
        [
            {
                "error": {
                    "code": "provider_unavailable",
                    "message": "temporary",
                    "retryable": True,
                }
            },
            {"text": "finished"},
        ],
        task=task,
        task_directory=task_directory,
        config=config,
        sleeper=advance,
        clock=clock,
    )

    attempts = record.model_events[0].attempts
    assert [attempt.request_timeout_seconds for attempt in attempts] == pytest.approx([10, 9])
    assert record.model_events[0].effective_settings[
        "effective_request_timeout_seconds"
    ] == pytest.approx(9)
