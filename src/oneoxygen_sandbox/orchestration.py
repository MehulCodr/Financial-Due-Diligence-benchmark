"""Durable agent state machine and SQLite recovery store."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field, field_validator, model_validator

from oneoxygen_sandbox.batching.models import BatchItemResult
from oneoxygen_sandbox.errors import LifecycleError
from oneoxygen_sandbox.models import (
    AgentTerminationReason,
    InferenceTransport,
    ModelRunConfig,
    ModelTurnRequest,
    ModelUsage,
    ProvenanceClassification,
    StrictModel,
    ToolDefinition,
)

AGENT_RUN_STATE_SCHEMA_VERSION = 1


class AgentRunStatus(StrEnum):
    CREATED = "created"
    READY_FOR_MODEL = "ready_for_model"
    BATCH_DRAFTED = "batch_drafted"
    BATCH_SUBMITTED = "batch_submitted"
    WAITING_FOR_MODEL = "waiting_for_model"
    MODEL_RESULT_READY = "model_result_ready"
    EXECUTING_TOOLS = "executing_tools"
    READY_FOR_NEXT_TURN = "ready_for_next_turn"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS: dict[AgentRunStatus, frozenset[AgentRunStatus]] = {
    AgentRunStatus.CREATED: frozenset(
        {AgentRunStatus.READY_FOR_MODEL, AgentRunStatus.FAILED, AgentRunStatus.CANCELLED}
    ),
    AgentRunStatus.READY_FOR_MODEL: frozenset(
        {
            AgentRunStatus.BATCH_DRAFTED,
            AgentRunStatus.WAITING_FOR_MODEL,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.BATCH_DRAFTED: frozenset(
        {
            AgentRunStatus.BATCH_SUBMITTED,
            AgentRunStatus.READY_FOR_MODEL,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.BATCH_SUBMITTED: frozenset(
        {
            AgentRunStatus.WAITING_FOR_MODEL,
            AgentRunStatus.FAILED,
            AgentRunStatus.EXPIRED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.WAITING_FOR_MODEL: frozenset(
        {
            AgentRunStatus.MODEL_RESULT_READY,
            AgentRunStatus.READY_FOR_MODEL,
            AgentRunStatus.FAILED,
            AgentRunStatus.EXPIRED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.MODEL_RESULT_READY: frozenset(
        {
            AgentRunStatus.EXECUTING_TOOLS,
            AgentRunStatus.COMPLETED,
            AgentRunStatus.INCOMPLETE,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.EXECUTING_TOOLS: frozenset(
        {
            AgentRunStatus.READY_FOR_NEXT_TURN,
            AgentRunStatus.SUBMITTED,
            AgentRunStatus.INCOMPLETE,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.READY_FOR_NEXT_TURN: frozenset(
        {
            AgentRunStatus.READY_FOR_MODEL,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.SUBMITTED: frozenset({AgentRunStatus.COMPLETED, AgentRunStatus.FAILED}),
    AgentRunStatus.COMPLETED: frozenset(),
    AgentRunStatus.INCOMPLETE: frozenset(),
    AgentRunStatus.FAILED: frozenset(),
    AgentRunStatus.EXPIRED: frozenset(),
    AgentRunStatus.CANCELLED: frozenset(),
}


class BatchRequestReference(StrictModel):
    internal_request_id: str = Field(min_length=1, max_length=128)
    internal_batch_id: str | None = Field(default=None, max_length=128)
    provider_batch_id: str | None = Field(default=None, max_length=512)
    provider_custom_id: str | None = Field(default=None, max_length=128)
    compiled_request_body_sha256: str | None = None
    compiled_request_body_reference: str | None = Field(default=None, max_length=1_024)
    turn_number: int = Field(ge=1)
    attempt_number: int = Field(ge=1, le=99)

    @field_validator("compiled_request_body_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("batch request reference hash must be a SHA-256 digest")
        return value

    @field_validator("compiled_request_body_reference")
    @classmethod
    def validate_optional_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("batch request reference must be a safe relative path")
        return path.as_posix()


class AgentRunState(StrictModel):
    schema_version: int = AGENT_RUN_STATE_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    run_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    task_version: str = Field(min_length=1, max_length=128)
    model_configuration: ModelRunConfig
    transport: InferenceTransport
    provenance: ProvenanceClassification
    api_host: str | None = Field(default=None, min_length=1, max_length=255)
    official_route: bool | None = None
    upstream_provider_verifiable: bool | None = None
    experiment_namespace: str | None = Field(default=None, min_length=1, max_length=128)
    status: AgentRunStatus = AgentRunStatus.CREATED
    current_turn: int = Field(default=1, ge=1)
    normalized_conversation_history: tuple[dict[str, Any], ...] = ()
    model_trace: tuple[dict[str, Any], ...] = ()
    tool_trace: tuple[dict[str, Any], ...] = ()
    provider_conversation_state: dict[str, Any] = Field(default_factory=dict)
    prompt_sha256: str
    tool_schema_sha256: str
    usage_totals: ModelUsage = Field(default_factory=ModelUsage)
    total_tool_calls: int = Field(default=0, ge=0)
    per_tool_call_totals: dict[str, int] = Field(default_factory=dict)
    batch_request_references: tuple[BatchRequestReference, ...] = ()
    retry_counts: dict[str, int] = Field(default_factory=dict)
    termination_reason: AgentTerminationReason | None = None
    termination_message: str | None = Field(default=None, max_length=2_000)
    workspace_checkpoint_reference: str | None = Field(default=None, max_length=1_024)
    workspace_checkpoint_generation: int | None = Field(default=None, ge=0)
    pending_tool_results: tuple[dict[str, Any], ...] = ()
    system_prompt: str = Field(default="", max_length=256_000)
    initial_task_instruction: str = Field(default="", max_length=1_000_000)
    tool_definitions: tuple[ToolDefinition, ...] = ()
    system_prompt_version: str = Field(default="standard_agent_v1", max_length=128)
    task_configuration: dict[str, Any] = Field(default_factory=dict)
    task_directory_reference: str | None = Field(default=None, max_length=4_096)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    @field_validator("prompt_sha256", "tool_schema_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("agent-state hashes must be lowercase SHA-256 digests")
        return value

    @field_validator("workspace_checkpoint_reference")
    @classmethod
    def validate_checkpoint_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("checkpoint reference must be a safe relative path")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_route(self) -> AgentRunState:
        if (
            self.transport is not self.model_configuration.transport
            or self.provenance is not self.model_configuration.provenance
        ):
            raise ValueError("agent-state route must match its immutable model configuration")
        expected_official = (
            self.model_configuration.provenance is ProvenanceClassification.OFFICIAL_PROVIDER
        )
        expected_namespace = (
            "gateway_unverified"
            if self.model_configuration.provenance
            is ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED
            else "official"
            if expected_official
            else "scripted_test"
        )
        route_values = (
            (self.api_host, self.model_configuration.api_host, "API host"),
            (
                self.official_route,
                expected_official,
                "official-route classification",
            ),
            (
                self.upstream_provider_verifiable,
                self.model_configuration.upstream_provider_verifiable,
                "upstream verification",
            ),
            (self.experiment_namespace, expected_namespace, "experiment namespace"),
        )
        for actual, expected, label in route_values:
            if actual is not None and actual != expected:
                raise ValueError(f"agent-state {label} must match its immutable route")
        object.__setattr__(self, "api_host", self.model_configuration.api_host)
        object.__setattr__(self, "official_route", expected_official)
        object.__setattr__(
            self,
            "upstream_provider_verifiable",
            self.model_configuration.upstream_provider_verifiable,
        )
        object.__setattr__(self, "experiment_namespace", expected_namespace)
        if (
            self.workspace_checkpoint_reference is None
            and self.workspace_checkpoint_generation is not None
        ):
            raise ValueError("checkpoint generation requires a checkpoint reference")
        return self


class ReadyModelTurn(StrictModel):
    run_id: str = Field(min_length=1, max_length=128)
    turn_number: int = Field(ge=1)
    attempt_number: int = Field(default=1, ge=1, le=99)
    request: ModelTurnRequest
    provider_conversation_state: dict[str, Any] = Field(default_factory=dict)
    prompt_sha256: str
    tool_schema_sha256: str
    system_prompt_version: str = Field(min_length=1, max_length=128)
    data_policy_class: str | None = Field(default=None, max_length=64)


class AgentStateMachine:
    """Validate all direct and suspended-run lifecycle changes fail closed."""

    def __init__(self, status: AgentRunStatus = AgentRunStatus.CREATED) -> None:
        self.status = status

    def transition(self, target: AgentRunStatus) -> AgentRunStatus:
        allowed = _ALLOWED_TRANSITIONS[self.status]
        if target not in allowed:
            raise LifecycleError(
                f"invalid agent state transition: {self.status.value} -> {target.value}"
            )
        self.status = target
        return target

    @staticmethod
    def validate(source: AgentRunStatus, target: AgentRunStatus) -> None:
        AgentStateMachine(source).transition(target)


class SQLiteAgentStateStore:
    """Atomic revision-checked state transitions surviving process restarts."""

    def __init__(self, path: str) -> None:
        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(resolved)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_run_states (
                    run_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_run_status
                    ON agent_run_states(status);
                """
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create(self, state: AgentRunState) -> AgentRunState:
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO agent_run_states(
                        run_id, revision, status, payload_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        state.run_id,
                        state.revision,
                        state.status.value,
                        state.model_dump_json(),
                        state.updated_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise LifecycleError("agent run ID already exists") from exc
        return state

    def load(self, run_id: str) -> AgentRunState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM agent_run_states WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise LifecycleError("agent run state was not found")
        return AgentRunState.model_validate_json(row[0])

    def list(self, status: AgentRunStatus | None = None) -> tuple[AgentRunState, ...]:
        with self._connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT payload_json FROM agent_run_states ORDER BY run_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT payload_json FROM agent_run_states
                    WHERE status = ? ORDER BY run_id
                    """,
                    (status.value,),
                ).fetchall()
        return tuple(AgentRunState.model_validate_json(row[0]) for row in rows)

    def transition(
        self,
        run_id: str,
        target: AgentRunStatus,
        *,
        updates: dict[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> AgentRunState:
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT revision, payload_json FROM agent_run_states WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise LifecycleError("agent run state was not found")
            revision, payload = int(row[0]), str(row[1])
            if expected_revision is not None and revision != expected_revision:
                raise LifecycleError("agent run state revision conflict")
            current = AgentRunState.model_validate_json(payload)
            AgentStateMachine.validate(current.status, target)
            updated = current.model_copy(
                update={
                    **(updates or {}),
                    "status": target,
                    "revision": revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            validated = AgentRunState.model_validate(updated.model_dump(mode="python"))
            cursor = connection.execute(
                """
                UPDATE agent_run_states
                SET revision = ?, status = ?, payload_json = ?, updated_at = ?
                WHERE run_id = ? AND revision = ?
                """,
                (
                    validated.revision,
                    validated.status.value,
                    validated.model_dump_json(),
                    validated.updated_at.isoformat(),
                    run_id,
                    revision,
                ),
            )
            if cursor.rowcount != 1:
                raise LifecycleError("agent run state transition lost a revision race")
        return validated


def retryable_failed_items(
    requests: tuple[str, ...],
    results: tuple[BatchItemResult, ...],
    *,
    maximum_retries: int,
    prior_retry_counts: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Select only failed retryable IDs; successful requests are never repeated."""
    known = set(requests)
    counts = prior_retry_counts or {}
    selected: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.request_id not in known or result.request_id in seen:
            continue
        seen.add(result.request_id)
        if (
            not result.success
            and result.retryable
            and counts.get(result.request_id, 0) < maximum_retries
        ):
            selected.append(result.request_id)
    return tuple(sorted(selected))
