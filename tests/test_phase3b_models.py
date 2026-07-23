from __future__ import annotations

import os
from pathlib import Path

import pytest

from oneoxygen_sandbox.batching.models import (
    BatchItemResult,
    BatchRequest,
    deterministic_custom_id,
    group_compatible_requests,
    retry_failed_requests,
)
from oneoxygen_sandbox.checkpoints import WorkspaceCheckpoint
from oneoxygen_sandbox.errors import LifecycleError, PathSafetyError
from oneoxygen_sandbox.models import (
    InferenceTransport,
    ModelProvider,
    ModelRunConfig,
    ProvenanceClassification,
)
from oneoxygen_sandbox.orchestration import (
    AgentRunState,
    AgentRunStatus,
    SQLiteAgentStateStore,
    retryable_failed_items,
)

HASH = "a" * 64


def test_transport_and_provenance_are_derived_without_merging_routes() -> None:
    official = ModelRunConfig(provider=ModelProvider.OPENAI, model="configured-model")
    gateway = ModelRunConfig(
        provider=ModelProvider.AIRFORCE,
        model="gateway-model",
        transport=InferenceTransport.GATEWAY_DIRECT,
    )
    scripted = ModelRunConfig(provider=ModelProvider.SCRIPTED, model="script")

    assert official.provenance is ProvenanceClassification.OFFICIAL_PROVIDER
    assert official.api_host == "api.openai.com"
    assert gateway.provenance is ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED
    assert gateway.api_host == "api.airforce"
    assert gateway.upstream_provider_verifiable is False
    assert scripted.provenance is ProvenanceClassification.SCRIPTED_TEST


def test_airforce_cannot_be_relabelled_as_an_official_route() -> None:
    with pytest.raises(ValueError):
        ModelRunConfig(
            provider=ModelProvider.AIRFORCE,
            model="gateway-model",
            transport=InferenceTransport.GATEWAY_DIRECT,
            provenance=ProvenanceClassification.OFFICIAL_PROVIDER,
        )


def test_state_machine_transitions_are_atomic_and_recoverable(tmp_path: Path) -> None:
    store_path = tmp_path / "states.sqlite3"
    store = SQLiteAgentStateStore(str(store_path))
    config = ModelRunConfig(
        provider=ModelProvider.OPENAI,
        model="configured-model",
        transport=InferenceTransport.PROVIDER_BATCH,
    )
    state = AgentRunState(
        run_id="run-1",
        task_id="task",
        task_version="1",
        model_configuration=config,
        transport=config.transport,
        provenance=config.provenance,
        prompt_sha256=HASH,
        tool_schema_sha256=HASH,
    )
    store.create(state)
    ready = store.transition("run-1", AgentRunStatus.READY_FOR_MODEL)

    restarted = SQLiteAgentStateStore(str(store_path))
    assert restarted.load("run-1") == ready
    with pytest.raises(LifecycleError):
        restarted.transition("run-1", AgentRunStatus.COMPLETED)
    with pytest.raises(LifecycleError):
        restarted.transition(
            "run-1",
            AgentRunStatus.WAITING_FOR_MODEL,
            expected_revision=0,
        )


def test_checkpoint_creation_restore_and_hash_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    (workspace / "nested" / "report.txt").write_text("safe\n", encoding="utf-8")
    checkpoints = WorkspaceCheckpoint(tmp_path / "checkpoints", "run-1")

    manifest = checkpoints.capture(workspace, 0)
    restored = tmp_path / "restored"
    restored.mkdir()
    checkpoints.restore(0, restored)

    assert manifest.total_size_bytes == len((workspace / "nested" / "report.txt").read_bytes())
    assert (restored / "nested" / "report.txt").read_text(encoding="utf-8") == "safe\n"
    checkpoint_file = checkpoints.generation_path(0) / "workspace/nested/report.txt"
    os.chmod(checkpoint_file, 0o600)
    checkpoint_file.write_text("tampered", encoding="utf-8")
    with pytest.raises(PathSafetyError, match="hash"):
        checkpoints.verify(0)


def test_checkpoint_size_limit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "large.bin").write_bytes(b"x" * 9)
    checkpoint = WorkspaceCheckpoint(
        tmp_path / "checkpoints",
        "run-1",
        maximum_size_bytes=8,
    )
    with pytest.raises(PathSafetyError, match="size"):
        checkpoint.capture(workspace, 0)


def test_checkpoint_rejects_symbolic_links_when_supported(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "target"
    target.write_text("secret", encoding="utf-8")
    try:
        (workspace / "linked").symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")
    checkpoint = WorkspaceCheckpoint(tmp_path / "checkpoints", "run-1")
    with pytest.raises(PathSafetyError, match="symbolic"):
        checkpoint.capture(workspace, 0)


def _request(identifier: str, *, model: str = "model-a") -> BatchRequest:
    return BatchRequest(
        internal_request_id=identifier,
        run_id=f"run-{identifier}",
        turn_number=1,
        attempt_number=1,
        provider=ModelProvider.OPENAI,
        model=model,
        endpoint="/v1/responses",
        compiled_request_body_sha256=HASH,
        compiled_request_body_reference=f"requests/{identifier}.json",
        tool_schema_sha256=HASH,
        system_prompt_version="v1",
        schema_mode="portable",
        effective_generation_settings_sha256=HASH,
    )


def test_batch_grouping_includes_model_and_all_compatibility_dimensions() -> None:
    groups = group_compatible_requests(
        [_request("a"), _request("b"), _request("c", model="model-b")]
    )
    assert sorted(len(group) for group in groups) == [1, 2]


def test_custom_ids_are_deterministic_opaque_and_unique() -> None:
    first = deterministic_custom_id("customer-looking-run-id", 1, 1)
    assert first == deterministic_custom_id("customer-looking-run-id", 1, 1)
    assert first != deterministic_custom_id("customer-looking-run-id", 2, 1)
    assert "customer" not in first
    assert len(first) <= 64


def test_retry_selection_never_repeats_successful_items() -> None:
    results = (
        BatchItemResult(
            request_id="success",
            provider_custom_id="custom-success",
            success=True,
            response={"ok": True},
        ),
        BatchItemResult(
            request_id="retry",
            provider_custom_id="custom-retry",
            success=False,
            normalized_error={"code": "server_error"},
            retryable=True,
        ),
        BatchItemResult(
            request_id="permanent",
            provider_custom_id="custom-permanent",
            success=False,
            normalized_error={"code": "invalid_request"},
            retryable=False,
        ),
    )
    assert retryable_failed_items(
        ("success", "retry", "permanent"),
        results,
        maximum_retries=2,
    ) == ("retry",)


def test_retry_request_uses_new_attempt_and_preserves_compiled_body() -> None:
    original = _request("retry")
    retry = retry_failed_requests(
        (original,),
        (
            BatchItemResult(
                request_id=original.internal_request_id,
                provider_custom_id="custom-retry",
                success=False,
                normalized_error={"code": "server_error"},
                retryable=True,
            ),
        ),
        maximum_retries=2,
    )[0]

    assert retry.internal_request_id != original.internal_request_id
    assert retry.attempt_number == 2
    assert retry.provider_custom_id is None
    assert retry.compiled_request_body_sha256 == original.compiled_request_body_sha256
    assert retry.compiled_request_body_reference == original.compiled_request_body_reference
