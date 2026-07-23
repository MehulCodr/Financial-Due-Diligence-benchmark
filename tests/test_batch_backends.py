from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from oneoxygen_sandbox.batching import (
    BatchArtifactStore,
    BatchState,
    MockBatchBackend,
    OpenAIBatchBackend,
    SQLiteBatchStore,
)
from oneoxygen_sandbox.batching.models import BatchRequest
from oneoxygen_sandbox.errors import ConfigurationError, ModelError
from oneoxygen_sandbox.model_adapters.openai import (
    apply_openai_batch_response_state,
    compile_openai_batch_turn,
    compile_openai_responses_request,
    parse_openai_responses_response,
)
from oneoxygen_sandbox.models import (
    InferenceTransport,
    ModelErrorCode,
    ModelProvider,
    ModelRunConfig,
    ModelTurnRequest,
    ToolResult,
)

HASH = "b" * 64


def _request(
    artifacts: BatchArtifactStore,
    identifier: str,
    *,
    model: str = "configured-model",
) -> BatchRequest:
    body = {
        "model": model,
        "instructions": "system",
        "input": [{"role": "user", "content": "synthetic"}],
        "tools": [],
        "max_output_tokens": 10,
        "store": False,
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
    }
    reference, digest = artifacts.write_json("requests", identifier, body)
    return BatchRequest(
        internal_request_id=identifier,
        run_id=f"run-{identifier}",
        turn_number=1,
        attempt_number=1,
        provider=ModelProvider.OPENAI,
        transport=InferenceTransport.PROVIDER_BATCH,
        model=model,
        endpoint="/v1/responses",
        compiled_request_body_sha256=digest,
        compiled_request_body_reference=reference,
        tool_schema_sha256=HASH,
        system_prompt_version="v1",
        schema_mode="portable",
        effective_generation_settings_sha256=HASH,
    )


def _stores(tmp_path: Path) -> tuple[BatchArtifactStore, SQLiteBatchStore]:
    return (
        BatchArtifactStore(tmp_path / "files"),
        SQLiteBatchStore(tmp_path / "state.sqlite3"),
    )


def test_mock_batch_out_of_order_partial_missing_and_restart(tmp_path: Path) -> None:
    artifacts, store = _stores(tmp_path)
    requests = tuple(_request(artifacts, value) for value in ("a", "b", "c"))
    backend = MockBatchBackend(
        artifacts,
        store,
        responses={
            "a": {"id": "r-a", "model": "configured-model", "status": "completed", "output": []},
            "c": {"id": "r-c", "model": "configured-model", "status": "completed", "output": []},
        },
        errors={"b": {"code": "server_error", "message": "retry", "retryable": True}},
        delayed_status_polls=1,
        missing_result_ids=("c",),
    )
    job = backend.submit_batch(backend.build_batch(requests))
    assert backend.get_status(job).state is BatchState.IN_PROGRESS

    restarted = MockBatchBackend(
        artifacts,
        SQLiteBatchStore(tmp_path / "state.sqlite3"),
        responses=backend.responses,
        errors=backend.errors,
        delayed_status_polls=1,
        missing_result_ids=("c",),
    )
    completed = restarted.get_status(job)
    results = restarted.retrieve_results(completed)
    by_id = {result.request_id: result for result in results}
    assert by_id["a"].success
    assert by_id["b"].retryable
    assert by_id["c"].normalized_error["code"] == "missing_result"


@pytest.mark.parametrize(
    ("duplicate", "unknown"),
    [("a", False), (None, True)],
)
def test_mock_batch_rejects_duplicate_and_unknown_results(
    tmp_path: Path, duplicate: str | None, unknown: bool
) -> None:
    artifacts, store = _stores(tmp_path)
    backend = MockBatchBackend(
        artifacts,
        store,
        duplicate_result_id=duplicate,
        include_unknown_result=unknown,
    )
    job = backend.submit_batch(backend.build_batch((_request(artifacts, "a"),)))
    completed = backend.get_status(job)
    with pytest.raises(ModelError) as captured:
        backend.retrieve_results(completed)
    assert captured.value.model_code is ModelErrorCode.BATCH_CORRELATION_ERROR


def test_mock_batch_expiration_and_cancellation(tmp_path: Path) -> None:
    artifacts, store = _stores(tmp_path)
    expiring = MockBatchBackend(artifacts, store, expire=True)
    job = expiring.submit_batch(expiring.build_batch((_request(artifacts, "a"),)))
    assert expiring.get_status(job).state is BatchState.EXPIRED

    artifacts2, store2 = _stores(tmp_path / "cancel")
    backend = MockBatchBackend(artifacts2, store2)
    second = backend.submit_batch(backend.build_batch((_request(artifacts2, "b"),)))
    assert backend.cancel_batch(second).state is BatchState.CANCELLED

    artifacts3, store3 = _stores(tmp_path / "provider-failure")
    failing = MockBatchBackend(artifacts3, store3, provider_failure=True)
    failed = failing.submit_batch(failing.build_batch((_request(artifacts3, "c"),)))
    assert failing.get_status(failed).state is BatchState.FAILED


class _FakeFiles:
    def __init__(self, output: bytes, errors: bytes) -> None:
        self.output = output
        self.errors = errors
        self.upload_purpose = None
        self.upload_data = b""

    def create(self, *, file, purpose: str):
        self.upload_purpose = purpose
        self.upload_data = file.read()
        return SimpleNamespace(id="file-input")

    def content(self, file_id: str):
        return SimpleNamespace(content=self.output if file_id == "file-output" else self.errors)


class _FakeBatches:
    def __init__(self) -> None:
        self.create_kwargs = None
        self.cancelled = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id="batch-remote", status="validating")

    def retrieve(self, batch_id: str):
        assert batch_id == "batch-remote"
        return SimpleNamespace(
            status="completed",
            output_file_id="file-output",
            error_file_id="file-errors",
            request_counts=SimpleNamespace(completed=1, failed=1),
        )

    def cancel(self, batch_id: str):
        self.cancelled = batch_id
        return SimpleNamespace(status="cancelling")


def test_openai_batch_upload_shape_status_retrieval_and_cancel(tmp_path: Path) -> None:
    artifacts, store = _stores(tmp_path)
    requests = (_request(artifacts, "a"), _request(artifacts, "b"))
    files = _FakeFiles(b"", b"")
    batches = _FakeBatches()
    client = SimpleNamespace(files=files, batches=batches)
    backend = OpenAIBatchBackend(
        artifacts,
        store,
        client=client,
        environ={"OPENAI_API_KEY": "unit-test-secret-value"},
    )
    job = backend.build_batch(requests)
    mappings = store.mappings(job.internal_batch_id)
    custom_ids = list(mappings)
    files.output = (
        json.dumps(
            {
                "custom_id": custom_ids[1],
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "response-b",
                        "model": "configured-model",
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "total_tokens": 3,
                        },
                    },
                },
                "error": None,
            }
        ).encode()
        + b"\n"
    )
    files.errors = (
        json.dumps(
            {
                "custom_id": custom_ids[0],
                "response": None,
                "error": {"code": "server_error", "message": "retry"},
            }
        ).encode()
        + b"\n"
    )

    submitted = backend.submit_batch(job)
    assert files.upload_purpose == "batch"
    assert b"unit-test-secret-value" not in files.upload_data
    lines = [json.loads(line) for line in files.upload_data.splitlines()]
    assert all(line["url"] == "/v1/responses" for line in lines)
    assert all(line["method"] == "POST" for line in lines)
    assert all(line["body"].get("stream") is not True for line in lines)
    assert batches.create_kwargs == {
        "input_file_id": "file-input",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }

    completed = backend.get_status(submitted)
    assert completed.state is BatchState.PARTIALLY_COMPLETED
    results = backend.retrieve_results(completed)
    assert sum(result.success for result in results) == 1
    assert sum(result.retryable for result in results) == 1
    assert backend.cancel_batch(submitted).state is BatchState.CANCELLING


def test_openai_batch_enforces_one_model_per_file(tmp_path: Path) -> None:
    artifacts, store = _stores(tmp_path)
    backend = OpenAIBatchBackend(artifacts, store)
    with pytest.raises(ConfigurationError, match="one model"):
        backend.build_batch(
            (
                _request(artifacts, "a", model="model-a"),
                _request(artifacts, "b", model="model-b"),
            )
        )


def test_openai_capabilities_record_documented_provider_discount(tmp_path: Path) -> None:
    artifacts, store = _stores(tmp_path)
    discount = OpenAIBatchBackend(artifacts, store).capabilities().discount
    assert discount is not None
    assert discount.documented_discount_fraction == 0.5
    assert discount.estimated is False
    assert discount.provider_reported is True
    assert discount.source_url == "https://developers.openai.com/api/docs/guides/batch"


def test_openai_unknown_submission_state_prevents_resubmission(tmp_path: Path) -> None:
    artifacts, store = _stores(tmp_path)

    class TimeoutBatches:
        def create(self, **_kwargs):
            raise TimeoutError("uncertain")

    client = SimpleNamespace(
        files=SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(id="file-input")),
        batches=TimeoutBatches(),
    )
    backend = OpenAIBatchBackend(
        artifacts,
        store,
        client=client,
        environ={"OPENAI_API_KEY": "unit-test-secret-value"},
    )
    job = backend.build_batch((_request(artifacts, "a"),))
    with pytest.raises(ModelError) as first:
        backend.submit_batch(job)
    assert first.value.model_code is ModelErrorCode.REMOTE_STATE_UNKNOWN
    with pytest.raises(ModelError) as second:
        backend.submit_batch(job)
    assert second.value.model_code is ModelErrorCode.REMOTE_STATE_UNKNOWN


def test_openai_direct_and_batch_share_identical_api_body_compilation() -> None:
    direct_config = ModelRunConfig(
        provider=ModelProvider.OPENAI,
        model="configured-model",
        transport=InferenceTransport.DIRECT,
        maximum_output_tokens=77,
        temperature=0.25,
    )
    batch_config = direct_config.model_copy(update={"transport": InferenceTransport.PROVIDER_BATCH})
    batch_config = ModelRunConfig.model_validate(batch_config.model_dump(mode="python"))
    direct_request = ModelTurnRequest(
        turn_number=1,
        system_prompt="same system",
        initial_task_instruction="same task",
        tool_definitions=(),
        run_config=direct_config,
    )
    batch_request = direct_request.model_copy(update={"run_config": batch_config})
    input_items = [{"role": "user", "content": "same task"}]
    direct_body = compile_openai_responses_request(direct_config, direct_request, input_items)
    direct_body.pop("timeout")
    batch_body, pending = compile_openai_batch_turn(batch_config, batch_request)
    assert direct_body == batch_body
    assert pending["input_items"] == input_items


def test_openai_batch_conversation_continues_with_shared_parser() -> None:
    config = ModelRunConfig(
        provider=ModelProvider.OPENAI,
        model="configured-model",
        transport=InferenceTransport.PROVIDER_BATCH,
    )
    first_request = ModelTurnRequest(
        turn_number=1,
        system_prompt="system",
        initial_task_instruction="task",
        tool_definitions=(),
        run_config=config,
    )
    _first_body, pending = compile_openai_batch_turn(config, first_request)
    raw_response = {
        "id": "resp-batch",
        "model": "returned-model",
        "status": "completed",
        "output_text": "",
        "output": [
            {
                "id": "fc-batch",
                "type": "function_call",
                "status": "completed",
                "call_id": "call-batch",
                "name": "read_text_file",
                "arguments": '{"path":"input.txt"}',
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }
    parsed, serialized = parse_openai_responses_response(config, raw_response, latency_seconds=0)
    assert parsed.tool_calls[0].call_id == "call-batch"
    continued_state = apply_openai_batch_response_state(pending, serialized, 1)
    now = datetime.now(UTC)
    second_request = ModelTurnRequest(
        turn_number=2,
        system_prompt="system",
        initial_task_instruction="task",
        tool_definitions=(),
        tool_results=(
            ToolResult(
                call_id="call-batch",
                tool_name="read_text_file",
                success=True,
                content={"text": "synthetic"},
                start_timestamp=now,
                end_timestamp=now,
                duration_seconds=0,
            ),
        ),
        run_config=config,
    )
    second_body, _pending = compile_openai_batch_turn(config, second_request, continued_state)
    assert [item.get("type", "message") for item in second_body["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
