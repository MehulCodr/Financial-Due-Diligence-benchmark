from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from oneoxygen_sandbox.batching import (
    BatchArtifactStore,
    OpenAIBatchBackend,
    SQLiteBatchStore,
)
from oneoxygen_sandbox.batching.models import BatchRequest
from oneoxygen_sandbox.models import InferenceTransport, ModelProvider

pytestmark = pytest.mark.openai_batch_live


def test_opt_in_openai_batch_submit_and_cancel(tmp_path: Path) -> None:
    if os.environ.get("ONEOXYGEN_RUN_OPENAI_BATCH_LIVE_TESTS") != "1":
        pytest.skip("set ONEOXYGEN_RUN_OPENAI_BATCH_LIVE_TESTS=1 to opt in")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("ONEOXYGEN_OPENAI_BATCH_TEST_MODEL", "").strip()
    if not key or not model:
        pytest.skip("OPENAI_API_KEY and ONEOXYGEN_OPENAI_BATCH_TEST_MODEL are required")
    warnings.warn(
        "This test incurs OpenAI charges and a batch can take up to 24 hours.",
        UserWarning,
        stacklevel=1,
    )
    artifacts = BatchArtifactStore(tmp_path / "files")
    store = SQLiteBatchStore(tmp_path / "batch.sqlite3")
    body = {
        "model": model,
        "instructions": "Return only OK.",
        "input": [{"role": "user", "content": "Synthetic smoke test."}],
        "tools": [],
        "max_output_tokens": 8,
        "store": False,
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
    }
    reference, digest = artifacts.write_json("requests", "live", body)
    request = BatchRequest(
        internal_request_id="live-request",
        run_id="live-run",
        turn_number=1,
        attempt_number=1,
        provider=ModelProvider.OPENAI,
        transport=InferenceTransport.PROVIDER_BATCH,
        model=model,
        endpoint="/v1/responses",
        compiled_request_body_sha256=digest,
        compiled_request_body_reference=reference,
        tool_schema_sha256="0" * 64,
        system_prompt_version="live-v1",
        schema_mode="portable",
        effective_generation_settings_sha256="0" * 64,
        data_policy_class="synthetic",
    )
    backend = OpenAIBatchBackend(artifacts, store)
    submitted = backend.submit_batch(backend.build_batch((request,)))
    assert submitted.provider_batch_id
    cancelled = backend.cancel_batch(submitted)
    assert cancelled.state.value in {"cancelling", "cancelled"}
