from __future__ import annotations

import os

import pytest

from oneoxygen_sandbox.model_adapters.airforce import AirforceModelAdapter
from oneoxygen_sandbox.models import (
    DataClassification,
    InferenceTransport,
    ModelProvider,
    ModelRunConfig,
    ModelTurnRequest,
    ToolDefinition,
)

pytestmark = pytest.mark.airforce_live


def test_opt_in_synthetic_airforce_tool_smoke() -> None:
    if os.environ.get("ONEOXYGEN_RUN_AIRFORCE_LIVE_TESTS") != "1":
        pytest.skip("set ONEOXYGEN_RUN_AIRFORCE_LIVE_TESTS=1 to opt in")
    key = os.environ.get("AIRFORCE_API_KEY", "").strip()
    model = os.environ.get("ONEOXYGEN_AIRFORCE_TEST_MODEL", "").strip()
    if not key or not model:
        pytest.skip("AIRFORCE_API_KEY and ONEOXYGEN_AIRFORCE_TEST_MODEL are required")
    config = ModelRunConfig(
        provider=ModelProvider.AIRFORCE,
        model=model,
        maximum_output_tokens=32,
        transport=InferenceTransport.GATEWAY_DIRECT,
    )
    adapter = AirforceModelAdapter(
        config,
        data_classification=DataClassification.SYNTHETIC,
        allow_third_party_gateway=True,
    )
    try:
        catalog = {item["id"]: item for item in adapter.discover_models()}
        selected = catalog.get(model)
        if selected is None:
            pytest.skip("selected model is not available to this account")
        if not selected["supports_tools"]:
            pytest.skip("selected model lacks reported tool support; text-only validation remains")
        if (
            selected["input_price_cents_per_million"] != 0
            or selected["output_price_cents_per_million"] != 0
        ):
            pytest.skip("selected model is not explicitly zero-cost")
        request = ModelTurnRequest(
            turn_number=1,
            system_prompt="Use the supplied function. The data is entirely synthetic.",
            initial_task_instruction="Call record_value with value 1.",
            tool_definitions=(
                ToolDefinition(
                    name="record_value",
                    description="Record one synthetic integer.",
                    arguments_schema={
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),
            ),
            run_config=config,
        )
        adapter.start_conversation(request)
        response = adapter.generate_next_turn(request)
        if not response.tool_calls:
            pytest.skip("transport worked but the selected free model did not call the tool")
        assert response.provider is ModelProvider.AIRFORCE
        assert response.returned_model
        assert response.upstream_provider_verifiable is False
    finally:
        adapter.close()
