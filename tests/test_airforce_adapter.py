from __future__ import annotations

from types import SimpleNamespace

import pytest

from oneoxygen_sandbox.errors import ModelError
from oneoxygen_sandbox.model_adapters.airforce import (
    AIRFORCE_BASE_URL,
    AirforceModelAdapter,
)
from oneoxygen_sandbox.models import (
    DataClassification,
    InferenceTransport,
    ModelErrorCode,
    ModelProvider,
    ModelRunConfig,
    ModelTurnRequest,
    ProvenanceClassification,
)


def _config() -> ModelRunConfig:
    return ModelRunConfig(
        provider=ModelProvider.AIRFORCE,
        model="gateway-model",
        transport=InferenceTransport.GATEWAY_DIRECT,
    )


def _request(config: ModelRunConfig) -> ModelTurnRequest:
    return ModelTurnRequest(
        turn_number=1,
        system_prompt="system",
        initial_task_instruction="synthetic task",
        tool_definitions=(),
        run_config=config,
    )


@pytest.mark.parametrize(
    "classification",
    [
        None,
        DataClassification.INTERNAL,
        DataClassification.CONFIDENTIAL,
        DataClassification.RESTRICTED,
    ],
)
def test_airforce_rejects_missing_or_sensitive_classification(
    classification: DataClassification | None,
) -> None:
    with pytest.raises(ModelError) as captured:
        AirforceModelAdapter(
            _config(),
            data_classification=classification,
            allow_third_party_gateway=True,
            client=object(),
            environ={"AIRFORCE_API_KEY": "unit-air-test-value"},
        )
    assert captured.value.model_code is ModelErrorCode.DATA_POLICY_VIOLATION


def test_airforce_requires_explicit_acknowledgement() -> None:
    with pytest.raises(ModelError) as captured:
        AirforceModelAdapter(
            _config(),
            data_classification=DataClassification.SYNTHETIC,
            client=object(),
            environ={"AIRFORCE_API_KEY": "unit-air-test-value"},
        )
    assert captured.value.model_code is ModelErrorCode.DATA_POLICY_VIOLATION


def test_airforce_normalizes_missing_usage_and_preserves_gateway_identity() -> None:
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="gateway-response",
            model="exact-returned-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    adapter = AirforceModelAdapter(
        _config(),
        data_classification=DataClassification.SYNTHETIC,
        allow_third_party_gateway=True,
        client=client,
        environ={
            "AIRFORCE_API_KEY": "unit-air-test-value",
            "OPENAI_API_KEY": "unit-openai-must-not-leak",
        },
    )
    request = _request(adapter.config)
    adapter.start_conversation(request)
    response = adapter.generate_next_turn(request)

    assert captured["stream"] is False
    assert "models" not in captured
    assert "unit-openai-must-not-leak" not in repr(captured)
    assert response.returned_model == "exact-returned-model"
    assert response.provenance is ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED
    assert response.upstream_provider_verifiable is False
    assert response.usage.total_tokens is None


def test_airforce_client_uses_only_fixed_base_url_and_airforce_key() -> None:
    constructor_arguments: dict = {}

    class FakeSDK:
        @staticmethod
        def OpenAI(**kwargs):
            constructor_arguments.update(kwargs)
            return SimpleNamespace()

    adapter = AirforceModelAdapter(
        _config(),
        data_classification=DataClassification.PUBLIC,
        allow_third_party_gateway=True,
        sdk_module=FakeSDK,
        environ={
            "AIRFORCE_API_KEY": "unit-air-only",
            "OPENAI_API_KEY": "unit-official-never-send",
        },
    )
    adapter.start_conversation(_request(adapter.config))
    assert constructor_arguments["base_url"] == AIRFORCE_BASE_URL
    assert constructor_arguments["api_key"] == "unit-air-only"
    assert "unit-official-never-send" not in repr(constructor_arguments)


@pytest.mark.parametrize(
    ("status_code", "expected", "retryable"),
    [
        (401, ModelErrorCode.AUTHENTICATION_FAILED, False),
        (403, ModelErrorCode.PERMISSION_DENIED, False),
        (429, ModelErrorCode.RATE_LIMITED, True),
        (500, ModelErrorCode.PROVIDER_UNAVAILABLE, True),
    ],
)
def test_airforce_normalizes_gateway_http_errors(
    status_code: int, expected: ModelErrorCode, retryable: bool
) -> None:
    class GatewayError(Exception):
        pass

    error = GatewayError("must not be exposed")
    error.status_code = status_code
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(error))
        )
    )
    adapter = AirforceModelAdapter(
        _config(),
        data_classification=DataClassification.SYNTHETIC,
        allow_third_party_gateway=True,
        client=client,
        environ={"AIRFORCE_API_KEY": "unit-air-test-value"},
    )
    request = _request(adapter.config)
    adapter.start_conversation(request)
    with pytest.raises(ModelError) as captured:
        adapter.generate_next_turn(request)
    assert captured.value.model_code is expected
    assert captured.value.retryable is retryable


def test_airforce_rejects_malformed_compatibility_response() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(choices=[]))
        )
    )
    adapter = AirforceModelAdapter(
        _config(),
        data_classification=DataClassification.SYNTHETIC,
        allow_third_party_gateway=True,
        client=client,
        environ={"AIRFORCE_API_KEY": "unit-air-test-value"},
    )
    request = _request(adapter.config)
    adapter.start_conversation(request)
    with pytest.raises(ModelError) as captured:
        adapter.generate_next_turn(request)
    assert captured.value.model_code is ModelErrorCode.INVALID_PROVIDER_RESPONSE
