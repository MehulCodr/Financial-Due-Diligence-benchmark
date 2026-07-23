"""Experimental Api.Airforce OpenAI-compatible direct gateway adapter."""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Final, Never

from oneoxygen_sandbox.errors import ModelError
from oneoxygen_sandbox.models import (
    DataClassification,
    InferenceTransport,
    ModelCapabilities,
    ModelErrorCode,
    ModelProvider,
    ModelRunConfig,
    ModelTurnRequest,
    ModelTurnResponse,
    ModelUsage,
    NormalizedFinishReason,
    ProvenanceClassification,
    ToolCall,
    ToolResult,
    ToolSchemaMode,
)

AIRFORCE_BASE_URL: Final = "https://api.airforce/v1"
AIRFORCE_API_HOST: Final = "api.airforce"
_SDK_UNSET: Final = object()


class AirforceModelAdapter:
    """Use Api.Airforce as an explicitly unverified third-party gateway."""

    _capabilities = ModelCapabilities(
        tool_calling=True,
        multiple_tool_calls_per_turn=True,
        temperature_support=True,
    )

    def __init__(
        self,
        config: ModelRunConfig,
        *,
        data_classification: DataClassification | str | None = None,
        allow_third_party_gateway: bool = False,
        client: Any | None = None,
        sdk_module: Any = _SDK_UNSET,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = self.validate_config(config)
        self._classification = (
            DataClassification(data_classification) if data_classification is not None else None
        )
        if not allow_third_party_gateway:
            raise ModelError(
                ModelErrorCode.DATA_POLICY_VIOLATION,
                "Api.Airforce requires explicit third-party gateway acknowledgement.",
            )
        if self._classification not in {
            DataClassification.SYNTHETIC,
            DataClassification.PUBLIC,
        }:
            raise ModelError(
                ModelErrorCode.DATA_POLICY_VIOLATION,
                "Api.Airforce accepts only explicitly synthetic or public tasks.",
            )
        self._client = client
        self._sdk_module = sdk_module
        self._environ = os.environ if environ is None else environ
        self._clock = clock
        self._messages: list[dict[str, Any]] = []
        self._started = False
        self._closed = False
        self._last_completed_turn = 0

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.AIRFORCE

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def validate_config(self, config: ModelRunConfig) -> ModelRunConfig:
        if config.provider is not ModelProvider.AIRFORCE:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "AirforceModelAdapter requires provider 'airforce'.",
            )
        if (
            config.transport is not InferenceTransport.GATEWAY_DIRECT
            or config.provenance is not ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED
            or config.api_host != AIRFORCE_API_HOST
        ):
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "Api.Airforce must retain the fixed unverified gateway route.",
            )
        if config.provider_settings:
            raise ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "Api.Airforce provider settings are not accepted.",
            )
        if config.tool_schema_mode is not ToolSchemaMode.PORTABLE:
            raise ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "Api.Airforce supports only portable tool schemas.",
            )
        if config.store_provider_response:
            raise ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "Api.Airforce response storage is not enabled.",
            )
        return config

    def start_conversation(self, request: ModelTurnRequest) -> None:
        if self._closed or self._started:
            raise ModelError(
                ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                "the Api.Airforce conversation cannot be started",
            )
        self._validate_request(request)
        if request.turn_number != 1 or request.tool_results:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "the first Api.Airforce request must be turn 1 without tool results",
            )
        self._ensure_client()
        self._messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.initial_task_instruction},
        ]
        self._started = True

    def generate_next_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        if not self._started or self._closed:
            raise ModelError(
                ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                "the Api.Airforce conversation is not active",
            )
        self._validate_request(request)
        if request.turn_number != self._last_completed_turn + 1:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "Api.Airforce turns must be generated sequentially",
            )
        pending = [self._tool_result_message(result) for result in request.tool_results]
        messages = [*deepcopy(self._messages), *pending]
        arguments: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": deepcopy(definition.arguments_schema),
                    },
                }
                for definition in request.tool_definitions
            ],
            "max_tokens": self.config.maximum_output_tokens,
            "stream": False,
            "timeout": (request.request_timeout_seconds or self.config.model_call_timeout_seconds),
        }
        if self.config.temperature is not None:
            arguments["temperature"] = self.config.temperature
        started = self._clock()
        try:
            raw = self._client.chat.completions.create(**arguments)
        except Exception as exc:
            raise self._normalize_error(exc) from None
        latency = max(0.0, self._clock() - started)
        normalized, assistant = self._normalize_response(raw, latency)
        self._messages.extend(pending)
        self._messages.append(assistant)
        self._last_completed_turn = request.turn_number
        return normalized

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self._messages.clear()
        self._closed = True
        self._started = False

    def discover_models(self) -> tuple[dict[str, Any], ...]:
        """Return bounded discovery metadata from the authenticated live catalog."""
        self._ensure_client()
        try:
            response = self._client.models.list()
        except Exception as exc:
            raise self._normalize_error(exc) from None
        data = _read(response, "data")
        if not isinstance(data, (list, tuple)):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "Api.Airforce returned a malformed model catalog.",
            )
        rows: list[dict[str, Any]] = []
        for item in data:
            model_id = _read(item, "id")
            if not isinstance(model_id, str) or not model_id:
                continue
            rows.append(
                {
                    "id": model_id[:256],
                    "owned_by": _bounded(_read(item, "owned_by")),
                    "supports_tools": _read(item, "supports_tools") is True,
                    "status": _bounded(_read(item, "status")),
                    "input_price_cents_per_million": _nonnegative_number(
                        _read(item, "pricepermilliontokens")
                    ),
                    "output_price_cents_per_million": _nonnegative_number(
                        _read(item, "output_pricepermilliontokens")
                    ),
                }
            )
        return tuple(rows)

    def _validate_request(self, request: ModelTurnRequest) -> None:
        if self.validate_config(request.run_config) != self.config:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "model configuration cannot change during an Api.Airforce run",
            )

    def _ensure_client(self) -> None:
        api_key = self._environ.get("AIRFORCE_API_KEY", "").strip()
        if not api_key:
            raise ModelError(
                ModelErrorCode.MISSING_API_KEY,
                "AIRFORCE_API_KEY is required for the Api.Airforce gateway.",
            )
        if self._client is not None:
            return
        sdk = self._resolve_sdk()
        constructor = getattr(sdk, "OpenAI", None)
        if not callable(constructor):
            raise ModelError(
                ModelErrorCode.PROVIDER_NOT_CONFIGURED,
                "the installed OpenAI SDK does not expose the required client",
            )
        try:
            self._client = constructor(
                api_key=api_key,
                base_url=AIRFORCE_BASE_URL,
                timeout=self.config.model_call_timeout_seconds,
                max_retries=0,
            )
        except Exception as exc:
            raise self._normalize_error(exc) from None

    def _resolve_sdk(self) -> Any:
        if self._sdk_module is None:
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                "Api.Airforce support requires the 'openai' optional dependency.",
            )
        if self._sdk_module is not _SDK_UNSET:
            return self._sdk_module
        try:
            self._sdk_module = importlib.import_module("openai")
        except ImportError:
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                "Api.Airforce support requires the 'openai' optional dependency.",
            ) from None
        return self._sdk_module

    def _tool_result_message(self, result: ToolResult) -> dict[str, Any]:
        payload = {
            "success": result.success,
            "content": result.content,
            "error": (
                {
                    "code": result.error.code.value,
                    "message": result.error.message[:2_000],
                }
                if result.error is not None
                else None
            ),
        }
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }

    def _normalize_response(
        self, response: Any, latency: float
    ) -> tuple[ModelTurnResponse, dict[str, Any]]:
        choices = _read(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "Api.Airforce returned a malformed compatibility response.",
            )
        choice = choices[0]
        message = _read(choice, "message")
        if message is None:
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "Api.Airforce returned no assistant message.",
            )
        content = _read(message, "content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "Api.Airforce returned invalid assistant content.",
            )
        raw_calls = _read(message, "tool_calls") or []
        if not isinstance(raw_calls, (list, tuple)):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "Api.Airforce returned invalid tool calls.",
            )
        tool_calls: list[ToolCall] = []
        assistant_calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(raw_calls):
            function = _read(raw_call, "function")
            call_id = _read(raw_call, "id")
            name = _read(function, "name")
            arguments_text = _read(function, "arguments")
            if not all(isinstance(value, str) for value in (call_id, name, arguments_text)):
                raise ModelError(
                    ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                    "Api.Airforce returned an invalid function call.",
                )
            try:
                arguments = json.loads(
                    arguments_text,
                    object_pairs_hook=_object_without_duplicates,
                    parse_constant=_reject_constant,
                )
            except (ValueError, json.JSONDecodeError):
                raise ModelError(
                    ModelErrorCode.MALFORMED_TOOL_ARGUMENTS,
                    "Api.Airforce returned malformed function arguments.",
                ) from None
            if not isinstance(arguments, dict):
                raise ModelError(
                    ModelErrorCode.MALFORMED_TOOL_ARGUMENTS,
                    "Api.Airforce function arguments must be a JSON object.",
                )
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    tool_name=name,
                    arguments=arguments,
                    original_index=index,
                )
            )
            assistant_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_text},
                }
            )
        returned_model = _read(response, "model")
        if not isinstance(returned_model, str) or not returned_model:
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "Api.Airforce did not identify the returned gateway model.",
            )
        finish = _finish_reason(_read(choice, "finish_reason"), bool(tool_calls))
        usage = _read(response, "usage")
        normalized = ModelTurnResponse(
            response_id=_string_or_none(_read(response, "id")),
            provider=ModelProvider.AIRFORCE,
            requested_model=self.config.model,
            returned_model=returned_model,
            text=content,
            tool_calls=tuple(tool_calls),
            finish_reason=finish,
            usage=ModelUsage(
                input_tokens=_nonnegative_integer(_read(usage, "prompt_tokens")),
                output_tokens=_nonnegative_integer(_read(usage, "completion_tokens")),
                total_tokens=_nonnegative_integer(_read(usage, "total_tokens")),
            ),
            latency_seconds=latency,
            transport=InferenceTransport.GATEWAY_DIRECT,
            api_host=AIRFORCE_API_HOST,
            provenance=ProvenanceClassification.THIRD_PARTY_GATEWAY_UNVERIFIED,
            official_route=False,
            upstream_provider_verifiable=False,
            provider_metadata={"gateway_route": "upstream_unverified"},
        )
        return normalized, {
            "role": "assistant",
            "content": content,
            "tool_calls": assistant_calls,
        }

    def _normalize_error(self, exc: Exception) -> ModelError:
        status = _nonnegative_integer(getattr(exc, "status_code", None))
        metadata = {"status_code": status} if status is not None else {}
        name = type(exc).__name__
        if status == 401 or name == "AuthenticationError":
            code, message, retryable = (
                ModelErrorCode.AUTHENTICATION_FAILED,
                "Api.Airforce authentication failed.",
                False,
            )
        elif status == 403 or name == "PermissionDeniedError":
            code, message, retryable = (
                ModelErrorCode.PERMISSION_DENIED,
                "Api.Airforce denied permission for the request.",
                False,
            )
        elif status == 429 or name == "RateLimitError":
            code, message, retryable = (
                ModelErrorCode.RATE_LIMITED,
                "Api.Airforce rate-limited the request.",
                True,
            )
        elif status == 408 or name == "APITimeoutError" or isinstance(exc, TimeoutError):
            code, message, retryable = (
                ModelErrorCode.REQUEST_TIMEOUT,
                "The Api.Airforce request timed out.",
                True,
            )
        elif status is not None and status >= 500:
            code, message, retryable = (
                ModelErrorCode.PROVIDER_UNAVAILABLE,
                "Api.Airforce is temporarily unavailable.",
                True,
            )
        elif status in {400, 404, 422}:
            code, message, retryable = (
                ModelErrorCode.INVALID_REQUEST,
                "Api.Airforce rejected the request.",
                False,
            )
        else:
            code, message, retryable = (
                ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                "The Api.Airforce adapter encountered an unexpected gateway error.",
                False,
            )
        return ModelError(code, message, retryable=retryable, provider_metadata=metadata)


def _read(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _nonnegative_integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _nonnegative_number(value: Any) -> int | float | None:
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        else None
    )


def _bounded(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:256]


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _finish_reason(value: Any, has_calls: bool) -> NormalizedFinishReason:
    if has_calls:
        return NormalizedFinishReason.TOOL_CALLS
    if value in {"stop", "end_turn"}:
        return NormalizedFinishReason.COMPLETED
    if value in {"length", "max_tokens"}:
        return NormalizedFinishReason.LENGTH
    if value in {"content_filter", "safety"}:
        return NormalizedFinishReason.CONTENT_FILTER
    return NormalizedFinishReason.UNKNOWN


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    raise ValueError(f"invalid JSON constant: {value}")
