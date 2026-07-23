"""Optional OpenAI Responses API adapter.

The adapter deliberately keeps provider conversation state on the host and
replays response items locally.  It never enables OpenAI-hosted tools and it
never passes provider credentials to the sandbox.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from enum import Enum
from typing import Any, Final, Never

from oneoxygen_sandbox.errors import ModelError, sanitize_model_error_message
from oneoxygen_sandbox.models import (
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
    ToolDefinition,
    ToolResult,
    ToolSchemaMode,
)

_SDK_UNSET: Final = object()
_MAX_PROVIDER_TOOL_RESULT_BYTES: Final = 128 * 1024
_MAX_METADATA_STRING_LENGTH: Final = 256


class OpenAIModelAdapter:
    """Translate provider-neutral model turns to the OpenAI Responses API."""

    _capabilities = ModelCapabilities(
        tool_calling=True,
        multiple_tool_calls_per_turn=True,
        reasoning_token_reporting=True,
        cached_token_reporting=True,
        temperature_support=True,
        seed_support=False,
        strict_tool_schemas=False,
        response_storage_control=True,
    )

    def __init__(
        self,
        config: ModelRunConfig,
        *,
        client: Any | None = None,
        sdk_module: Any = _SDK_UNSET,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = self.validate_config(config)
        self._client = client
        self._sdk_module = sdk_module
        self._environ = os.environ if environ is None else environ
        self._clock = clock
        self._input_items: list[dict[str, Any]] = []
        self._started = False
        self._closed = False
        self._last_completed_turn = 0

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def validate_config(self, config: ModelRunConfig) -> ModelRunConfig:
        """Validate settings supported by the standardized OpenAI adapter."""

        if config.provider is not ModelProvider.OPENAI:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "OpenAIModelAdapter requires provider 'openai'.",
            )
        if config.provider_settings:
            raise ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "OpenAI provider settings are not supported in the standardized mode.",
            )
        if config.transport is not InferenceTransport.DIRECT:
            raise ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "OpenAIModelAdapter executes only direct requests; use OpenAIBatchBackend "
                "for provider_batch.",
            )
        if config.tool_schema_mode is not ToolSchemaMode.PORTABLE:
            raise ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "The OpenAI adapter supports only portable tool schemas in Phase 3A.",
            )
        return config

    def start_conversation(self, request: ModelTurnRequest) -> None:
        """Initialize a fresh local, stateless Responses conversation."""

        if self._closed:
            raise ModelError(
                ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                "The OpenAI adapter is closed.",
            )
        if self._started:
            raise ModelError(
                ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                "The OpenAI conversation has already started.",
            )
        self._validate_request_config(request)
        if request.turn_number != 1 or request.tool_results:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "The first OpenAI request must be turn 1 without tool results.",
            )
        self._ensure_client()
        self._input_items = [
            {
                "role": "user",
                "content": request.initial_task_instruction,
            }
        ]
        self._last_completed_turn = 0
        self._started = True

    def generate_next_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        """Generate one normalized turn without executing any returned tools."""

        if not self._started or self._closed:
            raise ModelError(
                ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                "The OpenAI conversation is not active.",
            )
        self._validate_request_config(request)
        expected_turn = self._last_completed_turn + 1
        if request.turn_number != expected_turn:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "Model turns must be generated sequentially.",
            )

        pending_results = [self._translate_tool_result(result) for result in request.tool_results]
        request_input = deepcopy([*self._input_items, *pending_results])
        kwargs = compile_openai_responses_request(self.config, request, request_input)

        started_at = self._clock()
        try:
            response = self._client.responses.create(**kwargs)
        except ModelError:
            raise
        except Exception as exc:
            raise self._normalize_sdk_error(exc) from None
        latency_seconds = max(0.0, self._clock() - started_at)

        try:
            normalized, serialized_output = parse_openai_responses_response(
                self.config,
                response,
                latency_seconds=latency_seconds,
            )
        except ModelError:
            raise
        except Exception:
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "OpenAI returned a response that could not be normalized.",
            ) from None

        # Mutate conversation state only after a complete provider response has
        # been validated. This makes a centrally retried failed attempt safe.
        self._input_items.extend(pending_results)
        self._input_items.extend(serialized_output)
        self._last_completed_turn = request.turn_number
        return normalized

    def close(self) -> None:
        """Close the SDK client and discard private provider conversation state."""

        if self._closed:
            return
        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        finally:
            self._input_items.clear()
            self._closed = True
            self._started = False

    def export_conversation_state(self) -> dict[str, Any]:
        """Return only serializable provider state for durable turn resumption."""
        return {
            "schema_version": 1,
            "input_items": deepcopy(self._input_items),
            "last_completed_turn": self._last_completed_turn,
            "started": self._started,
        }

    def restore_conversation_state(self, state: Mapping[str, Any]) -> None:
        """Restore validated state without serializing an SDK client or credentials."""
        if self._closed or self._started:
            raise ModelError(
                ModelErrorCode.INTERNAL_ADAPTER_ERROR,
                "OpenAI conversation state can only be restored into a fresh adapter.",
            )
        if state.get("schema_version") != 1:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "unsupported OpenAI conversation-state schema",
            )
        input_items = _json_safe(state.get("input_items"))
        turn = state.get("last_completed_turn")
        if (
            not isinstance(input_items, list)
            or isinstance(turn, bool)
            or not isinstance(turn, int)
            or turn < 0
        ):
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "invalid OpenAI conversation state",
            )
        self._ensure_client()
        self._input_items = deepcopy(input_items)
        self._last_completed_turn = turn
        self._started = bool(state.get("started", True))

    def _validate_request_config(self, request: ModelTurnRequest) -> None:
        effective = self.validate_config(request.run_config)
        if effective != self.config:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "Model configuration cannot change after adapter creation.",
            )
        if (
            request.request_timeout_seconds is not None
            and request.request_timeout_seconds > self.config.model_call_timeout_seconds
        ):
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "The effective request timeout may not exceed the configured model-call timeout.",
            )

    def _ensure_client(self) -> None:
        api_key = self._environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ModelError(
                ModelErrorCode.MISSING_API_KEY,
                "OPENAI_API_KEY is required for the OpenAI provider.",
            )
        if self._client is not None:
            return

        sdk = self._resolve_sdk()
        constructor = getattr(sdk, "OpenAI", None)
        if not callable(constructor):
            raise ModelError(
                ModelErrorCode.PROVIDER_NOT_CONFIGURED,
                "The installed OpenAI SDK does not expose the required client.",
            )
        try:
            self._client = constructor(
                api_key=api_key,
                timeout=self.config.model_call_timeout_seconds,
                max_retries=0,
            )
        except Exception as exc:
            raise self._normalize_sdk_error(exc) from None

    def _resolve_sdk(self) -> Any:
        if self._sdk_module is None:
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                "OpenAI support is not installed; install the 'openai' optional dependency.",
            )
        if self._sdk_module is not _SDK_UNSET:
            return self._sdk_module
        try:
            self._sdk_module = importlib.import_module("openai")
        except ImportError:
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                "OpenAI support is not installed; install the 'openai' optional dependency.",
            ) from None
        return self._sdk_module

    def _request_arguments(
        self,
        request: ModelTurnRequest,
        request_input: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return compile_openai_responses_request(self.config, request, request_input)

    def _translate_tool_definition(self, definition: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "name": definition.name,
            "description": definition.description,
            "parameters": deepcopy(definition.arguments_schema),
            "strict": False,
        }

    def _translate_tool_result(self, result: ToolResult) -> dict[str, Any]:
        return translate_openai_tool_result(result)

    def _normalize_response(
        self,
        response: Any,
        *,
        latency_seconds: float,
    ) -> tuple[ModelTurnResponse, list[dict[str, Any]]]:
        raw_output = _read(response, "output")
        if not isinstance(raw_output, (list, tuple)):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "OpenAI returned an invalid output collection.",
            )
        self._raise_for_failed_response(response)

        serialized_output = [_serialize_output_item(item) for item in raw_output]
        tool_calls: list[ToolCall] = []
        refused = False
        for item in raw_output:
            item_type = _read(item, "type")
            if item_type == "function_call":
                tool_calls.append(self._normalize_tool_call(item, original_index=len(tool_calls)))
            elif item_type == "message":
                refused = refused or _message_contains_refusal(item)

        finish_reason = _finish_reason(response, bool(tool_calls), refused)
        warnings: tuple[str, ...] = ()
        if _read(response, "error") is not None:
            warnings = ("OpenAI returned a failed response.",)

        provider_metadata: dict[str, Any] = {}
        status = _bounded_metadata_value(_read(response, "status"))
        if status is not None:
            provider_metadata["status"] = status
            provider_metadata["original_finish_reason"] = status
        incomplete_reason = _bounded_metadata_value(
            _read(_read(response, "incomplete_details"), "reason")
        )
        if incomplete_reason is not None:
            provider_metadata["incomplete_reason"] = incomplete_reason
            provider_metadata["original_finish_reason"] = incomplete_reason
        service_tier = _bounded_metadata_value(_read(response, "service_tier"))
        if service_tier is not None:
            provider_metadata["service_tier"] = service_tier
        provider_error_code = _bounded_metadata_value(_read(_read(response, "error"), "code"))
        if provider_error_code is not None:
            provider_metadata["provider_error_code"] = provider_error_code

        response_id = _read(response, "id")
        returned_model = _read(response, "model")
        if response_id is not None and not isinstance(response_id, str):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "OpenAI returned an invalid response identifier.",
            )
        if returned_model is not None and not isinstance(returned_model, str):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "OpenAI returned an invalid model identifier.",
            )

        return (
            ModelTurnResponse(
                response_id=response_id,
                provider=ModelProvider.OPENAI,
                requested_model=self.config.model,
                returned_model=returned_model,
                text=_response_text(response, raw_output),
                tool_calls=tuple(tool_calls),
                finish_reason=finish_reason,
                usage=_normalize_usage(_read(response, "usage")),
                latency_seconds=latency_seconds,
                attempts=(),
                attempt_count=1,
                warnings=warnings,
                provider_metadata=provider_metadata,
                transport=self.config.transport,
                api_host=self.config.api_host,
                provenance=ProvenanceClassification.OFFICIAL_PROVIDER,
                official_route=True,
                upstream_provider_verifiable=True,
            ),
            serialized_output,
        )

    def _raise_for_failed_response(self, response: Any) -> None:
        status = _read(response, "status")
        error = _read(response, "error")
        if status != "failed" and error is None:
            return
        code = _read(error, "code")
        if code in {"content_filter", "safety"}:
            return
        metadata: dict[str, Any] = {}
        bounded_code = _bounded_metadata_value(code)
        if bounded_code is not None:
            metadata["provider_error_code"] = bounded_code
        response_id = _bounded_metadata_value(_read(response, "id"))
        if response_id is not None:
            metadata["request_id"] = response_id

        if _is_context_limit_error(code):
            raise ModelError(
                ModelErrorCode.CONTEXT_LIMIT_EXCEEDED,
                "The OpenAI response exceeded the model context limit.",
                provider_metadata=metadata,
            )
        if code in {"rate_limit_exceeded", "rate_limited"}:
            raise ModelError(
                ModelErrorCode.RATE_LIMITED,
                "OpenAI rate-limited the request.",
                retryable=True,
                provider_metadata=metadata,
            )
        if code in {"server_error", "provider_unavailable"}:
            raise ModelError(
                ModelErrorCode.PROVIDER_UNAVAILABLE,
                "OpenAI is temporarily unavailable.",
                retryable=True,
                provider_metadata=metadata,
            )
        if code in {"authentication_error", "invalid_api_key"}:
            raise ModelError(
                ModelErrorCode.AUTHENTICATION_FAILED,
                "OpenAI authentication failed.",
                provider_metadata=metadata,
            )
        if code in {"permission_denied", "insufficient_quota"}:
            raise ModelError(
                ModelErrorCode.PERMISSION_DENIED,
                "OpenAI denied permission for the request.",
                provider_metadata=metadata,
            )
        if _is_unsupported_parameter(code, _read(error, "param")):
            raise ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "OpenAI does not support a requested parameter.",
                provider_metadata=metadata,
            )
        if code in {"invalid_request", "invalid_prompt"}:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "OpenAI rejected the request.",
                provider_metadata=metadata,
            )
        raise ModelError(
            ModelErrorCode.INVALID_PROVIDER_RESPONSE,
            "OpenAI returned a failed response.",
            provider_metadata=metadata,
        )

    def _normalize_tool_call(self, item: Any, *, original_index: int) -> ToolCall:
        call_id = _read(item, "call_id")
        name = _read(item, "name")
        raw_arguments = _read(item, "arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "OpenAI returned an invalid function call.",
            )
        if not isinstance(raw_arguments, str):
            raise ModelError(
                ModelErrorCode.MALFORMED_TOOL_ARGUMENTS,
                "OpenAI returned function arguments that were not JSON text.",
            )
        try:
            arguments = json.loads(
                raw_arguments,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise ModelError(
                ModelErrorCode.MALFORMED_TOOL_ARGUMENTS,
                "OpenAI returned malformed JSON function arguments.",
            ) from None
        if not isinstance(arguments, dict):
            raise ModelError(
                ModelErrorCode.MALFORMED_TOOL_ARGUMENTS,
                "OpenAI function arguments must decode to a JSON object.",
            )
        try:
            return ToolCall(
                call_id=call_id,
                tool_name=name,
                arguments=arguments,
                original_index=original_index,
            )
        except Exception:
            raise ModelError(
                ModelErrorCode.INVALID_PROVIDER_RESPONSE,
                "OpenAI returned an invalid function call.",
            ) from None

    def _normalize_sdk_error(self, exc: Exception) -> ModelError:
        name = type(exc).__name__
        status_code = _integer_or_none(getattr(exc, "status_code", None))
        error_code = _provider_error_field(exc, "code")
        parameter = _provider_error_field(exc, "param")
        metadata: dict[str, Any] = {}
        if status_code is not None:
            metadata["status_code"] = status_code
        request_id = _bounded_metadata_value(getattr(exc, "request_id", None))
        if request_id is not None:
            metadata["request_id"] = request_id
        bounded_code = _bounded_metadata_value(error_code)
        if bounded_code is not None:
            metadata["provider_error_code"] = bounded_code
        bounded_parameter = _bounded_metadata_value(parameter)
        if bounded_parameter is not None:
            metadata["parameter"] = bounded_parameter

        if name == "AuthenticationError" or status_code == 401:
            return ModelError(
                ModelErrorCode.AUTHENTICATION_FAILED,
                "OpenAI authentication failed.",
                provider_metadata=metadata,
            )
        if name == "PermissionDeniedError" or status_code == 403:
            return ModelError(
                ModelErrorCode.PERMISSION_DENIED,
                "OpenAI denied permission for the request.",
                provider_metadata=metadata,
            )
        if name == "RateLimitError" or status_code == 429:
            return ModelError(
                ModelErrorCode.RATE_LIMITED,
                "OpenAI rate-limited the request.",
                retryable=True,
                provider_metadata=metadata,
            )
        if name == "APITimeoutError" or isinstance(exc, TimeoutError) or status_code == 408:
            return ModelError(
                ModelErrorCode.REQUEST_TIMEOUT,
                "The OpenAI request timed out.",
                retryable=True,
                provider_metadata=metadata,
            )
        if _is_context_limit_error(error_code):
            return ModelError(
                ModelErrorCode.CONTEXT_LIMIT_EXCEEDED,
                "The OpenAI request exceeded the model context limit.",
                provider_metadata=metadata,
            )
        if _is_unsupported_parameter(error_code, parameter):
            return ModelError(
                ModelErrorCode.UNSUPPORTED_PARAMETER,
                "OpenAI does not support a requested parameter.",
                provider_metadata=metadata,
            )
        if name in {"BadRequestError", "UnprocessableEntityError"} or status_code in {
            400,
            422,
        }:
            return ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "OpenAI rejected the request.",
                provider_metadata=metadata,
            )
        if name in {"APIConnectionError", "InternalServerError"} or (
            status_code is not None and (status_code == 409 or status_code >= 500)
        ):
            return ModelError(
                ModelErrorCode.PROVIDER_UNAVAILABLE,
                "OpenAI is temporarily unavailable.",
                retryable=True,
                provider_metadata=metadata,
            )
        return ModelError(
            ModelErrorCode.INTERNAL_ADAPTER_ERROR,
            "The OpenAI adapter encountered an unexpected provider error.",
            provider_metadata=metadata,
        )


def compile_openai_responses_request(
    config: ModelRunConfig,
    request: ModelTurnRequest,
    request_input: list[dict[str, Any]],
    *,
    include_client_timeout: bool = True,
) -> dict[str, Any]:
    """Compile the canonical Responses body used by direct and Batch transports."""
    if config.provider is not ModelProvider.OPENAI:
        raise ModelError(
            ModelErrorCode.INVALID_REQUEST,
            "the OpenAI Responses compiler requires provider 'openai'",
        )
    arguments: dict[str, Any] = {
        "model": config.model,
        "instructions": request.system_prompt,
        "input": deepcopy(request_input),
        "tools": [
            {
                "type": "function",
                "name": definition.name,
                "description": definition.description,
                "parameters": deepcopy(definition.arguments_schema),
                "strict": False,
            }
            for definition in request.tool_definitions
        ],
        "max_output_tokens": config.maximum_output_tokens,
        "store": config.store_provider_response,
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
    }
    if config.temperature is not None:
        arguments["temperature"] = config.temperature
    if include_client_timeout:
        arguments["timeout"] = request.request_timeout_seconds or config.model_call_timeout_seconds
    return arguments


def compile_openai_batch_turn(
    config: ModelRunConfig,
    request: ModelTurnRequest,
    conversation_state: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile one batch turn and return the pending serializable state."""
    raw_state = dict(conversation_state or {})
    if raw_state and raw_state.get("schema_version") != 1:
        raise ModelError(
            ModelErrorCode.INVALID_REQUEST,
            "unsupported OpenAI conversation-state schema",
        )
    prior = _json_safe(raw_state.get("input_items", []))
    if not isinstance(prior, list):
        raise ModelError(
            ModelErrorCode.INVALID_REQUEST,
            "invalid OpenAI conversation-state input items",
        )
    if request.turn_number == 1 and not prior:
        prior = [{"role": "user", "content": request.initial_task_instruction}]
    pending_results = [translate_openai_tool_result(result) for result in request.tool_results]
    request_input = [*deepcopy(prior), *pending_results]
    body = compile_openai_responses_request(
        config,
        request,
        request_input,
        include_client_timeout=False,
    )
    return body, {
        "schema_version": 1,
        "input_items": request_input,
        "last_completed_turn": request.turn_number - 1,
        "started": True,
    }


def apply_openai_batch_response_state(
    pending_state: Mapping[str, Any],
    serialized_output: list[dict[str, Any]],
    turn_number: int,
) -> dict[str, Any]:
    """Advance serializable OpenAI state only after a validated batch result."""
    prior = _json_safe(pending_state.get("input_items"))
    output = _json_safe(serialized_output)
    if not isinstance(prior, list) or not isinstance(output, list):
        raise ModelError(
            ModelErrorCode.INVALID_PROVIDER_RESPONSE,
            "invalid OpenAI batch conversation state",
        )
    return {
        "schema_version": 1,
        "input_items": [*prior, *output],
        "last_completed_turn": turn_number,
        "started": True,
    }


def translate_openai_tool_result(result: ToolResult) -> dict[str, Any]:
    """Translate one normalized tool result for both direct and batch continuation."""
    error: dict[str, Any] | None = None
    if result.error is not None:
        error = {
            "code": _enum_value(result.error.code),
            "message": sanitize_model_error_message(result.error.message)[:2_000],
        }
    payload: dict[str, Any] = {
        "success": result.success,
        "content": _json_safe(result.content),
        "error": error,
    }
    encoded = _compact_json(payload)
    if len(encoded.encode("utf-8")) > _MAX_PROVIDER_TOOL_RESULT_BYTES:
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        payload = {
            "success": result.success,
            "content": {
                "truncated": True,
                "original_content_sha256": digest,
            },
            "error": error,
        }
        encoded = _compact_json(payload)
    return {
        "type": "function_call_output",
        "call_id": result.call_id,
        "output": encoded,
    }


def parse_openai_responses_response(
    config: ModelRunConfig,
    response: Any,
    *,
    latency_seconds: float = 0.0,
) -> tuple[ModelTurnResponse, list[dict[str, Any]]]:
    """Normalize a Responses result identically for direct and Batch transports."""
    parser = object.__new__(OpenAIModelAdapter)
    parser.config = config
    return parser._normalize_response(response, latency_seconds=latency_seconds)


def _read(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _serialize_output_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        dumped = item.model_dump(mode="json")
    elif isinstance(item, Mapping):
        dumped = dict(item)
    elif hasattr(item, "__dict__"):
        dumped = {key: value for key, value in vars(item).items() if not key.startswith("_")}
    else:
        raise ModelError(
            ModelErrorCode.INVALID_PROVIDER_RESPONSE,
            "OpenAI returned an output item that could not be serialized.",
        )
    serialized = _json_safe(dumped)
    if not isinstance(serialized, dict) or not isinstance(serialized.get("type"), str):
        raise ModelError(
            ModelErrorCode.INVALID_PROVIDER_RESPONSE,
            "OpenAI returned an invalid output item.",
        )
    return serialized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "<non-finite>"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return {
            key: _json_safe(item) for key, item in vars(value).items() if not key.startswith("_")
        }
    return "<unsupported>"


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"invalid JSON constant: {value}")


def _enum_value(value: Any) -> str:
    raw = value.value if isinstance(value, Enum) else value
    return str(raw)


def _response_text(response: Any, output: list[Any] | tuple[Any, ...]) -> str:
    helper_text = _read(response, "output_text")
    if isinstance(helper_text, str) and helper_text:
        return helper_text
    parts: list[str] = []
    for item in output:
        if _read(item, "type") != "message":
            continue
        content = _read(item, "content")
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            part_type = _read(part, "type")
            if part_type == "output_text":
                text = _read(part, "text")
                if isinstance(text, str):
                    parts.append(text)
            elif part_type == "refusal":
                refusal = _read(part, "refusal")
                if isinstance(refusal, str):
                    parts.append(refusal)
    return "".join(parts)


def _message_contains_refusal(item: Any) -> bool:
    content = _read(item, "content")
    return isinstance(content, (list, tuple)) and any(
        _read(part, "type") == "refusal" for part in content
    )


def _finish_reason(
    response: Any,
    has_tool_calls: bool,
    refused: bool,
) -> NormalizedFinishReason:
    if refused:
        return NormalizedFinishReason.REFUSED
    status = _read(response, "status")
    incomplete_reason = _read(_read(response, "incomplete_details"), "reason")
    provider_error_code = _read(_read(response, "error"), "code")
    if incomplete_reason in {"max_output_tokens", "length"}:
        return NormalizedFinishReason.LENGTH
    if incomplete_reason in {"content_filter", "safety"} or provider_error_code in {
        "content_filter",
        "safety",
    }:
        return NormalizedFinishReason.CONTENT_FILTER
    if status == "incomplete":
        return NormalizedFinishReason.PROVIDER_ERROR
    if status in {"cancelled", "canceled"}:
        return NormalizedFinishReason.CANCELLED
    if status == "failed" or _read(response, "error") is not None:
        return NormalizedFinishReason.PROVIDER_ERROR
    if has_tool_calls:
        return NormalizedFinishReason.TOOL_CALLS
    if status == "completed":
        return NormalizedFinishReason.COMPLETED
    return NormalizedFinishReason.UNKNOWN


def _normalize_usage(value: Any) -> ModelUsage:
    if value is None:
        return ModelUsage()
    input_details = _read(value, "input_tokens_details")
    output_details = _read(value, "output_tokens_details")
    return ModelUsage(
        input_tokens=_integer_or_none(_read(value, "input_tokens")),
        output_tokens=_integer_or_none(_read(value, "output_tokens")),
        reasoning_tokens=_integer_or_none(_read(output_details, "reasoning_tokens")),
        cached_input_tokens=_integer_or_none(_read(input_details, "cached_tokens")),
        total_tokens=_integer_or_none(_read(value, "total_tokens")),
        provider_metadata={},
    )


def _integer_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _bounded_metadata_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    return sanitize_model_error_message(value)[:_MAX_METADATA_STRING_LENGTH]


def _provider_error_field(exc: Exception, field: str) -> Any:
    direct = getattr(exc, field, None)
    if direct is not None:
        return direct
    body = getattr(exc, "body", None)
    direct_body_value = _read(body, field)
    if direct_body_value is not None:
        return direct_body_value
    error = _read(body, "error")
    return _read(error, field)


def _is_context_limit_error(value: Any) -> bool:
    return value in {
        "context_length_exceeded",
        "context_window_exceeded",
        "max_tokens_exceeded",
    }


def _is_unsupported_parameter(code: Any, parameter: Any) -> bool:
    if code in {"unsupported_parameter", "unsupported_value"}:
        return True
    return code == "invalid_parameter" and parameter in {
        "temperature",
        "seed",
        "store",
    }
