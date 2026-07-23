"""Deterministic, lazy registry of model-adapter factories."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oneoxygen_sandbox.errors import ModelError
from oneoxygen_sandbox.model_adapters.base import ModelAdapter
from oneoxygen_sandbox.models import ModelErrorCode, ModelProvider, ModelRunConfig

ModelAdapterFactory = Callable[..., ModelAdapter]


@dataclass(frozen=True)
class ModelAdapterInfo:
    provider: ModelProvider
    optional_dependency: str | None
    dependency_available: bool


@dataclass(frozen=True)
class _Registration:
    factory: ModelAdapterFactory
    optional_dependency: str | None


class ModelAdapterRegistry:
    """Resolve factories without importing SDKs or contacting providers while listing."""

    def __init__(self) -> None:
        self._registrations: dict[ModelProvider, _Registration] = {}

    def register(
        self,
        provider: ModelProvider | str,
        factory: ModelAdapterFactory,
        *,
        optional_dependency: str | None = None,
    ) -> None:
        normalized = self._provider(provider)
        if normalized in self._registrations:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                f"duplicate model adapter provider registered: {normalized.value}",
            )
        self._registrations[normalized] = _Registration(factory, optional_dependency)

    def list_providers(self) -> tuple[ModelProvider, ...]:
        return tuple(sorted(self._registrations, key=lambda provider: provider.value))

    def list_configured_providers(self) -> tuple[ModelProvider, ...]:
        return self.list_providers()

    def descriptions(self) -> tuple[ModelAdapterInfo, ...]:
        return tuple(
            ModelAdapterInfo(
                provider=provider,
                optional_dependency=registration.optional_dependency,
                dependency_available=self._dependency_available(registration.optional_dependency),
            )
            for provider, registration in sorted(
                self._registrations.items(), key=lambda item: item[0].value
            )
        )

    def resolve(self, provider: ModelProvider | str) -> ModelAdapterFactory:
        normalized = self._provider(provider)
        try:
            return self._registrations[normalized].factory
        except KeyError as exc:
            raise ModelError(
                ModelErrorCode.PROVIDER_NOT_CONFIGURED,
                f"model provider is not configured: {normalized.value}",
            ) from exc

    def create(
        self,
        provider: ModelProvider | str,
        config: ModelRunConfig,
        **kwargs: Any,
    ) -> ModelAdapter:
        normalized = self._provider(provider)
        if config.provider is not normalized:
            raise ModelError(
                ModelErrorCode.INVALID_REQUEST,
                "model configuration provider does not match the requested adapter",
            )
        registration = self._registrations.get(normalized)
        if registration is None:
            raise ModelError(
                ModelErrorCode.PROVIDER_NOT_CONFIGURED,
                f"model provider is not configured: {normalized.value}",
            )
        if not self._dependency_available(registration.optional_dependency):
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                f"optional dependency is not installed for provider {normalized.value}",
            )
        try:
            adapter = registration.factory(config, **kwargs)
        except ModuleNotFoundError as exc:
            if registration.optional_dependency is None:
                raise
            raise ModelError(
                ModelErrorCode.MISSING_DEPENDENCY,
                f"optional dependency is not installed for provider {normalized.value}",
            ) from exc
        return adapter

    @staticmethod
    def _dependency_available(dependency: str | None) -> bool:
        return dependency is None or importlib.util.find_spec(dependency) is not None

    @staticmethod
    def _provider(provider: ModelProvider | str) -> ModelProvider:
        try:
            return ModelProvider(provider)
        except ValueError as exc:
            raise ModelError(
                ModelErrorCode.PROVIDER_NOT_CONFIGURED,
                "requested model provider is not supported",
            ) from exc


def _scripted_factory(config: ModelRunConfig, **kwargs: Any) -> ModelAdapter:
    from oneoxygen_sandbox.model_adapters.scripted import ScriptedModelAdapter

    return ScriptedModelAdapter(config, **kwargs)


def _openai_factory(config: ModelRunConfig, **kwargs: Any) -> ModelAdapter:
    from oneoxygen_sandbox.model_adapters.openai import OpenAIModelAdapter

    return OpenAIModelAdapter(config, **kwargs)


def _airforce_factory(config: ModelRunConfig, **kwargs: Any) -> ModelAdapter:
    from oneoxygen_sandbox.model_adapters.airforce import AirforceModelAdapter

    return AirforceModelAdapter(config, **kwargs)


def default_model_adapter_registry() -> ModelAdapterRegistry:
    registry = ModelAdapterRegistry()
    registry.register(ModelProvider.SCRIPTED, _scripted_factory)
    registry.register(ModelProvider.OPENAI, _openai_factory, optional_dependency="openai")
    registry.register(ModelProvider.AIRFORCE, _airforce_factory, optional_dependency="openai")
    return registry
