"""Provider-independent model adapters and deterministic registry."""

from oneoxygen_sandbox.model_adapters.airforce import AirforceModelAdapter
from oneoxygen_sandbox.model_adapters.base import ModelAdapter
from oneoxygen_sandbox.model_adapters.registry import (
    ModelAdapterInfo,
    ModelAdapterRegistry,
    default_model_adapter_registry,
)
from oneoxygen_sandbox.model_adapters.scripted import (
    ScriptedModelAdapter,
    ScriptedModelScript,
    ScriptedTurn,
    load_scripted_model_script,
)

__all__ = [
    "AirforceModelAdapter",
    "ModelAdapter",
    "ModelAdapterInfo",
    "ModelAdapterRegistry",
    "ScriptedModelAdapter",
    "ScriptedModelScript",
    "ScriptedTurn",
    "default_model_adapter_registry",
    "load_scripted_model_script",
]
