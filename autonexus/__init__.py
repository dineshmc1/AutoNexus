"""AutoNexus: developer-friendly automated ML and monitoring framework."""

import sys


# Windows resolves package paths case-insensitively. Register both supported
# spellings before loading submodules so Python cannot create duplicate classes.
_package = sys.modules[__name__]
sys.modules.setdefault("autonexus", _package)
sys.modules.setdefault("AutoNexus", _package)

from autonexus.api import AutoNexus
from autonexus.callbacks import Callback, CallbackManager, Event
from autonexus.config import NexusConfig
from autonexus.data import (
    DataSource,
    FileSource,
    FrameSource,
    IterableSource,
    KafkaSource,
    SQLSource,
)
from autonexus.drift import DriftBaseline, DriftDetector, DriftReport, DriftSignal
from autonexus.deployment import DeploymentHandle
from autonexus.llm import (
    CallableLLMProvider,
    HTTPJSONProvider,
    LLMProvider,
    LiteLLMProvider,
    OllamaProvider,
    TransformersProvider,
)
from autonexus.memory import FAISSMetaMemory
from autonexus.model import NexusModel, UpdatePolicy, UpdateResult
from autonexus.monitoring import (
    JSONLSink,
    LoggingSink,
    NexusMonitor,
    PrometheusSink,
    WebhookSink,
)
from autonexus.plugins import PluginRegistry, plugins
from autonexus.registry import ModelRegistry

__all__ = [
    "AutoNexus",
    "Callback",
    "CallbackManager",
    "CallableLLMProvider",
    "DataSource",
    "DriftBaseline",
    "DriftDetector",
    "DriftReport",
    "DriftSignal",
    "DeploymentHandle",
    "Event",
    "FAISSMetaMemory",
    "FileSource",
    "FrameSource",
    "HTTPJSONProvider",
    "IterableSource",
    "JSONLSink",
    "KafkaSource",
    "LLMProvider",
    "LiteLLMProvider",
    "LoggingSink",
    "ModelRegistry",
    "NexusConfig",
    "NexusModel",
    "NexusMonitor",
    "OllamaProvider",
    "PluginRegistry",
    "PrometheusSink",
    "SQLSource",
    "TransformersProvider",
    "UpdateResult",
    "UpdatePolicy",
    "WebhookSink",
    "plugins",
]

__version__ = "0.3.0"
