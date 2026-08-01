"""AutoNexus: developer-friendly automated ML and monitoring framework."""

from .api import AutoNexus
from .callbacks import Callback, CallbackManager, Event
from .config import NexusConfig
from .data import (
    DataSource,
    FileSource,
    FrameSource,
    IterableSource,
    KafkaSource,
    SQLSource,
)
from .drift import DriftBaseline, DriftDetector, DriftReport, DriftSignal
from .llm import (
    CallableLLMProvider,
    HTTPJSONProvider,
    LLMProvider,
    LiteLLMProvider,
    OllamaProvider,
    TransformersProvider,
)
from .memory import FAISSMetaMemory
from .model import NexusModel, UpdatePolicy, UpdateResult
from .monitoring import (
    JSONLSink,
    LoggingSink,
    NexusMonitor,
    PrometheusSink,
    WebhookSink,
)
from .plugins import PluginRegistry, plugins
from .registry import ModelRegistry

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

__version__ = "0.1.0"
