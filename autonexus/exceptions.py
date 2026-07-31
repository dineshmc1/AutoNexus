"""Public AutoNexus exception hierarchy."""


class AutoNexusError(Exception):
    """Base exception for framework-level failures."""


class ConfigurationError(AutoNexusError, ValueError):
    """Raised when a framework configuration is invalid."""


class CapabilityError(AutoNexusError):
    """Raised when a model or connector lacks a requested capability."""


class ArtifactError(AutoNexusError):
    """Raised when a run bundle is missing or incompatible."""


class UpdateRejected(AutoNexusError):
    """Raised when a candidate online update fails its promotion gate."""

