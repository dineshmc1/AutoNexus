"""Lifecycle callbacks for training, monitoring, and model updates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

LOGGER = logging.getLogger("autonexus.callbacks")


@dataclass(frozen=True)
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


class Callback:
    """Override ``handle`` or event-specific ``on_<event>`` methods."""

    def handle(self, event: Event) -> None:
        method = getattr(self, f"on_{event.name}", None)
        if method is not None:
            method(event)


class CallableCallback(Callback):
    def __init__(self, function: Callable[[Event], None]) -> None:
        self.function = function

    def handle(self, event: Event) -> None:
        self.function(event)


class CallbackManager:
    def __init__(self, callbacks: Iterable[Callback | Callable] = ()) -> None:
        self.callbacks = [
            callback
            if isinstance(callback, Callback)
            else CallableCallback(callback)
            for callback in callbacks
        ]

    def emit(self, name: str, **payload: Any) -> None:
        event = Event(name=name, payload=payload)
        for callback in self.callbacks:
            try:
                callback.handle(event)
            except Exception as exc:
                LOGGER.warning("Callback failed for %s: %s", name, exc)

