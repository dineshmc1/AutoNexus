"""Railway ASGI entry point for the hosted Auto Nexus control plane."""

from __future__ import annotations

from .web import create_app


app = create_app()
