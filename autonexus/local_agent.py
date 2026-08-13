"""Paired loopback agent for explicit local CPU/GPU training."""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path

from .web import create_app, default_workspace
from .web_auth import AgentAuthenticator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autonexus-agent",
        description=(
            "Run a paired local AutoNexus worker. Every browser mission must "
            "include explicit local-compute consent."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument(
        "--workspace",
        default=str(default_workspace() / "local-agent"),
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Trusted Vercel origin; repeat for preview and production URLs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            'Local agent requires: pip install "AutoNexus[serve]"'
        ) from exc
    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The local agent may only bind to a loopback address.")
    origins = [origin.strip().rstrip("/") for origin in args.allow_origin if origin.strip()]
    if not origins:
        origins = [
            origin.strip().rstrip("/")
            for origin in os.getenv("AUTONEXUS_AGENT_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        ]
    token = secrets.token_urlsafe(32)
    workspace = Path(args.workspace).expanduser().resolve()
    print("\nAUTO NEXUS LOCAL AGENT")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print(f"Workspace: {workspace}")
    print(f"Pairing token: {token}")
    print("GPU permission: required again in the Studio for every mission.")
    print("Keep this terminal open. Press Ctrl+C to revoke the pairing token.\n")
    uvicorn.run(
        create_app(
            workspace=workspace,
            authenticator=AgentAuthenticator(token),
            cors_origins=origins,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
