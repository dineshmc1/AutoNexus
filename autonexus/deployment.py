"""Safe local background deployment for fitted AutoNexus models."""

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DeploymentHandle:
    url: str
    predict_url: str
    health_url: str
    host: str
    port: int
    _server: Any = field(repr=False)
    _thread: threading.Thread = field(repr=False)

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and not self._server.should_exit

    def stop(self, timeout: float = 10.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout)

    def wait(self) -> None:
        self._thread.join()

    def __enter__(self) -> "DeploymentHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


def create_inference_app(model: Any, *, api_key: str | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:
        raise RuntimeError(
            "Deployment requires: pip install AutoNexus[serve]"
        ) from exc

    app = FastAPI(title="AutoNexus Inference", version="1")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model.best_model,
            "problem_type": model.problem_type,
        }

    @app.post("/predict")
    async def predict(request: Request):
        if api_key:
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {api_key}"
            if not secrets.compare_digest(supplied, expected):
                raise HTTPException(status_code=401, detail="Invalid API key")
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records:
            raise HTTPException(status_code=422, detail="records cannot be empty")
        if not all(isinstance(record, dict) for record in records):
            raise HTTPException(status_code=422, detail="records must contain objects")
        frame = pd.DataFrame(records)
        response = {"predictions": model.predict(frame).tolist()}
        if bool(payload.get("include_probabilities", False)):
            try:
                response["probabilities"] = model.predict_proba(frame).tolist()
            except (AttributeError, TypeError):
                response["probabilities"] = None
        return response

    return app


def deploy_model(
    model: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
    allow_insecure_public: bool = False,
    startup_timeout: float = 10.0,
) -> DeploymentHandle:
    """Start a background Uvicorn server and return its lifecycle handle."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Deployment requires: pip install AutoNexus[serve]"
        ) from exc
    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be between 1 and 65535")
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in local_hosts:
        if not api_key:
            raise ValueError("Public binding requires an API key.")
        if not allow_insecure_public:
            raise ValueError(
                "Public HTTP binding requires allow_insecure_public=True; "
                "prefer TLS termination through a trusted reverse proxy."
            )

    app = create_inference_app(model, api_key=api_key)
    config = uvicorn.Config(
        app,
        host=host,
        port=int(port),
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        name=f"autonexus-deployment-{port}",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + startup_timeout
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=1.0)
        raise RuntimeError(
            f"AutoNexus deployment failed to start on {host}:{port}."
        )
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}"
    return DeploymentHandle(
        url=url,
        predict_url=f"{url}/predict",
        health_url=f"{url}/health",
        host=host,
        port=int(port),
        _server=server,
        _thread=thread,
    )
