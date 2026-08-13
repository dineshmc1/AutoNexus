"""Web control-plane safety, persistence, and lifecycle tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from autonexus.exceptions import ConfigurationError
from autonexus.web import (
    ARTIFACT_PATHS,
    RunManager,
    _RedactingLLMProvider,
    _build_llm_provider,
    _finite_projection_matrix,
    _geometry_numeric_candidates,
    _run_insights,
    _safe_upload_path,
    _validate_config,
    _validate_llm_config,
    create_app,
    inspect_dataset,
)
from autonexus.web_auth import (
    AgentAuthenticator,
    AuthenticationError,
    FirebaseAuthenticator,
)
from autonexus.web_storage import FirebaseStorageMirror
from autonexus.llm import LLMProvider
from nexus_predictor import NexusPredictor


def test_upload_paths_preserve_classes_and_reject_traversal():
    assert _safe_upload_path("animals/cats/image 01.jpg") == Path(
        "animals/cats/image 01.jpg"
    )
    assert _safe_upload_path("animals\\dogs\\image.png") == Path(
        "animals/dogs/image.png"
    )

    with pytest.raises(ValueError, match="unsafe"):
        _safe_upload_path("../private.txt")


def test_tabular_inspection_finds_columns_and_target_candidate(tmp_path):
    dataset = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "temperature": [22.0, 24.5, 30.0],
            "humidity": [80, 74, 62],
            "target_aqi_class": [1, 1, 2],
        }
    ).to_csv(dataset, index=False)

    profile = inspect_dataset(dataset)

    assert profile["modality"] == "tabular"
    assert profile["columns"] == [
        "temperature",
        "humidity",
        "target_aqi_class",
    ]
    assert profile["target_candidates"][0] == "target_aqi_class"


def test_geometry_rejects_extreme_and_low_information_columns():
    frame = pd.DataFrame(
        {
            "safe": np.linspace(0, 1, 100),
            "constant": np.ones(100),
            "extreme": np.linspace(1e100, 2e100, 100),
            "excluded": np.arange(100),
        }
    )

    candidates, rejected = _geometry_numeric_candidates(
        frame, excluded={"excluded"}
    )

    assert candidates == ["safe"]
    assert rejected["constant"] == "fewer than five distinct values"
    assert rejected["extreme"] == "numerically extreme values"
    assert rejected["excluded"] == "excluded target or target proxy"


def test_projection_matrix_median_fills_nonfinite_values():
    matrix = _finite_projection_matrix(
        np.asarray([[1.0, np.nan], [3.0, np.inf], [5.0, 9.0]])
    )

    assert np.isfinite(matrix).all()
    assert matrix[:, 1].tolist() == [9.0, 9.0, 9.0]


def test_tree_geometry_uses_raw_model_ranked_safe_axes(tmp_path):
    rows = 160
    label = np.arange(rows) % 2
    frame = pd.DataFrame(
        {
            "signal": label + np.linspace(0, 0.1, rows),
            "support": np.sin(np.linspace(0, 12, rows)),
            "extreme": np.linspace(1e100, 2e100, rows),
            "target_label": label,
            "label": label,
        }
    )
    dataset = tmp_path / "geometry.csv"
    frame.to_csv(dataset, index=False)
    features = ["signal", "support", "target_label"]
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [("numeric", SimpleImputer(strategy="median"), features)]
                ),
            ),
            (
                "model",
                RandomForestClassifier(n_estimators=20, random_state=42),
            ),
        ]
    ).fit(frame[features], label)
    output = tmp_path / "artifacts"
    output.mkdir()
    joblib.dump(
        NexusPredictor(
            model=pipeline,
            problem_type="classification",
            target_name="label",
            feature_names=features,
            class_names=["0", "1"],
        ),
        output / "model.pkl",
    )
    state = {
        "id": "geometry-run",
        "dataset": str(dataset),
        "output_dir": str(output),
        "best_model": "rf",
        "summary": {},
        "config": {"target": "label", "test_size": 0.2},
        "dataset_profile": {
            "modality": "tabular",
            "columns": list(frame.columns),
            "name": dataset.name,
        },
        "artifacts": [],
    }

    geometry = _run_insights(state)["geometry"]

    assert geometry["error"] is None
    assert len(geometry["points"]) == rows
    assert {axis["label"] for axis in geometry["axes"]} == {
        "signal",
        "support",
        "model confidence",
    }
    assert "target_label" not in {
        geometry["response_surface"]["feature_x"],
        geometry["response_surface"]["feature_z"],
    }
    assert any("extreme" in note for note in geometry["notes"])
    assert {point["label"] for point in geometry["points"]} == {"0", "1"}
    assert all(
        0.0 <= float(point["confidence"]) <= 1.0
        for point in geometry["points"]
    )


def test_web_config_requires_a_real_tabular_target():
    with pytest.raises(ConfigurationError, match="target column"):
        _validate_config({"preset": "fast"}, "tabular")

    config, secrets_for_run = _validate_config(
        {
            "preset": "fast",
            "target": "label",
            "test_size": 0.25,
            "contribute_memory": False,
        },
        "tabular",
    )

    assert config["cv"] == 2
    assert config["target"] == "label"
    assert config["test_size"] == 0.25
    assert not config["contribute_memory"]
    assert not secrets_for_run


def test_byok_configuration_is_normalized_and_secret_is_separate():
    public, secrets_for_run = _validate_llm_config(
        {
            "mode": "byok",
            "provider": "openrouter",
            "model": "vendor/model-name",
            "api_key": "private-key-value",
        }
    )

    assert public == {
        "mode": "byok",
        "provider": "openrouter",
        "model": "openrouter/vendor/model-name",
        "api_base": None,
        "credential_storage": "ephemeral_memory",
    }
    assert secrets_for_run == {"api_key": "private-key-value"}
    assert "private-key-value" not in json.dumps(public)
    assert isinstance(
        _build_llm_provider(public, secrets_for_run),
        _RedactingLLMProvider,
    )

    with pytest.raises(ConfigurationError, match="API key"):
        _validate_llm_config(
            {
                "mode": "byok",
                "provider": "openai",
                "model": "model-name",
            }
        )


def test_byok_provider_redacts_key_from_failures():
    class FailingProvider(LLMProvider):
        def generate(self, prompt, *, context):
            del prompt, context
            raise RuntimeError("request rejected for private-key-value")

    provider = _RedactingLLMProvider(
        FailingProvider(),
        ["private-key-value"],
    )

    with pytest.raises(RuntimeError, match="REDACTED") as failure:
        provider.generate("report", context={})
    assert "private-key-value" not in str(failure.value)


class _FakeModel:
    def __init__(self, output_dir: Path):
        self.manifest_path = output_dir / "run.json"


class _FakeTrainer:
    def __init__(self, **options):
        self.output_dir = Path(options["output_dir"])

    def fit(self, dataset, *, target=None):
        assert Path(dataset).is_file()
        assert target == "label"
        self.output_dir.mkdir(parents=True)
        manifest = {
            "best_model": "logistic",
            "problem_type": "classification",
            "run_summary": {
                "training_accuracy": 0.96,
                "validation_accuracy": 0.93,
                "testing_accuracy": 0.94,
                "total_pipeline_seconds": 2.5,
            },
        }
        for name, relative in ARTIFACT_PATHS.items():
            path = self.output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(manifest) if name == "manifest" else name,
                encoding="utf-8",
            )
        return _FakeModel(self.output_dir)


def test_run_manager_persists_completion_and_allowlists_artifacts(tmp_path):
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"feature": [0, 1], "label": ["no", "yes"]}).to_csv(
        dataset,
        index=False,
    )
    manager = RunManager(
        tmp_path / "workspace",
        trainer_factory=_FakeTrainer,
    )
    run_id = manager.new_run_id()
    config, secrets_for_run = _validate_config(
        {"preset": "fast", "target": "label", "llm": False},
        "tabular",
    )

    manager.enqueue(
        run_id,
        dataset,
        config,
        inspect_dataset(dataset),
        secrets_for_run,
    )
    deadline = time.monotonic() + 5
    while manager.get(run_id)["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)

    run = manager.get(run_id)
    assert run["status"] == "completed"
    assert run["best_model"] == "logistic"
    assert run["summary"]["testing_accuracy"] == 0.94
    assert manager.artifact(run_id, "model").name == "model.pkl"
    with pytest.raises(FileNotFoundError):
        manager.artifact(run_id, "../../private")
    persisted = json.loads(
        (manager.run_dir(run_id) / "web_run.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "completed"
    assert (manager.root / "studio.sqlite3").is_file()
    manager.shutdown()

    recovered = RunManager(
        tmp_path / "workspace",
        trainer_factory=_FakeTrainer,
    )
    assert recovered.get(run_id)["best_model"] == "logistic"
    assert recovered.get(run_id)["storage"]["metadata"] == "sqlite"
    recovered.shutdown()


def test_run_manager_never_persists_byok_key(tmp_path):
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"feature": [0, 1], "label": ["no", "yes"]}).to_csv(
        dataset,
        index=False,
    )
    manager = RunManager(
        tmp_path / "workspace",
        trainer_factory=_FakeTrainer,
    )
    config, secrets_for_run = _validate_config(
        {
            "preset": "fast",
            "target": "label",
            "llm_config": {
                "mode": "byok",
                "provider": "openai",
                "model": "model-name",
                "api_key": "private-key-value",
            },
        },
        "tabular",
    )
    run_id = manager.new_run_id()

    state = manager.enqueue(
        run_id,
        dataset,
        config,
        inspect_dataset(dataset),
        secrets_for_run,
    )
    assert "private-key-value" not in json.dumps(state)
    deadline = time.monotonic() + 5
    while manager.get(run_id)["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)

    persisted = (manager.run_dir(run_id) / "web_run.json").read_text(
        encoding="utf-8"
    )
    assert "private-key-value" not in persisted
    assert run_id not in manager._secrets
    manager.shutdown()


def test_studio_http_contract_uploads_trains_and_downloads(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    manager = RunManager(
        tmp_path / "workspace",
        trainer_factory=_FakeTrainer,
    )
    app = create_app(manager=manager)
    config = {
        "preset": "fast",
        "target": "label",
        "llm_config": {
            "mode": "byok",
            "provider": "openai",
            "model": "model-name",
            "api_key": "http-private-key-value",
        },
        "contribute_memory": False,
    }
    csv_data = "feature,label\n0,no\n1,yes\n"

    with fastapi_testclient.TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["cache-control"].startswith("no-store")
        assert "LLM INTELLIGENCE / BYOK" in index.text
        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert asset.headers["cache-control"].startswith("no-store")
        assert "_rawX: Number(item.x)" in asset.text
        assert "formatGeometryAxisValue" in asset.text
        assert "confidence * 100" in asset.text
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "online"
        response = client.post(
            "/api/runs",
            data={"config": json.dumps(config)},
            files={"files": ("data.csv", csv_data, "text/csv")},
        )
        assert response.status_code == 202, response.text
        assert "http-private-key-value" not in response.text
        run_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while True:
            run = client.get(f"/api/runs/{run_id}").json()
            assert "http-private-key-value" not in json.dumps(run)
            if run["status"] not in {"queued", "running"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)

        assert run["status"] == "completed"
        model = client.get(f"/api/runs/{run_id}/artifacts/model")
        assert model.status_code == 200
        assert model.content == b"model"
        assert client.get(
            f"/api/runs/{run_id}/artifacts/private"
        ).status_code == 404
    manager.shutdown()


def test_firebase_auth_isolates_runs_and_disables_server_paths(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    def verify(token):
        identities = {
            "alice-token": {"uid": "alice", "email": "alice@example.com"},
            "bob-token": {"uid": "bob", "email": "bob@example.com"},
        }
        if token not in identities:
            raise ValueError("invalid token")
        return identities[token]

    auth = FirebaseAuthenticator(
        api_key="public-web-api-key",
        project_id="test-project",
        token_verifier=verify,
    )
    manager = RunManager(tmp_path / "workspace", trainer_factory=_FakeTrainer)
    app = create_app(manager=manager, authenticator=auth)
    alice = {"Authorization": "Bearer alice-token"}
    bob = {"Authorization": "Bearer bob-token"}
    config = {
        "preset": "fast",
        "target": "label",
        "llm": False,
        "contribute_memory": False,
    }

    with fastapi_testclient.TestClient(app) as client:
        assert client.get("/api/runs").status_code == 401
        auth_config = client.get("/api/auth/config").json()
        assert auth_config["required"]
        assert not auth_config["local_paths_allowed"]
        assert client.post(
            "/api/datasets/inspect",
            headers=alice,
            json={"path": str(tmp_path / "private.csv")},
        ).status_code == 403

        response = client.post(
            "/api/runs",
            headers=alice,
            data={"config": json.dumps(config)},
            files={
                "files": (
                    "data.csv",
                    "feature,label\n0,no\n1,yes\n",
                    "text/csv",
                )
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while client.get(f"/api/runs/{run_id}", headers=alice).json()[
            "status"
        ] in {"queued", "running"}:
            assert time.monotonic() < deadline
            time.sleep(0.02)

        assert len(client.get("/api/runs", headers=alice).json()["runs"]) == 1
        assert client.get("/api/runs", headers=bob).json()["runs"] == []
        assert client.get(f"/api/runs/{run_id}", headers=bob).status_code == 404
        assert client.get(
            f"/api/runs/{run_id}/artifacts/model", headers=bob
        ).status_code == 404
    manager.shutdown()


def test_local_agent_requires_pairing_and_per_run_compute_consent(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    token = "local-agent-pairing-token-value"
    auth = AgentAuthenticator(token)
    manager = RunManager(tmp_path / "agent-workspace", trainer_factory=_FakeTrainer)
    app = create_app(
        manager=manager,
        authenticator=auth,
        cors_origins=["https://studio.example"],
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Origin": "https://studio.example",
    }
    config = {
        "preset": "fast",
        "target": "label",
        "llm": False,
        "contribute_memory": False,
        "execution_target": "local_agent",
    }
    csv_data = "feature,label\n0,no\n1,yes\n"

    with pytest.raises(AuthenticationError):
        auth.authenticate("Bearer wrong-token-value-that-is-long")

    with fastapi_testclient.TestClient(app) as client:
        preflight = client.options(
            "/api/agent/capabilities",
            headers={
                "Origin": "https://studio.example",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-private-network"] == "true"
        capabilities = client.get("/api/agent/capabilities", headers=headers)
        assert capabilities.status_code == 200
        assert capabilities.json()["consent_required_for_every_run"]
        assert capabilities.headers["access-control-allow-origin"] == (
            "https://studio.example"
        )

        denied = client.post(
            "/api/runs",
            headers=headers,
            data={"config": json.dumps(config)},
            files={"files": ("data.csv", csv_data, "text/csv")},
        )
        assert denied.status_code == 400
        assert "permission" in denied.json()["detail"].lower()

        config["local_gpu_consent"] = True
        accepted = client.post(
            "/api/runs",
            headers=headers,
            data={"config": json.dumps(config)},
            files={"files": ("data.csv", csv_data, "text/csv")},
        )
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["config"]["execution_target"] == "local_agent"
    manager.shutdown()


def test_frontend_supports_remote_api_and_explicit_local_compute_permission():
    script = Path("autonexus/web_static/app.js").read_text(encoding="utf-8")
    page = Path("autonexus/web_static/index.html").read_text(encoding="utf-8")

    assert "runtimeConfig.apiBaseUrl" in script
    assert "local_gpu_consent" in script
    assert "window.confirm" in script
    assert "/api/agent/capabilities" in script
    assert "LOCAL AGENT URL" in page
    assert "/assets/config.js" in page


def test_firebase_storage_mirror_keeps_compact_sqlite_metadata(tmp_path):
    uploaded: list[tuple[str, Path]] = []

    class FakeBlob:
        def __init__(self, name):
            self.name = name

        def upload_from_filename(self, filename):
            uploaded.append((self.name, Path(filename)))

    class FakeBucket:
        def blob(self, name):
            return FakeBlob(name)

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "cats").mkdir()
    (dataset / "cats" / "one.jpg").write_bytes(b"cat")
    (dataset / "dogs").mkdir()
    (dataset / "dogs" / "two.jpg").write_bytes(b"dogs")
    mirror = object.__new__(FirebaseStorageMirror)
    mirror.bucket_name = "project.firebasestorage.app"
    mirror.prefix = "autonexus"
    mirror._bucket = FakeBucket()
    state = {
        "id": "20260808-120000-abcdef",
        "owner_id": "user@example.com",
        "dataset": str(dataset),
    }

    result = mirror.mirror_dataset(state)

    assert result["object_count"] == 2
    assert result["size_bytes"] == 7
    assert "objects" not in result
    assert result["prefix"].startswith("gs://project.firebasestorage.app/")
    assert {name.rsplit("/", 2)[-2] for name, _ in uploaded} == {"cats", "dogs"}
