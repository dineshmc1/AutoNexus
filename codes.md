# AutoNexus Code Guide

This file is the practical reference for the public AutoNexus framework API.
Every successful training run writes:

- `run.json`
- `model.pkl`
- `analysis.ipynb`
- `report/explanation.md`
- `search_profile.json`

## 1. Install

```powershell
pip install AutoNexus
```

Install only the capabilities required by the application:

```powershell
pip install "AutoNexus[vision]"
pip install "AutoNexus[llm]"
pip install "AutoNexus[memory]"
pip install "AutoNexus[monitoring]"
pip install "AutoNexus[streaming]"
pip install "AutoNexus[serve]"
pip install "AutoNexus[analytics]"
pip install "AutoNexus[all]"
```

## 2. Train in Fewer Than Five Lines

```python
from autonexus import AutoNexus

model = AutoNexus(output_dir="artifacts/customer").fit(
    "customers.csv", target="churn"
)
```

The capitalized compatibility import is also supported:

```python
import AutoNexus

model = AutoNexus.AutoNexus().fit("customers.csv", target="churn")
```

## 3. Train from a DataFrame

```python
import pandas as pd
from autonexus import AutoNexus

frame = pd.read_parquet("training.parquet")
model = AutoNexus(preset="balanced").fit(frame, label_column="label")
```

## 4. Train an Image Classifier

```python
from autonexus import AutoNexus

model = AutoNexus(
    task="vision",
    preset="accurate",
    backbones=["auto"],
    adapt_lora=True,
).fit("dataset/")
```

Supported layouts include class folders and explicit train/validation/test
folders:

```text
dataset/
  train/
    cats/
    dogs/
  val/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

## 5. Fast, Accurate, Low-Memory, and Online Presets

```python
from autonexus import AutoNexus

fast = AutoNexus(preset="fast")
accurate = AutoNexus(preset="accurate", max_time="4h")
small_machine = AutoNexus(preset="low_memory")
incremental = AutoNexus(preset="online")
```

The `online` preset restricts selection to estimators with native
`partial_fit` support.

## 6. Fully Customized Training

```python
from autonexus import AutoNexus

trainer = AutoNexus(
    task="classification",
    output_dir="artifacts/fraud-v3",
    models=["logistic", "sgd_clf", "et_clf", "xgb_clf"],
    cv=5,
    max_time="90m",
    sample_fraction=0.15,
    feature_engineering=True,
    tune=True,
    tune_iterations=30,
    shap=True,
    use_memory=True,
    contribute_memory=True,
)
model = trainer.fit("fraud.csv", target="is_fraud")
```

## 7. Configure FAISS Search Memory

Retrieval and contribution are enabled by default. Retrieval advises the
shortlist using compatible nearby runs; current validation evidence can always
override it. Future contributions store selection evidence, not held-out test
metrics, and never store raw training rows.

```python
from autonexus import AutoNexus

model = AutoNexus(
    use_memory=False,
    contribute_memory=False,
).fit(
    "private.csv", target="label"
)
```

Use a project-specific memory directory:

```python
from autonexus import AutoNexus

model = AutoNexus(
    contribute_memory=True,
    memory_dir="./team_memory",
).fit("data.csv", target="label")
```

Search the memory directly:

```python
import json
import numpy as np
from autonexus import FAISSMetaMemory

profile = json.load(open("artifacts/search_profile.json", encoding="utf-8"))
memory = FAISSMetaMemory("./team_memory")
neighbors = memory.search(np.asarray(profile["embedding"]), k=5)
```

FAISS is used when installed. A NumPy index is maintained as a portable
fallback.

## 8. Predict and Get Probabilities

```python
import pandas as pd

new_rows = pd.read_csv("incoming.csv")
predictions = model.predict(new_rows)
probabilities = model.predict_proba(new_rows)
```

Image inference accepts an image, folder, or list of image paths:

```python
predictions = model.predict("unseen_images/")
```

## 9. Save and Load a Run

```python
from autonexus import AutoNexus, NexusModel

model.save("deployable/customer-churn")
loaded = NexusModel.load("deployable/customer-churn")
predictions = loaded.predict(new_rows)
```

Load through the main facade:

```python
loaded = AutoNexus.load("artifacts/customer")
```

Only load `.pkl` or Joblib artifacts from trusted sources because Python
pickle-compatible formats can execute code while loading.

## 10. Inspect Artifacts

```python
print(model.artifacts)
print(model.best_model)
print(model.problem_type)
print(model.explain())
```

## 11. Lifecycle Callbacks

```python
from autonexus import AutoNexus, Callback

class TrainingLogger(Callback):
    def on_training_started(self, event):
        print("Starting:", event.payload["dataset"])

    def on_training_completed(self, event):
        print("Selected:", event.payload["best_model"])

    def on_drift(self, event):
        print("Drift report:", event.payload["report"])

model = AutoNexus(callbacks=[TrainingLogger()]).fit(
    "data.csv", target="label"
)
```

A function can be used directly:

```python
model = AutoNexus(
    callbacks=[lambda event: print(event.name)]
).fit("data.csv", target="label")
```

## 12. Register a Custom Estimator

```python
from sklearn.linear_model import PassiveAggressiveClassifier
from autonexus import AutoNexus

trainer = AutoNexus(models=[])
trainer.register_model(
    "passive_aggressive",
    PassiveAggressiveClassifier(random_state=42),
    problem_type="classification",
)
model = trainer.fit("data.csv", target="label")
```

## 13. Offline or Custom LLM

Disable external LLM calls while still producing `explanation.md`:

```python
model = AutoNexus(llm=False).fit("data.csv", target="label")
```

Use any Python callable:

```python
from autonexus import AutoNexus

def my_llm(*, prompt, context):
    return f"# Internal Report\n\nSelected: {context['best_model']}"

model = AutoNexus(llm=my_llm).fit("data.csv", target="label")
```

## 14. LiteLLM and OpenAI-Compatible APIs

```python
from autonexus import AutoNexus, LiteLLMProvider

provider = LiteLLMProvider(
    "openai/my-model",
    api_base="https://my-company.example/v1",
    api_key="read-from-a-secret-manager",
)
model = AutoNexus(llm=provider).fit("data.csv", target="label")
```

LiteLLM supports many hosted and OpenAI-compatible providers. Do not hard-code
API keys in source control.

## 15. Local Ollama

```python
from autonexus import AutoNexus, OllamaProvider

provider = OllamaProvider("llama3.2")
model = AutoNexus(llm=provider).fit("data.csv", target="label")
```

## 16. Local Hugging Face Model

```python
from autonexus import AutoNexus, TransformersProvider

provider = TransformersProvider(
    "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens=1000,
)
model = AutoNexus(llm=provider).fit("data.csv", target="label")
```

## 17. Arbitrary JSON LLM API

```python
from autonexus import AutoNexus, HTTPJSONProvider

provider = HTTPJSONProvider(
    "https://llm.internal/generate",
    headers={"Authorization": "Bearer secret-from-vault"},
    request_builder=lambda prompt, context: {
        "prompt": prompt,
        "run": context,
    },
    response_parser=lambda response: response["text"],
)
model = AutoNexus(llm=provider).fit("data.csv", target="label")
```

LLMs only write explanations. They do not promote models or override
deterministic validation gates.

## 18. Build a Drift Baseline

Training creates `monitoring/baseline.json` automatically.

```python
from autonexus import DriftBaseline, DriftDetector

baseline = DriftBaseline.load("artifacts/monitoring/baseline.json")
detector = DriftDetector(
    baseline,
    feature_threshold=0.2,
    categorical_threshold=0.15,
    performance_drop_threshold=0.03,
)
```

## 19. Monitor One Batch

```python
import pandas as pd

monitor = model.monitor(
    feature_threshold=0.2,
    prediction_threshold=0.15,
    minimum_samples=100,
)
report = monitor.observe(pd.read_csv("production_batch.csv"))
print(report.to_dict())
```

If the batch contains fewer rows than `minimum_samples`, the report returns
`severity="insufficient_data"` and suppresses population-level alarms. Schema
and invalid-type checks still run immediately.

## 20. Monitoring Sinks

```python
from autonexus import JSONLSink, LoggingSink, WebhookSink

monitor = model.monitor(
    sinks=[
        LoggingSink(),
        JSONLSink("monitoring/drift.jsonl"),
        WebhookSink("https://alerts.internal/autonexus"),
    ]
)
```

Optional Prometheus metrics:

```python
from autonexus import PrometheusSink

monitor = model.monitor(sinks=[PrometheusSink(namespace="fraud_model")])
```

## 21. Incremental Learning

Train with the online preset:

```python
from autonexus import AutoNexus

model = AutoNexus(preset="online").fit(
    "initial_data.csv", target="label"
)
```

Update with new labelled rows:

```python
import pandas as pd

new_batch = pd.read_csv("new_labelled_rows.csv")
result = model.update(new_batch, target="label")
print(result)
```

With `strategy="auto"`, AutoNexus uses `partial_fit` when it is genuinely
supported. Trees and ensembles receive an immutable retrained challenger.
Every candidate is evaluated on an unseen new-data gate and promoted only
when it strictly outperforms the champion.

Customize the policy:

```python
from autonexus import UpdatePolicy

policy = UpdatePolicy(
    minimum_batch_size=100,
    validation_fraction=0.25,
    minimum_improvement=0.002,
)
result = model.update(new_batch, target="label", policy=policy)
```

## 22. Non-Incremental Model Replacement

Force a full challenger even for an incremental estimator:

```python
result = model.update(new_batch, target="label", strategy="retrain")
```

Replacement training writes a separate run under `updates/`; it does not
silently overwrite the current production model.

## 23. DataFrame Streaming

```python
from autonexus import FrameSource

source = FrameSource(training_frame, batch_size=5000)
model = AutoNexus(preset="online").fit_source(
    source,
    target="label",
    initial_batches=10,
)
```

## 24. File and SQL Sources

```python
from autonexus import FileSource, SQLSource

csv_source = FileSource("events.csv", batch_size=10_000)
sql_source = SQLSource(
    "SELECT * FROM labelled_events",
    "events.sqlite",
    batch_size=10_000,
)

model = AutoNexus().fit_source(
    sql_source, target="label", initial_batches=20
)
```

## 25. Python Iterator Source

```python
from autonexus import IterableSource

def events():
    yield {"feature_a": 1, "feature_b": 2, "label": "yes"}
    yield {"feature_a": 0, "feature_b": 3, "label": "no"}

source = IterableSource(events())
```

## 26. Kafka or Redpanda

```python
from autonexus import KafkaSource

source = KafkaSource(
    "labelled-events",
    bootstrap_servers="localhost:9092",
    group_id="fraud-training",
    batch_size=2000,
)

model = AutoNexus(preset="online").fit_source(
    source,
    target="label",
    initial_batches=10,
)
```

## 27. Real-Time Monitoring and Gated Updates

```python
monitor = model.monitor(minimum_samples=200)

for report in monitor.run(
    source,
    update_on_drift=True,
    update_strategy="auto",
):
    print(report.severity, report.drifted)
```

The source must contain the label column for automatic performance monitoring
or model updates. Without labels, AutoNexus monitors schema, feature, and
prediction drift only.

## 28. Local Model Registry

```python
from autonexus import ModelRegistry

registry = ModelRegistry("./registry")
version = registry.register(
    model.output_dir,
    name="customer-churn",
    stage="challenger",
)
registry.promote("customer-churn", version.version)
```

Rollback:

```python
previous = registry.rollback("customer-churn")
print(previous.path)
```

Register directly from a loaded model:

```python
model.register(
    "customer-churn",
    registry=registry,
    version="2026-07-30",
    promote=True,
)
```

## 29. Launch Auto Nexus Studio

Install the optional web dependencies:

```powershell
pip install "AutoNexus[serve]"
```

Launch the local training interface:

```powershell
autonexus-web
```

Choose a different run workspace and port:

```powershell
autonexus-web --workspace D:\AutoNexusRuns --port 9000
```

The Studio opens on `http://127.0.0.1:8787` by default. Runs are stored under
`%LOCALAPPDATA%\AutoNexus\studio-runs` on Windows unless `--workspace` or
`AUTONEXUS_WEB_WORKSPACE` is set.

For Firebase-protected multi-user access:

```powershell
pip install "AutoNexus[serve,auth]"
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\secure\firebase-admin.json"
$env:AUTONEXUS_AUTH_MODE="firebase"
$env:AUTONEXUS_FIREBASE_API_KEY="your-web-api-key"
$env:AUTONEXUS_FIREBASE_PROJECT_ID="your-project-id"
$env:AUTONEXUS_FIREBASE_AUTH_DOMAIN="your-project-id.firebaseapp.com"
$env:AUTONEXUS_FIREBASE_APP_ID="your-web-app-id"
autonexus-web --host 0.0.0.0
```

Enable Email/Password in Firebase Authentication before launching. Remote
users are upload-only unless the administrator deliberately sets
`AUTONEXUS_ALLOW_REMOTE_LOCAL_PATHS=true`.

In **LLM Intelligence / BYOK**, choose one of:

```text
Server environment       Uses LLM_MODEL and its provider environment key
Bring your own API key   Choose provider, model ID, key, and optional endpoint
Local Ollama             Choose a local model and Ollama endpoint
Deterministic offline    Makes no external LLM request
```

BYOK keys exist only in process memory for the selected mission. They are not
saved in browser storage, run manifests, reports, logs, or artifacts.

Download **Runnable analytics bundle** from a completed mission, extract it,
and open `analysis.ipynb` from the extracted directory. The ZIP includes the
bounded `analysis_data` files required by the notebook.

## 30. One-Line Deployment

Start a safe localhost deployment in the background:

```python
deployment = model.deploy()
```

Inspect `deployment.predict_url`, and stop it with `deployment.stop()`.
Public binding requires explicit authentication and acknowledgement that TLS
must be terminated by a trusted reverse proxy:

```python
import os

deployment = model.deploy(
    host="0.0.0.0",
    port=8000,
    api_key=os.environ["AUTONEXUS_API_KEY"],
    allow_insecure_public=True,
)
```

The original blocking development server remains available:

```python
model.serve(host="127.0.0.1", port=8000)
```

Request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"feature_a":1.2,"feature_b":"blue"}]}'
```

Health endpoint:

```bash
curl http://localhost:8000/health
```

## 31. CLI

Interactive:

```powershell
autonexus
```

Tabular:

```powershell
autonexus data.csv --target label --output-dir artifacts/run-1
```

Images:

```powershell
autonexus dataset --adapt-lora --backbones auto
```

Disable memory contribution:

```powershell
autonexus data.csv --target label --no-contribute-memory
```

Disable both retrieval and contribution:

```powershell
autonexus data.csv --target label --no-memory-retrieval --no-contribute-memory
```

Use a project memory:

```powershell
autonexus data.csv --target label --memory-dir .\team_memory
```

## 32. Production Safety Checklist

```text
1. Keep the held-out test set untouched until final evaluation.
2. Use labelled, unseen update gates before promoting online candidates.
3. Treat model.pkl as trusted-only executable serialization.
4. Keep API keys in a secret manager.
5. Review drift thresholds using production seasonality.
6. Require human approval for high-risk model promotion.
7. Monitor per-class and per-group performance, not only global accuracy.
8. Keep champion versions available for rollback.
```
