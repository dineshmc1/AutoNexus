"""Real-time monitoring loop and pluggable observability sinks."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from .data import DataSource, as_source
from .drift import DriftBaseline, DriftDetector, DriftReport

if TYPE_CHECKING:
    from .model import NexusModel

LOGGER = logging.getLogger("autonexus.monitoring")


class MonitorSink(ABC):
    @abstractmethod
    def emit(self, report: DriftReport) -> None:
        raise NotImplementedError


class LoggingSink(MonitorSink):
    def emit(self, report: DriftReport) -> None:
        LOGGER.info(
            "drift=%s severity=%s samples=%d",
            report.drifted,
            report.severity,
            report.sample_count,
        )


class JSONLSink(MonitorSink):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def emit(self, report: DriftReport) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": time.time(), **report.to_dict()}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


class WebhookSink(MonitorSink):
    def __init__(self, url: str, *, timeout: float = 10.0):
        self.url = url
        self.timeout = timeout

    def emit(self, report: DriftReport) -> None:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(report.to_dict()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout):
            pass


class PrometheusSink(MonitorSink):
    def __init__(self, namespace: str = "autonexus"):
        try:
            from prometheus_client import Gauge
        except ImportError as exc:
            raise RuntimeError(
                "Prometheus support requires AutoNexus[monitoring]"
            ) from exc
        self.drift = Gauge(
            f"{namespace}_drift_detected", "Whether drift was detected"
        )
        self.samples = Gauge(
            f"{namespace}_monitor_samples", "Last monitored batch size"
        )

    def emit(self, report: DriftReport) -> None:
        self.drift.set(float(report.drifted))
        self.samples.set(report.sample_count)


class NexusMonitor:
    def __init__(
        self,
        model: "NexusModel",
        baseline: DriftBaseline,
        *,
        detector: DriftDetector | None = None,
        sinks: list[MonitorSink] | None = None,
        label_column: str | None = None,
    ):
        self.model = model
        self.baseline = baseline
        self.detector = detector or DriftDetector(baseline)
        self.sinks = sinks or [
            LoggingSink(),
            JSONLSink(model.output_dir / "monitoring" / "events.jsonl"),
        ]
        self.label_column = label_column or baseline.target_name

    def observe(self, batch: pd.DataFrame) -> DriftReport:
        y_true = None
        features = batch
        if self.label_column and self.label_column in batch:
            y_true = batch[self.label_column]
            features = batch.drop(columns=[self.label_column])
        predictions = self.model.predict(features)
        report = self.detector.detect(
            batch, predictions=predictions, y_true=y_true
        )
        for sink in self.sinks:
            try:
                sink.emit(report)
            except Exception as exc:
                LOGGER.warning("Monitoring sink failed: %s", exc)
        self.model.callbacks.emit("drift", report=report.to_dict())
        return report

    def run(
        self,
        source: DataSource | Any,
        *,
        max_batches: int | None = None,
        update_on_drift: bool = False,
        update_strategy: str = "auto",
    ):
        for index, batch in enumerate(as_source(source)):
            if max_batches is not None and index >= max_batches:
                break
            report = self.observe(batch)
            if (
                update_on_drift
                and report.drifted
                and self.label_column
                and self.label_column in batch
            ):
                self.model.update(
                    batch,
                    target=self.label_column,
                    strategy=update_strategy,
                )
            yield report

