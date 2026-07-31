"""Batch and streaming data-source interfaces."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd


class DataSource(ABC):
    """A restartable source of pandas batches."""

    @abstractmethod
    def batches(self) -> Iterator[pd.DataFrame]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return self.batches()


class FrameSource(DataSource):
    def __init__(self, frame: pd.DataFrame, batch_size: int | None = None):
        self.frame = frame.copy()
        self.batch_size = batch_size or len(frame)

    def batches(self) -> Iterator[pd.DataFrame]:
        for start in range(0, len(self.frame), self.batch_size):
            yield self.frame.iloc[start : start + self.batch_size].copy()


class FileSource(DataSource):
    def __init__(self, path: str | Path, batch_size: int = 10_000):
        self.path = Path(path)
        self.batch_size = batch_size

    def batches(self) -> Iterator[pd.DataFrame]:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            yield from pd.read_csv(self.path, chunksize=self.batch_size)
            return
        if suffix in {".xlsx", ".xls"}:
            yield pd.read_excel(self.path)
            return
        if suffix in {".parquet", ".pq"}:
            yield pd.read_parquet(self.path)
            return
        raise ValueError(f"Unsupported data source format: {suffix}")


class IterableSource(DataSource):
    def __init__(self, iterable: Iterable[Any]):
        self.iterable = iterable

    def batches(self) -> Iterator[pd.DataFrame]:
        for item in self.iterable:
            if isinstance(item, pd.DataFrame):
                yield item.copy()
            elif isinstance(item, dict):
                yield pd.DataFrame([item])
            else:
                yield pd.DataFrame(item)


class SQLSource(DataSource):
    """DB-API/SQLite query source with bounded batch reads."""

    def __init__(
        self,
        query: str,
        connection: Any,
        *,
        batch_size: int = 10_000,
    ):
        self.query = query
        self.connection = connection
        self.batch_size = batch_size

    def batches(self) -> Iterator[pd.DataFrame]:
        connection = (
            sqlite3.connect(self.connection)
            if isinstance(self.connection, (str, Path))
            else self.connection
        )
        try:
            yield from pd.read_sql_query(
                self.query, connection, chunksize=self.batch_size
            )
        finally:
            if isinstance(self.connection, (str, Path)):
                connection.close()


class KafkaSource(DataSource):
    """Optional Kafka/Redpanda JSON consumer."""

    def __init__(
        self,
        topic: str,
        *,
        bootstrap_servers: str,
        group_id: str = "autonexus",
        batch_size: int = 1000,
        poll_timeout_ms: int = 1000,
        **consumer_options: Any,
    ):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self.consumer_options = consumer_options

    def batches(self) -> Iterator[pd.DataFrame]:
        try:
            from kafka import KafkaConsumer
        except ImportError as exc:
            raise RuntimeError(
                "Kafka support requires: pip install AutoNexus[streaming]"
            ) from exc
        import json

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            **self.consumer_options,
        )
        pending = []
        try:
            while True:
                records = consumer.poll(timeout_ms=self.poll_timeout_ms)
                for messages in records.values():
                    pending.extend(message.value for message in messages)
                if len(pending) >= self.batch_size:
                    yield pd.DataFrame(pending[: self.batch_size])
                    pending = pending[self.batch_size :]
        finally:
            consumer.close()


def as_source(value: Any, *, batch_size: int = 10_000) -> DataSource:
    if isinstance(value, DataSource):
        return value
    if isinstance(value, pd.DataFrame):
        return FrameSource(value, batch_size=batch_size)
    if isinstance(value, (str, Path)):
        return FileSource(value, batch_size=batch_size)
    if isinstance(value, Iterable):
        return IterableSource(value)
    raise TypeError(f"Cannot convert {type(value).__name__} into a DataSource")

