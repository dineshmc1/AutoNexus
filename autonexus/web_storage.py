"""Optional Firebase Storage mirroring for Studio datasets and artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


class FirebaseStorageMirror:
    """Mirror local run blobs to Firebase Storage; never uses Firestore."""

    def __init__(self, bucket_name: str, *, prefix: str = "autonexus") -> None:
        try:
            from firebase_admin import storage
        except ImportError as exc:
            raise RuntimeError(
                'Firebase Storage requires: pip install "AutoNexus[cloud]"'
            ) from exc
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/") or "autonexus"
        self._bucket = storage.bucket(bucket_name)

    @classmethod
    def from_env(cls) -> "FirebaseStorageMirror | None":
        bucket = os.getenv("AUTONEXUS_FIREBASE_STORAGE_BUCKET", "").strip()
        if not bucket:
            return None
        return cls(
            bucket,
            prefix=os.getenv("AUTONEXUS_FIREBASE_STORAGE_PREFIX", "autonexus"),
        )

    def _base(self, state: dict[str, Any]) -> str:
        owner = _safe_segment(str(state.get("owner_id", "local-user")))
        run_id = _safe_segment(str(state["id"]))
        return f"{self.prefix}/{owner}/{run_id}"

    def _upload_file(self, source: Path, object_name: str) -> int:
        blob = self._bucket.blob(object_name)
        blob.upload_from_filename(str(source))
        return source.stat().st_size

    def mirror_dataset(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset = Path(str(state["dataset"])).resolve()
        files = [dataset] if dataset.is_file() else [
            item for item in dataset.rglob("*") if item.is_file()
        ]
        total_bytes = 0
        for source in files:
            relative = source.name if dataset.is_file() else source.relative_to(dataset).as_posix()
            total_bytes += self._upload_file(
                source, f"{self._base(state)}/input/{relative}"
            )
        return {
            "provider": "firebase_storage",
            "bucket": self.bucket_name,
            "prefix": f"gs://{self.bucket_name}/{self._base(state)}/input/",
            "object_count": len(files),
            "size_bytes": total_bytes,
        }

    def mirror_artifacts(self, state: dict[str, Any]) -> dict[str, Any]:
        output = Path(str(state["output_dir"])).resolve()
        files = [item for item in output.rglob("*") if item.is_file()]
        total_bytes = 0
        for source in files:
            relative = source.relative_to(output).as_posix()
            total_bytes += self._upload_file(
                source,
                f"{self._base(state)}/artifacts/{relative}",
            )
        return {
            "provider": "firebase_storage",
            "bucket": self.bucket_name,
            "prefix": f"gs://{self.bucket_name}/{self._base(state)}/artifacts/",
            "object_count": len(files),
            "size_bytes": total_bytes,
        }
