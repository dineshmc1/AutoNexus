"""Deterministic, cached vision embeddings for class-folder datasets."""

from __future__ import annotations

import hashlib
import gc
import random
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vision_backbones import DEFAULT_BACKBONE_KEY, VISION_BACKBONES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CACHE_VERSION = "vision-v3-multibackbone-normalized"
DEFAULT_VISION_MODEL_ID = VISION_BACKBONES[DEFAULT_BACKBONE_KEY].model_id


class TrainImageAugmentation:
    """Lightweight train-only augmentation without a torchvision dependency."""

    def __init__(
        self,
        crop_scale: tuple[float, float] = (0.8, 1.0),
        horizontal_flip_probability: float = 0.5,
        erase_probability: float = 0.15,
    ):
        self.crop_scale = crop_scale
        self.horizontal_flip_probability = horizontal_flip_probability
        self.erase_probability = erase_probability

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(*self.crop_scale)
        crop_width = max(1, int(width * scale))
        crop_height = max(1, int(height * scale))
        left = random.randint(0, max(0, width - crop_width))
        top = random.randint(0, max(0, height - crop_height))
        image = image.crop(
            (left, top, left + crop_width, top + crop_height)
        )
        if random.random() < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)

        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(0.85, 1.15)
        )
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(0.85, 1.15)
        )
        image = ImageEnhance.Color(image).enhance(
            random.uniform(0.9, 1.1)
        )
        if random.random() < self.erase_probability:
            draw = ImageDraw.Draw(image)
            erase_width = max(1, int(image.width * random.uniform(0.05, 0.15)))
            erase_height = max(
                1, int(image.height * random.uniform(0.05, 0.15))
            )
            erase_left = random.randint(
                0, max(0, image.width - erase_width)
            )
            erase_top = random.randint(
                0, max(0, image.height - erase_height)
            )
            draw.rectangle(
                (
                    erase_left,
                    erase_top,
                    erase_left + erase_width,
                    erase_top + erase_height,
                ),
                fill=(127, 127, 127),
            )
        return image


def discover_labeled_files(
    data_path: str, modality: str = "vision"
) -> tuple[list[str], list[str]]:
    """Return sorted image paths and labels inferred from parent folders."""
    if modality != "vision":
        raise ValueError("The production CLI currently supports vision only.")
    root = Path(data_path)
    if not root.is_dir():
        raise ValueError(f"Image directory not found: {root}")

    records: list[tuple[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue
        # The first directory under the supplied split/root is the class.
        records.append((str(path.resolve()), relative.parts[0]))
    records.sort(key=lambda item: item[0].casefold())
    if not records:
        raise ValueError(
            f"No labeled images found under {root}. Put images inside "
            "class-named subfolders."
        )
    files, labels = zip(*records)
    return list(files), list(labels)


class MultiModalDataset(Dataset):
    """Lazy image decoder retained as the shared LoRA/embedder dataset."""

    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[str],
        modality: str = "vision",
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ):
        if modality != "vision":
            raise ValueError("Only vision datasets are supported.")
        if len(files) != len(labels):
            raise ValueError("Image paths and labels must have equal length.")
        self.files = list(files)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        try:
            with Image.open(self.files[index]) as image:
                decoded = image.convert("RGB").copy()
                if self.transform is not None:
                    decoded = self.transform(decoded)
                return decoded, self.labels[index]
        except (OSError, ValueError):
            return None, self.labels[index]


def multimodal_collate(batch):
    """Drop unreadable images without losing label alignment."""
    valid = [(data, label) for data, label in batch if data is not None]
    if not valid:
        return [], []
    images, labels = zip(*valid)
    return list(images), list(labels)


def _vision_output_tensor(output):
    """Normalize tensors and Transformers model outputs to one feature tensor."""
    if torch.is_tensor(output):
        return output
    for attribute in ("image_embeds", "pooler_output", "last_hidden_state"):
        candidate = getattr(output, attribute, None)
        if candidate is None:
            continue
        if not torch.is_tensor(candidate):
            return _vision_output_tensor(candidate)
        return candidate[:, 0] if attribute == "last_hidden_state" else candidate
    if isinstance(output, (tuple, list)) and output:
        return _vision_output_tensor(output[0])
    raise RuntimeError("Backbone output has no supported image feature tensor.")


def extract_vision_features(model, processor, images, device):
    """Return one feature tensor for CLIP, ViT-style, or CNN backbones."""
    inputs = processor(images=images, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    feature_model = model
    if not hasattr(feature_model, "get_image_features"):
        base_model = getattr(model, "base_model", None)
        wrapped_model = getattr(base_model, "model", None)
        if hasattr(wrapped_model, "get_image_features"):
            feature_model = wrapped_model

    if hasattr(feature_model, "get_image_features"):
        output = feature_model.get_image_features(**inputs)
    else:
        output = model(**inputs)
    features = _vision_output_tensor(output)
    return features.flatten(start_dim=1) if features.ndim > 2 else features


class UniversalEmbedder:
    """Lazy generic vision embedder with exact, adapter-aware disk caching."""

    def __init__(
        self,
        device: torch.device | str = "cpu",
        batch_size: int = 32,
        domain: str = "general",
        max_files_per_class: int | None = None,
        modality: str = "vision",
        adapter_path: str | None = None,
        cache_dir: str = "embedding_cache",
        model_id: str = DEFAULT_VISION_MODEL_ID,
        model_revision: str = "main",
    ):
        if modality != "vision":
            raise ValueError("The production CLI currently supports vision only.")
        if domain != "general":
            raise ValueError("The production CLI currently supports domain='general'.")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.domain = domain
        self.max_files_per_class = max_files_per_class
        self.modality = modality
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.model_revision = model_revision
        self.resolved_model_revision: str | None = None
        self.adapter_path = (
            str(Path(adapter_path).resolve()) if adapter_path else None
        )
        if self.adapter_path and not Path(self.adapter_path).is_dir():
            raise ValueError(f"LoRA adapter not found: {self.adapter_path}")
        self.vision_model = None
        self.vision_processor = None
        self.last_cache_hit = False
        self.last_embedding_seconds = 0.0

    def release(self) -> None:
        """Release backbone objects before another representation is loaded."""
        self.vision_model = None
        self.vision_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self) -> None:
        if self.vision_model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        revision = self._resolve_model_revision()
        self.vision_processor = AutoProcessor.from_pretrained(
            self.model_id, revision=revision
        )
        model = AutoModel.from_pretrained(
            self.model_id, revision=revision
        )
        if self.adapter_path:
            from peft import PeftModel

            print(f"[Embedder] Loading LoRA adapter: {self.adapter_path}")
            model = PeftModel.from_pretrained(
                model, self.adapter_path, is_trainable=False
            ).merge_and_unload()
        self.vision_model = model.to(self.device)
        self.vision_model.eval()

    def _resolve_model_revision(self) -> str:
        """Resolve mutable branch names to a commit for exact cache identity."""
        if self.resolved_model_revision is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                self.model_id, revision=self.model_revision
            )
            self.resolved_model_revision = (
                getattr(config, "_commit_hash", None)
                or self.model_revision
            )
        return self.resolved_model_revision

    def _adapter_signature(self) -> str:
        if not self.adapter_path:
            return "frozen-base"
        files = sorted(
            path for path in Path(self.adapter_path).rglob("*") if path.is_file()
        )
        details = [
            f"{path.relative_to(self.adapter_path)}:{path.stat().st_size}:"
            f"{path.stat().st_mtime_ns}"
            for path in files
        ]
        return hashlib.sha256("|".join(details).encode()).hexdigest()

    def _cache_path(
        self,
        files: Sequence[str],
        labels: Sequence[str],
        cache_key: str,
    ) -> Path:
        signatures = []
        for filename, label in zip(files, labels):
            path = Path(filename)
            stat = path.stat()
            signatures.append(
                f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{label}"
            )
        identity = "|".join(
            [
                CACHE_VERSION,
                cache_key,
                self.model_id,
                self._resolve_model_revision(),
                self._adapter_signature(),
                *signatures,
            ]
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return self.cache_dir / f"vision_{digest}.npz"

    def _extract(self, files: Sequence[str], labels: Sequence[str]):
        self._load_model()
        loader = DataLoader(
            MultiModalDataset(files, labels),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            collate_fn=multimodal_collate,
        )
        embeddings: list[np.ndarray] = []
        valid_labels: list[str] = []
        with torch.inference_mode():
            for images, batch_labels in tqdm(loader, desc="Extracting images"):
                if not images:
                    continue
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    output = extract_vision_features(
                        self.vision_model,
                        self.vision_processor,
                        images,
                        self.device,
                    )
                if output.ndim > 2:
                    output = output.flatten(start_dim=1)
                output = torch.nn.functional.normalize(
                    output.float(), p=2, dim=1
                )
                embeddings.append(output.cpu().numpy())
                valid_labels.extend(batch_labels)
        if not embeddings:
            raise ValueError("No valid images could be decoded and embedded.")
        return np.vstack(embeddings), valid_labels

    def embed_files(
        self,
        files: Sequence[str],
        labels: Sequence[str],
        modality: str = "vision",
        cache_key: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Embed an explicit split without crossing train/test boundaries."""
        if modality != "vision":
            raise ValueError("Only vision datasets are supported.")
        if not files:
            raise ValueError("No image files were provided for embedding.")
        if len(files) != len(labels):
            raise ValueError("Image paths and labels must have equal length.")

        cache_path = self._cache_path(
            files, labels, cache_key or "default"
        )
        if cache_path.is_file():
            print(f"[Cache] Loading embeddings from {cache_path}.")
            with np.load(cache_path, allow_pickle=False) as cached:
                values = cached["X"].astype(np.float32, copy=False)
                cached_labels = cached["y"].astype(str).tolist()
                self.last_embedding_seconds = (
                    float(cached["embedding_seconds"])
                    if "embedding_seconds" in cached
                    else 0.0
                )
            self.last_cache_hit = True
        else:
            extraction_started = time.monotonic()
            values, cached_labels = self._extract(files, labels)
            self.last_embedding_seconds = (
                time.monotonic() - extraction_started
            )
            self.last_cache_hit = False
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary_path,
                X=values.astype(np.float16),
                y=np.asarray(cached_labels, dtype=str),
                embedding_seconds=np.asarray(self.last_embedding_seconds),
            )
            temporary_path.replace(cache_path)
            print(f"[Cache] Saved embeddings to {cache_path}.")

        columns = [f"feat_{index}" for index in range(values.shape[1])]
        return pd.DataFrame(values, columns=columns), pd.Series(cached_labels)
