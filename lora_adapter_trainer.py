"""Supervised LoRA adaptation with regularization and early stopping."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from image_splitting import split_labeled_indices
from multimodal_extractor import (
    MultiModalDataset,
    TrainImageAugmentation,
    extract_vision_features,
    multimodal_collate,
)
from vision_backbones import (
    DEFAULT_BACKBONE_KEY,
    VISION_BACKBONES,
    lora_configuration,
)


def _discover_files(data_dir: str, modality: str) -> tuple[list[str], list[str]]:
    extensions = {
        "vision": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
        "audio": {".wav", ".mp3", ".flac", ".ogg"},
        "text": {".txt", ".md"},
    }[modality]
    files: list[str] = []
    labels: list[str] = []
    for root, _, filenames in os.walk(data_dir):
        label = os.path.basename(root)
        if root == data_dir:
            continue
        if label.lower() == "images":
            label = os.path.basename(os.path.dirname(root))
        for filename in filenames:
            path = Path(root, filename)
            if path.suffix.lower() in extensions:
                files.append(str(path))
                labels.append(label)
    return files, labels


def train_universal_lora(
    modality: str,
    domain: str,
    data_dir: str,
    output_path: str,
    epochs: int = 8,
    batch_size: int = 32,
    patience: int = 2,
    min_delta: float = 1e-4,
    validation_size: float = 0.15,
    weight_decay: float | None = None,
    files: list[str] | None = None,
    labels: list[str] | None = None,
    groups: list[str] | None = None,
    random_state: int = 42,
    model_id: str | None = None,
    model_revision: str = "main",
):
    """Train an adapter and restore the checkpoint with lowest validation NLL."""
    if modality != "vision":
        raise ValueError("The production CLI currently supports vision LoRA only.")
    if files is None and not os.path.isdir(data_dir):
        raise ValueError(f"LoRA data directory not found: {data_dir}")
    selected_model_id = (
        model_id
        or VISION_BACKBONES[DEFAULT_BACKBONE_KEY].model_id
    )
    cfg = lora_configuration(selected_model_id)
    if cfg is None:
        raise ValueError(
            f"Backbone '{selected_model_id}' does not support transformer LoRA."
        )

    if files is None or labels is None:
        files, labels = _discover_files(data_dir, modality)
    else:
        files, labels = list(files), list(labels)
    if len(files) < 20 or len(set(labels)) < 2:
        raise ValueError(
            "LoRA adaptation requires at least 20 files across two classes."
        )
    if groups is not None and len(groups) != len(files):
        raise ValueError("LoRA groups must align with image files.")

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = None
    tokenizer = None
    if modality == "vision":
        from transformers import AutoConfig, AutoModel, AutoProcessor

        model_config = AutoConfig.from_pretrained(
            cfg["base_model"], revision=model_revision
        )
        resolved_revision = (
            getattr(model_config, "_commit_hash", None) or model_revision
        )
        processor = AutoProcessor.from_pretrained(
            cfg["base_model"], revision=resolved_revision
        )
        base_model = AutoModel.from_pretrained(
            cfg["base_model"], revision=resolved_revision
        ).to(device)
    elif modality == "audio":
        from transformers import AutoFeatureExtractor, ASTModel

        processor = AutoFeatureExtractor.from_pretrained(cfg["base_model"])
        base_model = ASTModel.from_pretrained(cfg["base_model"]).to(device)
    else:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        base_model = AutoModel.from_pretrained(cfg["base_model"]).to(device)

    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            target_modules=cfg["target_modules"],
            lora_dropout=0.1,
            bias="none",
            task_type=cfg["task_type"],
        ),
    )
    checkpointing_enabled = False
    if device.type == "cuda":
        total_vram_gb = (
            torch.cuda.get_device_properties(device).total_memory
            / (1024 ** 3)
        )
        if total_vram_gb <= 12 and hasattr(
            peft_model, "gradient_checkpointing_enable"
        ):
            peft_model.gradient_checkpointing_enable()
            if hasattr(peft_model, "enable_input_require_grads"):
                peft_model.enable_input_require_grads()
            checkpointing_enabled = True
    hidden_size = getattr(
        base_model.config,
        "projection_dim",
        getattr(base_model.config, "hidden_size", 768),
    )
    class_names = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(class_names)}
    classifier = torch.nn.Linear(hidden_size, len(class_names)).to(device)

    train_indices, val_indices, split_method = split_labeled_indices(
        labels,
        test_size=validation_size,
        random_state=random_state,
        groups=groups,
    )
    directional_tokens = (
        "left",
        "right",
        "clockwise",
        "counterclockwise",
    )
    directional_labels = any(
        token in str(label).lower()
        for label in labels
        for token in directional_tokens
    )
    flip_probability = 0.0 if directional_labels else 0.5
    train_dataset = MultiModalDataset(
        files,
        labels,
        modality=modality,
        transform=TrainImageAugmentation(
            horizontal_flip_probability=flip_probability
        ),
    )
    validation_dataset = MultiModalDataset(
        files, labels, modality=modality
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multimodal_collate,
    )
    val_loader = DataLoader(
        Subset(validation_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate,
    )

    decay = (
        cfg.get("weight_decay", 1e-4)
        if weight_decay is None
        else weight_decay
    )

    def make_optimizer():
        adapter_params = [
            parameter
            for parameter in peft_model.parameters()
            if parameter.requires_grad
        ]
        return torch.optim.AdamW(
            [
                {"params": adapter_params, "weight_decay": decay},
                {"params": classifier.parameters(), "weight_decay": decay},
            ],
            lr=1e-4,
        )

    optimizer = make_optimizer()
    loss_fn = torch.nn.CrossEntropyLoss()

    def extract_features(batch):
        if modality == "vision":
            features = extract_vision_features(
                peft_model, processor, batch, device
            )
        elif modality == "audio":
            inputs = processor(
                batch,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(device)
            outputs = peft_model(**inputs)
            features = outputs.last_hidden_state[:, 0]
        else:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            outputs = peft_model(**inputs)
            features = outputs.last_hidden_state[:, 0]
        return features.flatten(start_dim=1) if features.ndim > 2 else features

    max_epochs = 2 if len(files) > 50_000 else 3 if len(files) > 10_000 else epochs
    best_loss = float("inf")
    best_adapter_state = None
    best_classifier_state = None
    stale_epochs = 0
    print(
        f"[LoRA] Training {len(train_indices)} / validating "
        f"{len(val_indices)} samples; weight_decay={decay}; "
        f"gradient_checkpointing={checkpointing_enabled}; "
        f"split={split_method}; augmentation=train-only "
        f"(horizontal_flip_probability={flip_probability})."
    )

    for epoch in range(max_epochs):
        peft_model.train()
        classifier.train()
        train_total = 0.0
        train_count = 0
        for batch, batch_labels in tqdm(
            train_loader, desc=f"LoRA {epoch + 1}/{max_epochs}"
        ):
            if not batch:
                continue
            targets = torch.tensor(
                [label_to_id[label] for label in batch_labels],
                device=device,
            )
            optimizer.zero_grad()
            features = extract_features(batch)
            if classifier.in_features != features.shape[1]:
                classifier = torch.nn.Linear(
                    features.shape[1], len(class_names)
                ).to(device)
                optimizer = make_optimizer()
            loss = loss_fn(classifier(features), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in peft_model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            optimizer.step()
            train_total += loss.item() * len(targets)
            train_count += len(targets)

        peft_model.eval()
        classifier.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for batch, batch_labels in val_loader:
                if not batch:
                    continue
                targets = torch.tensor(
                    [label_to_id[label] for label in batch_labels],
                    device=device,
                )
                loss = loss_fn(classifier(extract_features(batch)), targets)
                val_total += loss.item() * len(targets)
                val_count += len(targets)
        if not val_count:
            raise RuntimeError("No valid LoRA validation samples were decoded.")

        train_loss = train_total / max(train_count, 1)
        val_loss = val_total / val_count
        print(
            f"[LoRA] Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_adapter_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(peft_model).items()
            }
            best_classifier_state = copy.deepcopy(classifier.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[LoRA] Early stopping at epoch {epoch + 1}.")
                break

    if best_adapter_state is None:
        raise RuntimeError("LoRA adaptation produced no valid checkpoint.")
    set_peft_model_state_dict(peft_model, best_adapter_state)
    classifier.load_state_dict(best_classifier_state)

    os.makedirs(output_path, exist_ok=True)
    peft_model.save_pretrained(output_path)
    if processor is not None:
        processor.save_pretrained(output_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)
    torch.save(classifier.state_dict(), Path(output_path, "classifier.pt"))
    metadata = {
        "best_validation_nll": best_loss,
        "epochs_completed": epoch + 1,
        "early_stopping_patience": patience,
        "weight_decay": decay,
        "gradient_checkpointing": checkpointing_enabled,
        "split_method": split_method,
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "train_only_augmentation": True,
        "horizontal_flip_probability": flip_probability,
        "directional_labels_detected": directional_labels,
        "random_state": random_state,
        "base_model": selected_model_id,
        "base_model_revision": resolved_revision,
        "target_modules": cfg["target_modules"],
    }
    Path(output_path, "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"[LoRA] Adapter saved to {output_path}; "
        f"best validation loss={best_loss:.4f}."
    )
    return output_path
