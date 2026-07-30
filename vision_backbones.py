"""Production registry and resource filtering for vision backbones."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


DEFAULT_BACKBONE_KEY = "clip"


@dataclass(frozen=True)
class VisionBackboneSpec:
    """Static capabilities used before a heavyweight model is imported."""

    key: str
    model_id: str
    family: str
    input_size: int
    embedding_dimension: int
    parameters_millions: float
    estimated_ram_gb: float
    estimated_vram_gb: float
    batch_size: int
    license_id: str
    adaptation: str
    revision: str = "main"
    lora_target_modules: tuple[str, ...] = ()
    lora_rank: int = 8
    lora_alpha: int = 16

    @property
    def supports_lora(self) -> bool:
        return self.adaptation == "lora" and bool(self.lora_target_modules)

    def to_dict(self) -> dict:
        return asdict(self)


VISION_BACKBONES: dict[str, VisionBackboneSpec] = {
    "clip": VisionBackboneSpec(
        key="clip",
        model_id="openai/clip-vit-base-patch32",
        family="vision-language-transformer",
        input_size=224,
        embedding_dimension=512,
        parameters_millions=151.3,
        estimated_ram_gb=2.0,
        estimated_vram_gb=1.5,
        batch_size=32,
        license_id="mit",
        adaptation="lora",
        lora_target_modules=("q_proj", "v_proj"),
    ),
    "dinov2": VisionBackboneSpec(
        key="dinov2",
        model_id="facebook/dinov2-small",
        family="self-supervised-transformer",
        input_size=224,
        embedding_dimension=384,
        parameters_millions=22.1,
        estimated_ram_gb=1.0,
        estimated_vram_gb=0.8,
        batch_size=48,
        license_id="apache-2.0",
        adaptation="lora",
        lora_target_modules=("query", "value"),
    ),
    "resnet": VisionBackboneSpec(
        key="resnet",
        model_id="microsoft/resnet-50",
        family="convolutional-network",
        input_size=224,
        embedding_dimension=2048,
        parameters_millions=25.6,
        estimated_ram_gb=1.0,
        estimated_vram_gb=0.9,
        batch_size=48,
        license_id="apache-2.0",
        # PEFT q/v LoRA is not structurally valid for convolutional blocks.
        adaptation="frozen-only",
    ),
    "siglip": VisionBackboneSpec(
        key="siglip",
        model_id="google/siglip-base-patch16-224",
        family="vision-language-transformer",
        input_size=224,
        embedding_dimension=768,
        parameters_millions=203.4,
        estimated_ram_gb=3.0,
        estimated_vram_gb=2.5,
        batch_size=16,
        license_id="apache-2.0",
        adaptation="lora",
        lora_target_modules=("q_proj", "v_proj"),
    ),
}


def resolve_backbones(requested: Sequence[str]) -> list[VisionBackboneSpec]:
    """Resolve CLI keys while keeping registry order deterministic."""
    normalized = [item.strip().lower() for item in requested if item.strip()]
    if not normalized or normalized == ["auto"]:
        return list(VISION_BACKBONES.values())
    if "auto" in normalized:
        raise ValueError("'auto' cannot be combined with explicit backbones.")
    unknown = sorted(set(normalized) - set(VISION_BACKBONES))
    if unknown:
        raise ValueError(
            f"Unknown vision backbone(s): {unknown}. Available: "
            f"{sorted(VISION_BACKBONES)} or auto."
        )
    return [VISION_BACKBONES[key] for key in dict.fromkeys(normalized)]


def filter_backbones_for_resources(
    candidates: Sequence[VisionBackboneSpec],
    available_ram_gb: float,
    available_vram_gb: float | None,
    device_type: str,
) -> tuple[list[VisionBackboneSpec], list[dict]]:
    """Remove candidates that cannot fit with a conservative safety margin."""
    accepted: list[VisionBackboneSpec] = []
    rejected: list[dict] = []
    for spec in candidates:
        reason = None
        if spec.estimated_ram_gb > max(0.5, available_ram_gb * 0.7):
            reason = (
                f"estimated RAM {spec.estimated_ram_gb:.1f} GiB exceeds "
                "the 70% available-RAM safety limit"
            )
        elif (
            device_type == "cuda"
            and available_vram_gb is not None
            and spec.estimated_vram_gb > max(0.5, available_vram_gb * 0.8)
        ):
            reason = (
                f"estimated VRAM {spec.estimated_vram_gb:.1f} GiB exceeds "
                "the 80% device-memory safety limit"
            )
        elif device_type == "cpu" and spec.key == "siglip":
            reason = "heavy-tier SigLIP is disabled for CPU-only search"

        if reason is None:
            accepted.append(spec)
        else:
            rejected.append({"key": spec.key, "reason": reason})

    if not accepted:
        fallback = VISION_BACKBONES[DEFAULT_BACKBONE_KEY]
        accepted = [fallback]
        rejected.append(
            {
                "key": fallback.key,
                "reason": "resource fallback forced because every candidate "
                "was filtered",
                "forced": True,
            }
        )
    return accepted, rejected


def lora_configuration(model_id: str) -> dict | None:
    """Return adapter settings for a registered transformer backbone."""
    spec = next(
        (item for item in VISION_BACKBONES.values() if item.model_id == model_id),
        None,
    )
    if spec is None or not spec.supports_lora:
        return None
    return {
        "base_model": spec.model_id,
        "target_modules": list(spec.lora_target_modules),
        "rank": spec.lora_rank,
        "alpha": spec.lora_alpha,
        "weight_decay": 1e-4,
        "task_type": "FEATURE_EXTRACTION",
    }
