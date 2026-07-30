"""Backward-compatible LoRA defaults backed by the vision registry."""

from vision_backbones import DEFAULT_BACKBONE_KEY, VISION_BACKBONES

_DEFAULT = VISION_BACKBONES[DEFAULT_BACKBONE_KEY]
VISION_MODEL_ID = _DEFAULT.model_id

LORA_REGISTRY = {
    "vision": {
        "general": {
            "base_model": _DEFAULT.model_id,
            "target_modules": list(_DEFAULT.lora_target_modules),
            "rank": _DEFAULT.lora_rank,
            "alpha": _DEFAULT.lora_alpha,
            "weight_decay": 1e-4,
            "task_type": "FEATURE_EXTRACTION",
        }
    }
}
