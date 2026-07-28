# lora_config.py
LORA_REGISTRY = {
    "vision": {
        "general": {
            "base_model": "openai/clip-vit-base-patch32",
            "target_modules": ["q_proj", "v_proj"],  # CLIP uses separate q/v
            "rank": 8, "alpha": 16, "weight_decay": 1e-4,
            "task_type": "FEATURE_EXTRACTION"
        },
        "biology": {
            "base_model": "imageomics/bioclip",
            "target_modules": ["q_proj", "v_proj"],
            "rank": 16, "alpha": 32, "weight_decay": 1e-4,
            "task_type": "FEATURE_EXTRACTION"
        }
    },
    "audio": {
        "general": {
            "base_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "target_modules": ["query", "value"],  # AST attention layers
            "rank": 8, "alpha": 16, "weight_decay": 1e-4,
            "task_type": "AUDIO_CLASSIFICATION"
        }
    },
    "text": {
        "general": {
            "base_model": "sentence-transformers/all-MiniLM-L6-v2",
            "target_modules": ["q_lin", "v_lin"],  # MiniLM linear projections
            "rank": 4, "alpha": 8, "weight_decay": 1e-4,
            "task_type": "FEATURE_EXTRACTION"
        }
    }
}
