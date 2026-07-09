# domain_registry.py
DOMAIN_REGISTRY = {
    "general": {
        "clip": {
            "model_id": "openai/clip-vit-base-patch32",
            "embed_dim": 512,
            "description": "General-purpose vision-language model. Best for natural images, objects, and scenes."
        },
        "resnet": {
            "model_id": "resnet50",
            "embed_dim": 2048,
            "description": "Classic CNN. Strong on textures, shapes, and general pattern recognition."
        }
    },
    "biology": {
        "clip": {
            "model_id": "imageomics/bioclip",
            "embed_dim": 512,
            "description": "Fine-tuned CLIP for biological imagery. Excels at microscopy, species identification, and cellular structures."
        },
        "resnet": {
            "model_id": "microsoft/beit-base-patch16-224-pt22k-ft22k",
            "embed_dim": 768,
            "description": "BEiT transformer pre-trained on medical/biological datasets. Superior for fine-grained biological features."
        },
        "dinov2": {
            "model_id": "facebook/dinov2-base",
            "embed_dim": 768,
            "description": "Self-supervised vision transformer. Excellent for dense features and organism part detection."
        },
        "siglip": {
            "model_id": "google/siglip-base-patch16-224",
            "embed_dim": 768,
            "description": "Sigmoid loss CLIP. Highly efficient for multi-lingual and broad biological concepts."
        }
    },
    "remote_sensing": {
        "clip": {
            "model_id": "nvidia/mit-b0",
            "embed_dim": 320,
            "description": "Mix Vision Transformer optimized for aerial/satellite imagery and spatial patterns."
        },
        "resnet": {
            "model_id": "resnet50",
            "embed_dim": 2048,
            "description": "Fallback for general geospatial classification tasks."
        }
    },
    "documents": {
        "clip": {
            "model_id": "microsoft/trocr-base-printed",
            "embed_dim": 768,
            "description": "TrOCR model. Optimized for scanned documents, receipts, and printed text recognition."
        },
        "resnet": {
            "model_id": "resnet50",
            "embed_dim": 2048,
            "description": "Fallback for document layout and stamp/signature detection."
        }
    }
}

def get_vision_model_config(domain: str = "general", architecture: str = "clip") -> dict:
    """Safely retrieves model config. Falls back to 'general' if domain is unknown."""
    domain = domain.lower().strip()
    if domain not in DOMAIN_REGISTRY:
        print(f"⚠️ Domain '{domain}' not found. Falling back to 'general'.")
        domain = "general"
        
    if architecture not in DOMAIN_REGISTRY[domain]:
        architecture = "clip" # Default architecture
        
    return DOMAIN_REGISTRY[domain][architecture]
