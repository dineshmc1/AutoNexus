import os
import torch
from peft import LoraConfig, get_peft_model
from lora_config import LORA_REGISTRY
from multimodal_extractor import MultiModalDataset, multimodal_collate
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_universal_lora(
    modality: str, 
    domain: str, 
    data_dir: str, 
    output_path: str,
    epochs: int = 5, 
    batch_size: int = 32
):
    # 1. Get config from registry
    if modality not in LORA_REGISTRY:
        print(f"⚠️ Modality '{modality}' not found in LORA_REGISTRY. Skipping LoRA training.")
        return None
    if domain not in LORA_REGISTRY[modality]:
        print(f"⚠️ Domain '{domain}' not found for modality '{modality}'. Falling back to 'general'.")
        domain = "general"
        
    cfg = LORA_REGISTRY[modality][domain]
    
    # Extract file paths and labels
    files = []
    labels = []
    if data_dir and os.path.exists(data_dir):
        for root, _, filenames in os.walk(data_dir):
            label = os.path.basename(root)
            if label.lower() == 'images':
                label = os.path.basename(os.path.dirname(root))
            if root == data_dir:
                label = 'unknown'
            for f in filenames:
                files.append(os.path.join(root, f))
                labels.append(label)
    else:
        print(f"⚠️ data_dir '{data_dir}' not found or not applicable. Skipping LoRA training.")
        return None

    # 2. Load base model + processor dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = None
    
    if modality == "vision":
        from transformers import AutoModel, AutoImageProcessor
        try:
            processor = AutoImageProcessor.from_pretrained(cfg["base_model"])
        except:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(cfg["base_model"])
        model = AutoModel.from_pretrained(cfg["base_model"]).to(device)
        DatasetClass = MultiModalDataset
        
    elif modality == "audio":
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
        processor = AutoFeatureExtractor.from_pretrained(cfg["base_model"])
        model = ASTForAudioClassification.from_pretrained(cfg["base_model"]).to(device)
        DatasetClass = MultiModalDataset
        
    elif modality == "text":
        from sentence_transformers import SentenceTransformer
        # For PEFT, we need the underlying huggingface model from sentence-transformers
        model = SentenceTransformer(cfg["base_model"]).to(device)
        processor = None
        DatasetClass = MultiModalDataset
    else:
        print(f"Modality {modality} not implemented for LoRA training.")
        return None

    dataset = DatasetClass(files, labels, modality=modality)
    
    total_samples = len(dataset)
    if total_samples > 50000:
        epochs = 2  # Prevent overfitting on massive datasets
    elif total_samples > 10000:
        epochs = 3
    else:
        epochs = 5
        
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        collate_fn=multimodal_collate,
        shuffle=True
    )

    # 3. Configure LoRA with registry params
    lora_config = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=0.1,
        bias="none",
        task_type=cfg["task_type"]
    )
    
    # For SentenceTransformers, we wrap the underlying transformer model
    if modality == "text":
        peft_model = get_peft_model(model[0].auto_model, lora_config)
        hidden_size = model[0].auto_model.config.hidden_size
    else:
        peft_model = get_peft_model(model, lora_config)
        hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "d_model", 768))
        
    peft_model.train()
    
    unique_labels = list(set(labels))
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    classifier = torch.nn.Linear(hidden_size, num_classes).to(device)
    
    optimizer = torch.optim.AdamW(list(peft_model.parameters()) + list(classifier.parameters()), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"🚀 Starting LoRA training for {modality} ({domain})...")
    # Basic training loop (since we don't have a task-specific loss here, this is a placeholder/mock loop 
    # to demonstrate integration as requested by "[Rest of training loop remains identical]")
    for epoch in range(epochs):
        peft_model.train()
        classifier.train()
        for batch_data, batch_labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            if not batch_data: continue
            
            targets = torch.tensor([label_to_id[l] for l in batch_labels]).to(device)
            
            try:
                if modality == "vision":
                    inputs = processor(images=batch_data, return_tensors="pt").to(device)
                    # Check for get_image_features first (bypasses full CLIPModel forward which expects text)
                    if hasattr(peft_model, 'get_image_features'):
                        outputs = peft_model.get_image_features(**inputs)
                    elif hasattr(peft_model, 'base_model') and hasattr(peft_model.base_model.model, 'get_image_features'):
                        outputs = peft_model.base_model.model.get_image_features(**inputs)
                    else:
                        outputs = peft_model(**inputs)
                        
                    if isinstance(outputs, torch.Tensor):
                        features = outputs
                    elif hasattr(outputs, 'image_embeds'):
                        features = outputs.image_embeds
                    else:
                        features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]
                elif modality == "audio":
                    inputs = processor(batch_data, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
                    outputs = peft_model(**inputs)
                    features = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state[:, 0, :]
                elif modality == "text":
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
                    inputs = tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt").to(device)
                    outputs = peft_model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :]
                    
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                    
                # Fix hidden size mismatch by lazily initializing classifier if needed
                if classifier.in_features != features.size(1):
                    classifier = torch.nn.Linear(features.size(1), num_classes).to(device)
                    optimizer = torch.optim.AdamW(list(peft_model.parameters()) + list(classifier.parameters()), lr=1e-4)
                    
                logits = classifier(features)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
            except Exception as e:
                tqdm.write(f"Error in batch: {e}")
                continue
            
    os.makedirs(output_path, exist_ok=True)
    peft_model.save_pretrained(output_path)
    if processor is not None:
        processor.save_pretrained(output_path)
    print(f"✅ LoRA Adapter saved to {output_path}")
    return output_path
