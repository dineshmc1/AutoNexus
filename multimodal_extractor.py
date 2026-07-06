import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import warnings
from dotenv import load_dotenv

load_dotenv() # Load .env file if it exists

# Optional: If the user provides a token, use it. If not, ignore the warning and use anonymous.
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
else:
    # Suppress the annoying warning for anonymous users
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Suppress annoying warnings
warnings.filterwarnings('ignore')


import hashlib
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import librosa
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import cv2

class MultiModalDataset(Dataset):
    def __init__(self, file_paths, labels, modality='vision'):
        self.file_paths = file_paths
        self.labels = labels
        self.modality = modality

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        if self.modality == 'vision':
            try:
                img = Image.open(path).convert('RGB')
                if img.size[0] < 32 or img.size[1] < 32:
                    img = img.resize((224, 224))
            except Exception as e:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            return img, label
        elif self.modality == 'audio':
            try:
                y, sr = librosa.load(path, sr=16000, duration=5.0)
                if len(y) < 1600:
                    raise ValueError("Audio too short")
                return y, label
            except Exception as e:
                return None, label
        elif self.modality == 'video':
            frames = []
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 24
                frame_count = 0
                success, frame = cap.read()
                while success:
                    if frame_count % int(fps) == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        if len(frames) >= 10: break
                    success, frame = cap.read()
                    frame_count += 1
                cap.release()
            except Exception as e:
                pass
            if not frames:
                frames = [Image.new("RGB", (224, 224), (0, 0, 0))]
            return frames, label
        elif self.modality == 'text':
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()[:5000]
            except Exception as e:
                text = ""
            return text, label

def multimodal_collate(batch):
    data = [item[0] for item in batch if item[0] is not None]
    labels = [item[1] for item in batch if item[0] is not None]
    
    if not data:
        return [], []
    return data, labels

# Process single file for CPU Parallelization
def _process_single_file(args):
    path, modality, domain = args
    # This is a stub for CPU parallelization as requested.
    # We would need to load the model inside the worker, but for simplicity we return dummy or call extractor.
    # In a full implementation, models are loaded lazily here.
    return None

class UniversalEmbedder:

    def __init__(self, device='cpu', batch_size=32, domain='general', max_files_per_class=None):
        self.device = device
        self.batch_size = batch_size
        self.domain = domain
        self.max_files_per_class = max_files_per_class
        
        # Lazy loading of models to save memory
        self.vision_model = None
        self.vision_processor = None
        self.text_model = None
        
    
    def _extract_fast(self, files, labels, modality):
        import torch
        dataset = MultiModalDataset(files, labels, modality=modality)
        
        # 🚀 MULTIPROCESSING DATALOADER 🚀
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=0 if os.name == 'nt' else 4, # Windows needs 0 inside notebooks/scripts without __main__ block guard, but we'll try 0 to be safe
            pin_memory=(self.device.type == 'cuda'),
            collate_fn=multimodal_collate
        )
        
        all_embeddings = []
        all_labels = []
        
        # Ensure model is loaded
        if modality == 'vision' and self.vision_model is None:
            from transformers import AutoProcessor, AutoModel
            from domain_registry import get_vision_model_config
            cfg = get_vision_model_config(self.domain, "clip")
            self.vision_processor = AutoProcessor.from_pretrained(cfg["model_id"])
            self.vision_model = AutoModel.from_pretrained(cfg["model_id"]).to(self.device)
            self.vision_model.eval()
        elif modality == 'audio' and not hasattr(self, "_ast_extractor"):
            from transformers import AutoFeatureExtractor, ASTForAudioClassification
            self._ast_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(self.device)
            self._ast_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            self._ast_model.eval()
        elif modality == 'video' and self.vision_model is None:
            from transformers import CLIPProcessor, CLIPModel
            self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.vision_model.eval()
        elif modality == 'text' and self.text_model is None:
            from sentence_transformers import SentenceTransformer
            self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(loader, desc=f"Fast Extracting {modality}"):
                
                # 🚨 SAFETY CHECK: Skip empty batches 🚨
                if not batch_data or len(batch_data) == 0:
                    continue
                    
                if modality == 'vision':
                    inputs = self.vision_processor(images=batch_data, return_tensors="pt")
                    if hasattr(self.vision_processor, 'pad'):
                        inputs = self.vision_processor.pad(inputs, return_tensors="pt")
                    inputs = inputs.to(self.device)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device.type=='cuda')):
                        if hasattr(self.vision_model, 'get_image_features'):
                            outputs = self.vision_model.get_image_features(**inputs)
                        else:
                            outputs = self.vision_model(**inputs)
                            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                                outputs = outputs.pooler_output
                            elif hasattr(outputs, 'last_hidden_state'):
                                if outputs.last_hidden_state.ndim == 4:
                                    outputs = outputs.last_hidden_state.mean(dim=[2, 3])
                                else:
                                    outputs = outputs.last_hidden_state.mean(dim=1)
                            else:
                                if outputs[0].ndim == 4:
                                    outputs = outputs[0].mean(dim=[2, 3])
                                else:
                                    outputs = outputs[0].mean(dim=1)
                        if isinstance(outputs, torch.Tensor): embeddings = outputs
                        elif hasattr(outputs, 'image_embeds'): embeddings = outputs.image_embeds 
                        elif hasattr(outputs, 'last_hidden_state'): embeddings = outputs.last_hidden_state[:, 0, :] 
                        else: embeddings = outputs[0]
                        all_embeddings.append(embeddings.float().cpu().numpy())
                        all_labels.extend(batch_labels)
                        
                elif modality == 'audio':
                    valid_data, valid_lbls = [], []
                    for d, l in zip(batch_data, batch_labels):
                        if d is not None and len(d) > 1600: # Ensure at least 0.1s of audio
                            valid_data.append(d)
                            valid_lbls.append(l)
                            
                    if not valid_data: 
                        continue # Skip this batch entirely if all files were bad
                        
                    # Process valid audio
                    inputs = self._ast_extractor(valid_data, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    # Move to device BEFORE autocast
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device.type=='cuda')):
                        outputs = self._ast_model(**inputs)
                        # AST logits are [batch_size, num_classes]. We use them as embeddings for now.
                        embedding = outputs.logits 
                        if embedding.ndim == 1: 
                            embedding = embedding.unsqueeze(0)
                            
                    all_embeddings.append(embedding.float().cpu().numpy())
                    all_labels.extend(valid_lbls)
                    
                elif modality == 'video':
                    # Simplistic video handling for DataLoader
                    for frames, label in zip(batch_data, batch_labels):
                        inputs = self.vision_processor(images=frames, return_tensors="pt", padding=True).to(self.device)
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device.type=='cuda')):
                            outputs = self.vision_model.get_image_features(**inputs) if hasattr(self.vision_model, 'get_image_features') else self.vision_model(**inputs)[0].mean(dim=1)
                            if isinstance(outputs, torch.Tensor): frame_features = outputs
                            elif hasattr(outputs, 'image_embeds'): frame_features = outputs.image_embeds
                            elif hasattr(outputs, 'last_hidden_state'): frame_features = outputs.last_hidden_state[:, 0, :]
                            else: frame_features = outputs[0]
                        video_embedding = torch.mean(frame_features, dim=0)
                        all_embeddings.append(video_embedding.float().cpu().numpy())
                        all_labels.append(label)
                        
                elif modality == 'text':
                    embeddings = self.text_model.encode(batch_data, batch_size=len(batch_data), show_progress_bar=False)
                    all_embeddings.append(embeddings)
                    all_labels.extend(batch_labels)
                    
        return np.vstack(all_embeddings), all_labels

    def _extract_cpu_parallel(self, files, labels, modality):
        # Stub for CPU parallelization
        print(f"🚀 [CPU Mode] Parallelizing extraction across CPU cores...")
        # Since pickling models is hard, we just fall back to fast extraction with num_workers=0 on CPU
        return self._extract_fast(files, labels, modality)
        
    def embed_directory(self, data_path, modality):
        # 1. Generate a unique hash for this dataset folder
        import hashlib
        folder_hash = hashlib.md5(data_path.encode()).hexdigest()
        cache_dir = "embedding_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{modality}_{folder_hash}.npz")
        
        # 2. Check if cache exists
        if os.path.exists(cache_path):
            print(f"⚡ [Cache Hit] Loading pre-computed {modality} embeddings from disk...")
            data = np.load(cache_path, allow_pickle=True)
            X_df = pd.DataFrame(data['X'], columns=[f"feat_{i}" for i in range(data['X'].shape[1])])
            y = pd.Series(data['y'])
            return X_df, y

        """
        Scans a directory (assuming subfolders are class labels),
        extracts embeddings, applies PCA, and returns X (DataFrame) and y (Series).
        """
        # --- NEW: AUTO-DETECT TRAIN/TEST SPLITS ---
        root_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        # If the folder contains standard ML split names, automatically dive into 'train'
        if set(['train', 'test', 'val']).intersection(set([d.lower() for d in root_dirs])):
            train_path = os.path.join(data_path, 'train')
            if os.path.exists(train_path):
                print(f"[Embedder] Detected Train/Test split structure. Automatically routing to: {train_path}")
                data_path = train_path
        # --------------------------------------------

        print(f"[UniversalEmbedder] Starting extraction for modality: {modality.upper()}")
        
        # Find all files and their class labels (subfolder names)
        files = []
        labels = []
        for root, _, filenames in os.walk(data_path):
            label = os.path.basename(root)
            if root == data_path:
                label = 'unknown'  # Files in root dir
                
            valid_filenames = []
            for f in filenames:
                # Basic filter for valid extensions
                ext = os.path.splitext(f)[1].lower()
                if modality == 'vision' and ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']: continue
                if modality == 'text' and ext not in ['.txt', '.md']: continue
                if modality == 'audio' and ext not in ['.wav', '.mp3', '.flac', '.ogg']: continue
                if modality == 'video' and ext not in ['.mp4', '.avi', '.mov', '.mkv']: continue
                
                valid_filenames.append(f)
                
            # 🚨 BENCHMARK SUBSAMPLING LOGIC 🚨
            if self.max_files_per_class and len(valid_filenames) > self.max_files_per_class:
                import random
                random.seed(42) # Reproducibility
                valid_filenames = random.sample(valid_filenames, self.max_files_per_class)
                print(f"  [Embedder] Subsampled {root} to {self.max_files_per_class} files for benchmark.")
                
            for f in valid_filenames:
                files.append(os.path.join(root, f))
                labels.append(label)
                
        if not files:
            raise ValueError(f"No valid {modality} files found in {data_path}")
            
        print(f"[UniversalEmbedder] Found {len(files)} files across {len(set(labels))} classes.")
        
        # 3. Fast Extraction using Level 1 & 2 Optimizations
        if self.device.type == 'cpu':
            X_raw, valid_labels = self._extract_cpu_parallel(files, labels, modality)
        else:
            X_raw, valid_labels = self._extract_fast(files, labels, modality)
            
        if len(X_raw) == 0:
            raise ValueError(f"No valid {modality} embeddings could be extracted. All files may be corrupted or unsupported.")

        y = pd.Series(valid_labels)
        
        print(f"[UniversalEmbedder] Raw embeddings shape: {X_raw.shape} | Labels: {len(y)}")
        
        # PCA Dimensionality Reduction (skip for small dims like MFCC ~40)
        if X_raw.shape[1] > 100:
            n_components = min(100, X_raw.shape[0])  # Can't have more components than samples
            print(f"[UniversalEmbedder] Applying PCA to reduce dimensions from {X_raw.shape[1]} to {n_components}...")
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X_raw)
            print(f"[UniversalEmbedder] PCA complete. Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
        else:
            X_reduced = X_raw
            
        # Convert to Pandas DataFrame
        feature_names = [f"feat_{i}" for i in range(X_reduced.shape[1])]
        X_df = pd.DataFrame(X_reduced, columns=feature_names)
        
        # 4. Save to cache for next time
        np.savez_compressed(cache_path, X=X_reduced, y=np.array(valid_labels))
        print(f"💾 [Cache Saved] Embeddings stored at {cache_path}")
        
        return X_df, y

    def _process_vision_batch(self, file_paths):
        from PIL import Image
        import torch
        
        if self.vision_model is None:
            from transformers import AutoProcessor, AutoModel
            from domain_registry import get_vision_model_config
            
            cfg = get_vision_model_config(self.domain, "clip")
            model_id = cfg["model_id"]
            
            print(f"\\n[UniversalEmbedder] Loading domain-specific vision model ({self.domain}): {model_id}...")
            
            # Using AutoProcessor / AutoModel to handle diverse architectures (CLIP, BEiT, TrOCR)
            self.vision_processor = AutoProcessor.from_pretrained(model_id)
            self.vision_model = AutoModel.from_pretrained(model_id).to(self.device)
            self.vision_model.eval()
            
        images = []
        for path in file_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Fallback to a blank image
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                
        # TrOCR processor expects 'images', while others might expect 'images' or 'pixel_values'
        # AutoProcessor handles this mostly, but we use 'images' explicitly
        inputs = self.vision_processor(images=images, return_tensors="pt")
        # Padding might be required depending on the exact processor
        if hasattr(self.vision_processor, 'pad'):
             inputs = self.vision_processor.pad(inputs, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.vision_model, 'get_image_features'):
                # CLIP and similar models
                outputs = self.vision_model.get_image_features(**inputs)
            else:
                # SegFormer, ResNet, ViT, etc.
                outputs = self.vision_model(**inputs)
                
                # Try to get pooled output first
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    outputs = outputs.pooler_output
                # Fallback to mean pooling of last hidden state
                elif hasattr(outputs, 'last_hidden_state'):
                    if outputs.last_hidden_state.ndim == 4:
                        outputs = outputs.last_hidden_state.mean(dim=[2, 3])
                    else:
                        outputs = outputs.last_hidden_state.mean(dim=1)
                else:
                    if outputs[0].ndim == 4:
                        outputs = outputs[0].mean(dim=[2, 3])
                    else:
                        outputs = outputs[0].mean(dim=1)
            
            # FIX: Extract the actual tensor from the output object
            if isinstance(outputs, torch.Tensor):
                image_features = outputs
            elif hasattr(outputs, 'image_embeds'):
                # Specific to CLIP Vision Model
                image_features = outputs.image_embeds 
            elif hasattr(outputs, 'last_hidden_state'):
                # Specific to standard HF Vision Transformers (ViT)
                image_features = outputs.last_hidden_state[:, 0, :] 
            else:
                # Fallback for other models
                image_features = outputs[0] 
            
        return image_features.cpu().numpy()
        
    def _process_text_batch(self, file_paths):
        if self.text_model is None:
            from sentence_transformers import SentenceTransformer
            print("\n[UniversalEmbedder] Loading SentenceTransformer...")
            self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            
        texts = []
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read()[:5000]) # Cap length for speed
            except Exception as e:
                texts.append("")
                
        embeddings = self.text_model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return embeddings

    def _process_audio_batch(self, file_paths, batch_labels):
        """Processes a batch of audio files using AST (527-dim).
        
        Skips corrupted/unsupported files entirely instead of falling back
        to MFCC — this prevents dimension mismatches in np.vstack downstream.
        Returns a tuple of (embeddings_array, valid_labels) so the caller
        can keep labels in sync with the kept embeddings.
        """
        batch_emb = []
        valid_labels = []
        
        for file_path, label in zip(file_paths, batch_labels):
            try:
                emb = self.extract_audio_embeddings_transformer(file_path)
                batch_emb.append(emb)
                valid_labels.append(label)
            except Exception as e:
                # Skip the file entirely — DO NOT fall back to MFCC,
                # or the resulting array will have mismatched dimensions.
                print(f"  [Audio] Skipping corrupted/unsupported file: {os.path.basename(file_path)} ({e})")
        
        # If the whole batch was corrupted, return empty arrays
        if not batch_emb:
            return np.array([]), []
        
        # Stack only the successful 527-dim embeddings
        return np.vstack(batch_emb), valid_labels

    def extract_audio_embeddings_transformer(self, audio_path):
        """Use Audio Spectrogram Transformer (AST) instead of MFCCs.
        
        Returns a 527-dim embedding from the AST logit space, which captures
        rich AudioSet-level acoustic semantics far beyond hand-crafted MFCCs.
        The model is lazy-loaded and cached on self to avoid repeated I/O.
        """
        import torch
        import librosa
        from transformers import AutoFeatureExtractor, ASTForAudioClassification

        _MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

        # Lazy-load and cache on the instance so we only download once per run
        if not hasattr(self, "_ast_extractor") or self._ast_extractor is None:
            print(f"\n[UniversalEmbedder] Loading AST model: {_MODEL_NAME} ...")
            self._ast_extractor = AutoFeatureExtractor.from_pretrained(_MODEL_NAME)
            self._ast_model = ASTForAudioClassification.from_pretrained(_MODEL_NAME).to(self.device)
            self._ast_model.eval()

        # Load audio — AST was trained on 16 kHz, 5-second clips
        y, sr = librosa.load(audio_path, sr=16000, duration=5)

        # Extract spectrogram features
        inputs = self._ast_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass — use logits as the embedding (527 AudioSet classes)
        with torch.no_grad():
            outputs = self._ast_model(**inputs)
            embedding = outputs.logits.squeeze().cpu().numpy()

        return embedding  # shape: (527,)

    def _process_video_batch(self, file_paths):
        import cv2
        from PIL import Image
        import torch
        
        if self.vision_model is None:
            from transformers import CLIPProcessor, CLIPModel
            print("\n[UniversalEmbedder] Loading CLIP model for Video Frames...")
            self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.vision_model.eval()
            
        batch_emb = []
        
        for path in file_paths:
            frames = []
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 24 # Fallback
                
                frame_count = 0
                success, frame = cap.read()
                
                while success:
                    # Extract 1 frame per second
                    if frame_count % int(fps) == 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        
                        # Cap at 10 frames max per video to save time/memory
                        if len(frames) >= 10:
                            break
                            
                    success, frame = cap.read()
                    frame_count += 1
                cap.release()
                
            except Exception as e:
                print(f"Error processing video {path}: {e}")
                
            if not frames:
                # Blank fallback
                batch_emb.append(np.zeros(512))
                continue
                
            # Embed all extracted frames
            inputs = self.vision_processor(images=frames, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.vision_model.get_image_features(**inputs)
                if isinstance(outputs, torch.Tensor):
                    frame_features = outputs
                elif hasattr(outputs, 'image_embeds'):
                    frame_features = outputs.image_embeds
                elif hasattr(outputs, 'last_hidden_state'):
                    frame_features = outputs.last_hidden_state[:, 0, :]
                else:
                    frame_features = outputs[0]
                
            # Average frame embeddings for the video representation
            video_embedding = torch.mean(frame_features, dim=0)
            batch_emb.append(video_embedding.cpu().numpy())
            
        return np.vstack(batch_emb)
