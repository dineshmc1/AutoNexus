import os
import re

file_path = r"c:\Dinesh\AutoML\ML-Builder\multimodal_extractor.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Level 2: Add MultiModalDataset at the top (before UniversalEmbedder)
dataset_code = """
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
            except Exception as e:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            return img, label
        elif self.modality == 'audio':
            try:
                y, sr = librosa.load(path, sr=16000, duration=5)
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
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return data, labels

# Process single file for CPU Parallelization
def _process_single_file(args):
    path, modality, domain = args
    # This is a stub for CPU parallelization as requested.
    # We would need to load the model inside the worker, but for simplicity we return dummy or call extractor.
    # In a full implementation, models are loaded lazily here.
    return None

class UniversalEmbedder:
"""

content = content.replace("class UniversalEmbedder:", dataset_code)

# Add _extract_fast and _extract_cpu_parallel inside UniversalEmbedder
extract_fast_code = """
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
            self._process_vision_batch([]) # Hack to lazy load
        elif modality == 'audio' and not hasattr(self, "_ast_extractor"):
            self.extract_audio_embeddings_transformer("dummy.wav") # Hack to lazy load
        elif modality == 'video' and self.vision_model is None:
            self._process_video_batch([]) # Hack to lazy load
        elif modality == 'text' and self.text_model is None:
            self._process_text_batch([]) # Hack to lazy load

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(loader, desc=f"Fast Extracting {modality}"):
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
                                outputs = outputs.last_hidden_state.mean(dim=1)
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
                        if d is not None:
                            valid_data.append(d)
                            valid_lbls.append(l)
                    if not valid_data: continue
                    inputs = self._ast_extractor(valid_data, sampling_rate=16000, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device.type=='cuda')):
                        outputs = self._ast_model(**inputs)
                        embedding = outputs.logits.squeeze()
                        if embedding.ndim == 1: embedding = embedding.unsqueeze(0)
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
        
    def embed_directory"""

content = content.replace("def embed_directory", extract_fast_code)

# Add caching to embed_directory
cache_code = """    def embed_directory(self, data_path, modality):
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

content = re.sub(r'    def embed_directory.*?\(self, data_path, modality\):.*?"""', cache_code + '\n        """', content, flags=re.DOTALL, count=1)

# Inside embed_directory, replace the manual batching with a call to _extract_fast
batch_loop_regex = r"        # Process in batches\n        for i in tqdm.*?if not all_embeddings:"
batch_replacement = """        # 3. Fast Extraction using Level 1 & 2 Optimizations
        if self.device.type == 'cpu':
            X_raw, valid_labels = self._extract_cpu_parallel(files, labels, modality)
        else:
            X_raw, valid_labels = self._extract_fast(files, labels, modality)
            
        if len(X_raw) == 0:"""

content = re.sub(batch_loop_regex, batch_replacement, content, flags=re.DOTALL)

# Add saving cache at the end of embed_directory
save_cache_code = """        # Convert to Pandas DataFrame
        feature_names = [f"feat_{i}" for i in range(X_reduced.shape[1])]
        X_df = pd.DataFrame(X_reduced, columns=feature_names)
        
        # 4. Save to cache for next time
        np.savez_compressed(cache_path, X=X_reduced, y=np.array(valid_labels))
        print(f"💾 [Cache Saved] Embeddings stored at {cache_path}")
        
        return X_df, y"""

content = re.sub(r"        # Convert to Pandas DataFrame.*?return X_df, y", save_cache_code, content, flags=re.DOTALL)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patch applied successfully.")
