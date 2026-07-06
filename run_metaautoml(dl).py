import os
import time
import psutil
import torch
import warnings
import threading

# Suppress warnings
warnings.filterwarnings('ignore')

from cold_start import MemoryStore
from phase4_pipeline import run_single_dataset_pipeline
from multimodal_extractor import UniversalEmbedder

def get_ram_usage():
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    start_time = time.time()
    
    # Thread to monitor peak RAM usage accurately
    stop_monitor = False
    peak_ram = 0.0
    
    def monitor_ram():
        nonlocal peak_ram
        while not stop_monitor:
            ram = get_ram_usage()
            if ram > peak_ram:
                peak_ram = ram
            time.sleep(0.05)
            
    monitor_thread = threading.Thread(target=monitor_ram, daemon=True)
    monitor_thread.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    dataset_path = r"C:\Dinesh\AutoGluon Test\EuroSAT"
    modality = 'vision'
    config = {'modality': 'vision', 'domain': 'remote_sensing'}
    
    print("="*50)
    print("Starting MetaAutoML Custom Pipeline")
    print(f"Dataset Path  : {dataset_path}")
    print(f"Modality      : {modality}")
    print(f"Domain        : {config['domain']}")
    print(f"Vision Model  : nvidia/mit-b0 (via remote_sensing domain)")
    print("="*50)
    
    MEMORY_INDEX_PATH = "memory_store.faiss"
    MEMORY_META_PATH  = "memory_store.pkl"
    store = MemoryStore()
    if os.path.exists(MEMORY_INDEX_PATH):
        loaded = store.load_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
        print(f"✅ Loaded {loaded} existing records from disk")
    else:
        print("⚠️ Memory store not found. Creating a new empty store.")
        
    print("🤖 Loading pre-trained Task Encoder...")
    from task_encoder import SiameseEncoder, TaskEncoderConfig
    cfg = TaskEncoderConfig(input_dim=10, hidden_dim=64, output_dim=32)
    encoder = SiameseEncoder(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim).to(device)
    encoder_path = cfg.encoder_save_path
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.eval()

    print("\n🚀 Extracting Embeddings...")
    embedder = UniversalEmbedder(device=device, batch_size=32, domain=config['domain'])
    X, y = embedder.embed_directory(dataset_path, modality)
    
    print(f"✅ Embedding Extraction Complete! Shape: {X.shape}")
    
    # Execute the single dataset pipeline
    result = run_single_dataset_pipeline(
        X, y, 'classification', store, encoder, 
        did=os.path.basename(dataset_path), 
        validate=False,
        modality=modality,
        config=config
    )
    
    # Stop RAM monitoring
    stop_monitor = True
    monitor_thread.join(timeout=1.0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    # Formatting output for terminal
    accuracy = result.get('score', 0.0) if isinstance(result, dict) else 0.0

    print("\n" + "="*50)
    print("FINAL TRAINING METRICS")
    print("="*50)
    print(f"Accuracy                : {accuracy:.4f}")
    print(f"Peak RAM (MB)           : {peak_ram:.2f}")
    print(f"Peak VRam (MB)          : {peak_vram:.2f}")
    print(f"Total Training Time (s) : {total_time:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
