import os
import time
import psutil
import torch
import warnings
import threading
import json

# Suppress warnings
warnings.filterwarnings('ignore')

from cold_start import MemoryStore
from phase4_pipeline import run_single_dataset_pipeline
from data_loader import load_local_dataset

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
        
    dataset_path = r"C:\Dinesh\AutoGluon Test\bank.csv"
    modality = 'tabular'
    target_column = 'deposit'
    config = {'modality': 'tabular', 'domain': 'general', 'target_column': target_column}
    
    print("="*50)
    print("Starting MetaAutoML Tabular Pipeline (ML)")
    print(f"Dataset Path  : {dataset_path}")
    print(f"Modality      : {modality}")
    print(f"Target Column : {target_column}")
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

    print("\n🚀 Loading Local Dataset...")
    X, y, problem_type = load_local_dataset(dataset_path, target_column)
    print(f"✅ Dataset Loaded! X shape: {X.shape}, Problem Type: {problem_type}")
    
    # Execute the single dataset pipeline
    result = run_single_dataset_pipeline(
        X, y, problem_type, store, encoder, 
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
    
    # Extract eval_metrics
    eval_metrics = result.get('eval_metrics', {}) if isinstance(result, dict) else {}
    cr = eval_metrics.get('classification_report', {})
    
    # Extract specific metrics based on output logic (macro avg or weighted avg)
    accuracy = cr.get('accuracy', result.get('score', 0.0) if isinstance(result, dict) else 0.0)
    precision = cr.get('macro avg', {}).get('precision', 0.0)
    recall = cr.get('macro avg', {}).get('recall', 0.0)
    f1 = cr.get('macro avg', {}).get('f1-score', 0.0)
    cm = eval_metrics.get('confusion_matrix', [])
    
    # Formatting the confusion matrix for better terminal readability
    if cm:
        cm_str = "\n".join(["\t" + str(row) for row in cm])
    else:
        cm_str = "Not available"

    print("\n" + "="*50)
    print("FINAL TRAINING METRICS (Tabular ML)")
    print("="*50)
    print(f"Accuracy                : {accuracy:.4f}")
    print(f"Precision (Macro)       : {precision:.4f}")
    print(f"Recall (Macro)          : {recall:.4f}")
    print(f"F1 Score (Macro)        : {f1:.4f}")
    print(f"Confusion Matrix        :\n{cm_str}")
    print(f"Peak RAM (MB)           : {peak_ram:.2f}")
    print(f"Total Training Time (s) : {total_time:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
