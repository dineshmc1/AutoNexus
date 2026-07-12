import time
import wandb

try:
    import cudf
    import cuml
    from cuml.model_selection import train_test_split as gpu_train_test_split
    GPU_TABULAR_AVAILABLE = True
except ImportError:
    import pandas as pd
    from sklearn.model_selection import train_test_split as cpu_train_test_split
    GPU_TABULAR_AVAILABLE = False

def load_and_preprocess_tabular(filepath, test_size=0.2):
    start_time = time.time()
    
    if GPU_TABULAR_AVAILABLE:
        print("🚀 Using RAPIDS (cudf/cuml) for GPU-accelerated tabular loading...")
        df = cudf.read_csv(filepath)
        # Drop NaNs natively on GPU
        df = df.dropna() 
        
        X = df.drop('target', axis=1) # Replace 'target' with your actual label column
        y = df['target']
        
        X_train, X_test, y_train, y_test = gpu_train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Convert back to numpy for XGBoost/LightGBM compatibility
        X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()
        X_test_np, y_test_np = X_test.to_numpy(), y_test.to_numpy()
    else:
        print("⚠️ RAPIDS not found. Falling back to CPU pandas/sklearn...")
        df = pd.read_csv(filepath).dropna()
        X = df.drop('target', axis=1)
        y = df['target']
        X_train_np, X_test_np, y_train_np, y_test_np = cpu_train_test_split(X, y, test_size=test_size, random_state=42)

    elapsed = time.time() - start_time
    
    # Log the preprocessing speedup to W&B
    wandb.log({
        "preprocessing_backend": "rapids_gpu" if GPU_TABULAR_AVAILABLE else "pandas_cpu",
        "preprocessing_time_seconds": elapsed,
        "dataset_rows": len(df)
    })
    
    return X_train_np, X_test_np, y_train_np, y_test_np
