import openml
import pandas as pd
import numpy as np

def profile_dataset(dataset_id, X, y, problem_type):
    """Gathers dataset context and health metrics for the LLM."""
    context = {"dataset_id": dataset_id, "problem_type": problem_type}
    
    # 1. Fetch OpenML Description (if available)
    if isinstance(dataset_id, int) or (isinstance(dataset_id, str) and dataset_id.isdigit()):
        try:
            ds = openml.datasets.get_dataset(int(dataset_id), download_data=False)
            context["description"] = ds.description[:500] if ds.description else "No description provided."
            context["domain"] = ds.creator if ds.creator else "Unknown"
        except:
            context["description"] = "Inferred from column names."
            context["domain"] = "Unknown"
    else:
        context["description"] = "Inferred from column names."
        context["domain"] = "Unknown"
        
    # 2. Infer Domain from Columns (Fallback)
    context["sample_columns"] = list(X.columns[:10]) if hasattr(X, 'columns') else [f"feature_{i}" for i in range(min(10, X.shape[1]))]
    
    # 3. Data Health Metrics
    if hasattr(X, 'isnull'):
        missing_pct = X.isnull().mean().mean() * 100
    else:
        missing_pct = np.isnan(X).mean() * 100
    context["missing_data_pct"] = round(float(missing_pct), 2)
    context["num_features"] = X.shape[1]
    context["num_samples"] = X.shape[0]
    
    # 4. Class Imbalance (for classification)
    if problem_type == 'classification':
        class_counts = pd.Series(y).value_counts(normalize=True)
        context["class_distribution"] = {str(k): round(float(v), 3) for k, v in class_counts.items()}
        context["imbalance_ratio"] = round(float(class_counts.max() / class_counts.min()), 2) if class_counts.min() > 0 else "Infinite"
        
    return context
