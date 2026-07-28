"""
resource_manager.py
Adaptive Resource-Aware Engine for MLBuilder pipeline.
Detects dataset constraints, prevents feature explosions, and dictates
safe preprocessing and modelling strategies before execution.
"""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd


class ResourceManager:
    """Analyzes datasets to prevent OOM errors and dynamically adapt data preparation."""
    
    def __init__(
        self,
        max_onehot_features: int = 5000,
        low_cardinality_threshold: float = 0.01,
        high_cardinality_threshold: float = 0.1,
        high_cardinality_strategy: str = "hash",
        small_size_threshold: int = 50000,
        medium_size_threshold: int = 200000,
    ) -> None:
        self.max_onehot_features = max_onehot_features
        self.low_cardinality_threshold = low_cardinality_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_strategy = high_cardinality_strategy
        self.small_size_threshold = small_size_threshold
        self.medium_size_threshold = medium_size_threshold

    def analyze(self, X: pd.DataFrame, problem_type: str) -> Dict[str, Any]:
        """Analyze dataset and return a resource decision configuration."""
        n_rows, _ = X.shape

        print("\n" + "─" * 72)
        print("  RESOURCE-AWARE ENGINE ANALYSIS")
        print("─" * 72)

        # 1. Dataset size classification and basic rules
        if n_rows < self.small_size_threshold:
            size_category = "small"
            models_to_run = ["logistic", "et_clf", "gb", "lgbm_clf", "xgb_clf"] if problem_type == "classification" else ["ridge", "rf_reg", "gb_reg", "lgbm_reg", "xgb_reg"]
            enable_fe = True
            fe_level = "full"
            interaction_k = 5
        elif n_rows <= self.medium_size_threshold:
            size_category = "medium"
            models_to_run = ["logistic", "et_clf", "lgbm_clf", "xgb_clf"] if problem_type == "classification" else ["ridge", "rf_reg", "lgbm_reg", "xgb_reg"]
            enable_fe = True
            fe_level = "medium"
            interaction_k = 2
        else:
            size_category = "large"
            # For large datasets, restrict to scalable models.
            models_to_run = ["logistic", "lgbm_clf", "xgb_clf"] if problem_type == "classification" else ["ridge", "lgbm_reg", "xgb_reg"]
            enable_fe = True  # Enable FE but restrict to safe level
            fe_level = "light"
            interaction_k = 0
            
        print(f"[ResourceManager] Dataset size: {size_category} ({n_rows} rows)")
        if size_category == 'large':
            print("[ResourceManager] Heavy feature engineering (interactions/polynomials) restricted due to memory constraints")
            
        # 2. Encoding Strategy Decisions
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        encoding_strategies: Dict[str, str] = {}
        target_cardinalities: Dict[str, int] = {}
        onehot_count = 0

        for col in cat_cols:
            n_unique = X[col].nunique()
            ratio = n_unique / n_rows if n_rows > 0 else 0

            if ratio < self.low_cardinality_threshold:
                strategy = "onehot"
                target_cardinalities[col] = n_unique
                onehot_count += n_unique
            elif ratio <= self.high_cardinality_threshold:
                strategy = "frequency"
                print(f"[ResourceManager] Medium cardinality detected in '{col}' (ratio={ratio:.3f}) → assigned to frequency encoding")
            else:
                strategy = self.high_cardinality_strategy
                print(f"[ResourceManager] High cardinality detected in '{col}' (ratio={ratio:.3f}) → assigned to {strategy} encoding")
                
            encoding_strategies[col] = strategy

        # Apply Hard Cap to OneHot mapping to prevent Feature Explosion
        if onehot_count > self.max_onehot_features:
            print(f"[ResourceManager] OneHotEncoder features ({onehot_count}) exceed safe limit ({self.max_onehot_features})!")
            
            # Sort low-cardinality features identified as onehot descendingly by unique categories
            sorted_cols = sorted(target_cardinalities.items(), key=lambda x: x[1], reverse=True)
            
            for col, n_unique in sorted_cols:
                if onehot_count <= self.max_onehot_features:
                    break
                encoding_strategies[col] = "frequency"
                onehot_count -= n_unique
                print(f"  [Fallback] '{col}' shifted to frequency encoding (saved {n_unique} features)")
                
        decisions = {
            "size_category": size_category,
            "models_to_run": models_to_run,
            "enable_fe": enable_fe,
            "fe_level": fe_level,
            "interaction_k": interaction_k,
            "encoding_strategies": encoding_strategies,
            "onehot_features_sum": onehot_count,
        }
        
        print(f"[ResourceManager] Final Directives: Models={models_to_run} | Max Interactions={interaction_k} | Expected sparse one-hot extensions={onehot_count}")
        return decisions
