"""
feature_processing.py - ULTIMATE MERGED VERSION
Keeps original FeatureHasher, SelectKBest, and dynamic scaling.
Adds AutoDL Bypass to preserve multi-modal PCA embeddings.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer


def hash_features(df):
    """Convert categorical values to named tokens for FeatureHasher."""
    from sklearn.feature_extraction import FeatureHasher

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=[str(i) for i in range(df.shape[1])])
    str_df = df.astype(str).fillna("missing")
    for column in str_df.columns:
        str_df[column] = column + "=" + str_df[column]
    return FeatureHasher(
        n_features=2048, input_type="string"
    ).transform(str_df.values)


def detect_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessor(
    X: pd.DataFrame,
    scaler_map: Optional[Dict[str, str]] = None,
    encoding_map: Optional[Dict[str, str]] = None,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    
    numeric_cols, categorical_cols = detect_column_types(X)

    # 🚨 AUTO-DL BYPASS 🚨
    # If the data is purely numeric and has no categorical columns, 
    # it is likely a pre-computed embedding (e.g., 100D PCA from CLIP/AST).
    # We MUST bypass the ColumnTransformer to avoid destroying the embedding geometry.
    if len(categorical_cols) == 0 and len(numeric_cols) > 0:
        # Check if it looks like an embedding (e.g., columns named like 'pca_0', 'pca_1' or just dense floats)
        # A safe heuristic: if there are > 50 numeric columns and 0 categorical, it's likely an embedding.
        if len(numeric_cols) >= 50: 
            print(f"[Features] Detected dense numeric embeddings ({len(numeric_cols)}D). Bypassing ColumnTransformer to preserve geometry.")
            # FunctionTransformer with func=None is a pickle-safe identity
            # transform. A local lambda would prevent saving the final model.
            preprocessor = FunctionTransformer(validate=False)
            return preprocessor, numeric_cols, categorical_cols

    transformers = []
    
    # Numeric Pipeline (Keeps your adaptive Standard vs Robust logic)
    if numeric_cols:
        if scaler_map:
            std_cols = [c for c in numeric_cols if scaler_map.get(c, "standard") == "standard"]
            rob_cols = [c for c in numeric_cols if scaler_map.get(c) == "robust"]
            if std_cols:
                transformers.append(("num_std", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), std_cols))
            if rob_cols:
                transformers.append(("num_rob", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), rob_cols))
        else:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))

    # Categorical Pipeline (Keeps your OneHot + FeatureHasher logic)
    if categorical_cols:
        onehot_cols = [c for c in categorical_cols if not encoding_map or encoding_map.get(c, 'onehot') == 'onehot']
        hash_cols = [c for c in categorical_cols if encoding_map and encoding_map.get(c) == 'hash']
        
        if onehot_cols:
            transformers.append(("cat_ohe", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), onehot_cols))
            
        if hash_cols:
            transformers.append(("cat_hash", FunctionTransformer(hash_features, validate=False), hash_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    preprocessor.set_output(transform="pandas")
    
    print(f"[Features] {len(numeric_cols)} numeric, {len(categorical_cols)} categorical column(s).")
    return preprocessor, numeric_cols, categorical_cols

def select_features(
    X: np.ndarray, y: np.ndarray, problem_type: str, method: str = "mutual_info", k: int = 10,
) -> Tuple[np.ndarray, SelectKBest]:
    k = min(k, X.shape[1])
    score_func = mutual_info_classif if problem_type == "classification" else mutual_info_regression
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    print(f"[Features] Selected top {k} features via {method} (from {X.shape[1]}).")
    return X_selected, selector
