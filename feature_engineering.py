"""
feature_engineering.py - ULTIMATE MERGED VERSION
Combines original robust logic (Datetime, Polynomials, Ratios) with 
God-Tier upgrades (Text NLP, Yeo-Johnson, Correlation Drop, Downcasting).
"""
from __future__ import annotations
import warnings
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold

class FeatureEngineer:
    def __init__(
        self,
        *,
        cardinality_threshold: float = 0.05,
        rare_threshold: float = 0.01,       # NEW: Group categories < 1%
        skew_threshold: float = 1.0,
        outlier_strategy: str = "cap",
        encoding_strategy: str = "target", 
        interaction_features: int = 5,
        corr_threshold: float = 0.95,       # NEW: Drop > 95% correlated
        enable_ratios: bool = True,
        random_state: int = 42,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.rare_threshold = rare_threshold
        self.skew_threshold = skew_threshold
        self.outlier_strategy = outlier_strategy
        self.encoding_strategy = encoding_strategy
        self.interaction_features = interaction_features
        self.corr_threshold = corr_threshold
        self.enable_ratios = enable_ratios
        self.random_state = random_state

        # Fitted state
        self._fitted = False
        self._zero_var_cols: List[str] = []
        self._text_cols: List[str] = []     # NEW
        self._datetime_cols: List[str] = []
        self._datetime_features: Dict[str, List[str]] = {}
        self._high_card_cols: List[str] = []
        self._encoding_maps: Dict[str, Dict[Any, float]] = {}
        self._encoding_fallbacks: Dict[str, float] = {}
        self._yeojohnson_transformer: Optional[PowerTransformer] = None
        self._skew_cols: List[str] = []
        self._outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self._scaler_map: Dict[str, str] = {}
        self._interaction_pairs: List[Tuple[str, str]] = []
        self._ratio_pairs: List[Tuple[str, str]] = []
        self._poly_cols: List[str] = []
        self._corr_drop_cols: List[str] = [] # NEW
        self._feature_types: Dict[str, str] = {}
        self.log: List[str] = []

    def _sanitize_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in X.columns]
        return X

    def _log(self, message: str) -> None:
        self.log.append(message)
        print(f"[FE] {message}")

    # ==========================================
    # PUBLIC API
    # ==========================================
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, problem_type: str = "classification") -> pd.DataFrame:
        self.log = []
        X = X.copy()
        self._log("Starting Ultimate Feature Engineering...")

        # Tag original features
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]): self._feature_types[col] = "numeric_original"
            elif pd.api.types.is_datetime64_any_dtype(X[col]): self._feature_types[col] = "datetime_original"
            else: self._feature_types[col] = "categorical_original"

        X = self._fit_zero_variance(X)
        X = self._fit_text_features(X)         # NEW: Extract NLP stats from text columns
        X = self._fit_datetime(X)              # ORIGINAL: Cyclical datetime extraction
        X = self._fit_rare_categories(X)       # NEW: Bucket rare categories
        X = self._fit_high_cardinality(X, y, problem_type)
        X = self._fit_yeojohnson(X)            # NEW: Better than log1p, handles negatives
        X = self._fit_outliers(X)
        self._fit_adaptive_scaling(X)          # ORIGINAL: Standard vs Robust decision
        X = self._fit_drop_correlated(X)       # NEW: Remove multicollinearity
        X = self._fit_interactions(X, y)
        if self.enable_ratios: X = self._fit_ratios(X, y)
        X = self._fit_polynomials(X)           # ORIGINAL: Squared terms
        X = self._downcast_numerics(X)         # NEW: Optimize RAM and Speed
        
        X = self._sanitize_columns(X)
        self._fitted = True
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted: raise RuntimeError("FeatureEngineer not fitted.")
        X = X.copy()
        
        X = self._apply_zero_variance(X)
        X = self._apply_text_features(X)
        X = self._apply_datetime(X)
        X = self._apply_high_cardinality(X)
        X = self._apply_yeojohnson(X)
        X = self._apply_outliers(X)
        X = self._apply_drop_correlated(X)
        X = self._apply_interactions(X)
        if self.enable_ratios: X = self._apply_ratios(X)
        X = self._apply_polynomials(X)
        X = self._downcast_numerics(X)
        
        return self._sanitize_columns(X)

    def get_scalers(self) -> Dict[str, str]:
        return self._scaler_map.copy()

    # ==========================================
    # NEW GOD-TIER METHODS
    # ==========================================
    def _fit_text_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self._text_cols = []
        for col in X.select_dtypes(include=['object']).columns:
            if self._feature_types.get(col) == "categorical_original":
                avg_len = X[col].dropna().astype(str).str.len().mean()
                if avg_len > 30:  # If average length > 30 chars, it's text, not a category
                    self._text_cols.append(col)
        
        for col in self._text_cols:
            text_series = X[col].fillna('').astype(str)
            X[f'{col}_length'] = text_series.str.len()
            X[f'{col}_word_count'] = text_series.str.split().str.len()
            X = X.drop(columns=[col])
            self._feature_types[f'{col}_length'] = "text_derived"
            self._log(f"Extracted NLP features from text column: '{col}'")
        return X

    def _apply_text_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._text_cols:
            if col in X.columns:
                text_series = X[col].fillna('').astype(str)
                X[f'{col}_length'] = text_series.str.len()
                X[f'{col}_word_count'] = text_series.str.split().str.len()
                X = X.drop(columns=[col])
        return X

    def _fit_rare_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [c for c, t in self._feature_types.items() if t == "categorical_original" and c in X.columns]
        for col in cat_cols:
            counts = X[col].value_counts(normalize=True)
            rare_cats = counts[counts < self.rare_threshold].index
            if len(rare_cats) > 0:
                X[col] = X[col].where(~X[col].isin(rare_cats), '_RARE')
                self._log(f"Grouped {len(rare_cats)} rare categories in '{col}' into '_RARE'")
        return X

    def _fit_yeojohnson(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = [c for c, t in self._feature_types.items() if t == "numeric_original" and c in X.columns]
        self._skew_cols = []
        
        for col in numeric:
            if abs(X[col].skew()) > self.skew_threshold:
                self._skew_cols.append(col)
        
        if self._skew_cols:
            self._yeojohnson_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            X[self._skew_cols] = self._yeojohnson_transformer.fit_transform(X[self._skew_cols])
            self._log(f"Applied Yeo-Johnson transform to {len(self._skew_cols)} skewed features.")
        return X

    def _apply_yeojohnson(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._yeojohnson_transformer is not None and self._skew_cols:
            valid_cols = [c for c in self._skew_cols if c in X.columns]
            if valid_cols:
                X[valid_cols] = self._yeojohnson_transformer.transform(X[valid_cols])
        return X

    def _fit_drop_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.select_dtypes(include=['number'])
        if numeric.shape[1] < 2: return X
        
        corr_matrix = numeric.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self._corr_drop_cols = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        
        if self._corr_drop_cols:
            X = X.drop(columns=self._corr_drop_cols)
            self._log(f"Dropped {len(self._corr_drop_cols)} highly correlated features (>{self.corr_threshold})")
        return X

    def _apply_drop_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        corr_drop = [c for c in self._corr_drop_cols if c in X.columns]
        return X.drop(columns=corr_drop) if corr_drop else X

    def _downcast_numerics(self, X: pd.DataFrame) -> pd.DataFrame:
        numerics = X.select_dtypes(include=['number'])
        for col in numerics.columns:
            if pd.api.types.is_float_dtype(X[col]):
                # Ratios and polynomial terms can exceed float32. Keep float64
                # and let the downstream imputer handle undefined values.
                X[col] = X[col].replace([np.inf, -np.inf], np.nan).astype(np.float64)
            elif X[col].dtype == np.int64:
                X[col] = pd.to_numeric(X[col], downcast='integer')
        return X

    # ==========================================
    # ORIGINAL METHODS (Kept & Preserved)
    # ==========================================
    def _fit_zero_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        variances = X.select_dtypes(include="number").var()
        self._zero_var_cols = variances[variances < 1e-6].index.tolist()
        return X.drop(columns=self._zero_var_cols) if self._zero_var_cols else X

    def _apply_zero_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self._zero_var_cols if c in X.columns]
        return X.drop(columns=cols_to_drop) if cols_to_drop else X

    def _fit_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        self._datetime_cols = [c for c, t in self._feature_types.items() if t == "datetime_original" and c in X.columns]
        for col in self._datetime_cols:
            X, new_cols = self._extract_dt_features(X, col)
            self._datetime_features[col] = new_cols
        return X

    def _apply_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._datetime_cols:
            if col in X.columns: X, _ = self._extract_dt_features(X, col)
        return X

    def _extract_dt_features(self, X: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, List[str]]:
        dt = pd.to_datetime(X[col], errors="coerce")
        new_cols = []
        for attr, name in [("year", f"{col}_year"), ("month", f"{col}_month"), ("day", f"{col}_day"), ("weekday", f"{col}_weekday")]:
            X[name] = getattr(dt.dt, attr); new_cols.append(name)
        X[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12); new_cols.append(f"{col}_month_sin")
        X[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12); new_cols.append(f"{col}_month_cos")
        X = X.drop(columns=[col])
        return X, new_cols

    def _fit_high_cardinality(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> pd.DataFrame:
        cat_cols = [c for c, t in self._feature_types.items() if t == "categorical_original" and c in X.columns]
        for col in cat_cols:
            unique_ratio = X[col].nunique() / len(X) if len(X) > 0 else 0
            if unique_ratio > self.cardinality_threshold:
                self._high_card_cols.append(col)
                if problem_type == "classification" and y.nunique() > 2:
                    # A scalar mean of arbitrary multiclass IDs is not a
                    # meaningful target encoding. Frequency is regularized
                    # and preserves no artificial class ordering.
                    frequencies = X[col].value_counts(normalize=True).to_dict()
                    fallback = 1.0 / max(len(X), 1)
                    self._encoding_maps[col] = frequencies
                    self._encoding_fallbacks[col] = fallback
                    X[col] = X[col].map(frequencies).fillna(fallback)
                else:
                    oof, full_map, fallback = self._target_encode_fit(
                        X[col], y
                    )
                    self._encoding_maps[col] = full_map
                    self._encoding_fallbacks[col] = fallback
                    X[col] = oof
                self._feature_types[col] = "categorical_encoded"
        return X

    def _apply_high_cardinality(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._high_card_cols:
            if col in X.columns:
                enc_map = self._encoding_maps[col]
                fallback = self._encoding_fallbacks[col]
                X[col] = X[col].map(enc_map).fillna(fallback).astype(float)
        return X

    def _target_encode_fit(
        self,
        series: pd.Series,
        y: pd.Series,
        n_splits: int = 5,
        smoothing: float = 20.0,
    ) -> Tuple[pd.Series, Dict[Any, float], float]:
        """Return leakage-safe OOF values and a smoothed inference mapping."""
        series = series.reset_index(drop=True)
        y = y.reset_index(drop=True).astype(float)
        encoding = pd.Series(np.nan, index=series.index, dtype=float)
        splits = min(n_splits, len(series))
        if splits < 2:
            fallback = float(y.mean())
            return (
                pd.Series(fallback, index=series.index),
                {},
                fallback,
            )
        kf = KFold(
            n_splits=splits, shuffle=True, random_state=self.random_state
        )
        global_mean = float(y.mean())
        for train_idx, val_idx in kf.split(series):
            train_series, train_y = series.iloc[train_idx], y.iloc[train_idx]
            counts = train_series.value_counts()
            means = train_y.groupby(train_series).mean()
            lambda_w = counts / (counts + smoothing)
            smoothed_means = lambda_w * means + (1 - lambda_w) * global_mean
            encoding.iloc[val_idx] = (
                series.iloc[val_idx].map(smoothed_means).fillna(global_mean)
            )

        full_counts = series.value_counts()
        full_means = y.groupby(series).mean()
        full_weights = full_counts / (full_counts + smoothing)
        full_map = (
            full_weights * full_means
            + (1 - full_weights) * global_mean
        ).to_dict()
        return encoding.fillna(global_mean), full_map, global_mean

    def _fit_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_strategy == "none": return X
        for col in X.select_dtypes(include="number").columns:
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                if ((X[col] < lower) | (X[col] > upper)).sum() > 0:
                    self._outlier_bounds[col] = (lower, upper)
                    X[col] = X[col].clip(lower, upper)
        return X

    def _apply_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self._outlier_bounds.items():
            if col in X.columns: X[col] = X[col].clip(lower, upper)
        return X

    def _fit_adaptive_scaling(self, X: pd.DataFrame) -> None:
        numeric = X.select_dtypes(include="number").columns.tolist()
        for col in numeric:
            values = X[col].dropna()
            if len(values) < 20: self._scaler_map[col] = "standard"; continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_value = stats.normaltest(values)
            self._scaler_map[col] = "standard" if p_value > 0.05 else "robust"

    def _fit_interactions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if self.interaction_features <= 0: return X
        numeric = X.select_dtypes(include="number").columns.tolist()
        if len(numeric) < 2: return X
        corrs = X[numeric].corrwith(y.astype(float)).abs().sort_values(ascending=False)
        top_cols = corrs.head(self.interaction_features).index.tolist()
        for i, c1 in enumerate(top_cols):
            for c2 in top_cols[i + 1:]:
                X[f"{c1}__x__{c2}"] = X[c1] * X[c2]
                self._interaction_pairs.append((c1, c2))
        return X

    def _apply_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        for c1, c2 in self._interaction_pairs:
            if c1 in X.columns and c2 in X.columns: X[f"{c1}__x__{c2}"] = X[c1] * X[c2]
        return X

    def _fit_ratios(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        k = min(self.interaction_features, 3)
        if k <= 1: return X
        numeric = X.select_dtypes(include="number").columns.tolist()
        if len(numeric) < 2: return X
        corrs = X[numeric].corrwith(y.astype(float)).abs().sort_values(ascending=False)
        top_cols = corrs.head(k).index.tolist()
        for i, c1 in enumerate(top_cols):
            for c2 in top_cols:
                if c1 == c2: continue
                X[f"{c1}__div__{c2}"] = X[c1] / (X[c2] + 1e-5)
                self._ratio_pairs.append((c1, c2))
        return X

    def _apply_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        for c1, c2 in self._ratio_pairs:
            if c1 in X.columns and c2 in X.columns: X[f"{c1}__div__{c2}"] = X[c1] / (X[c2] + 1e-5)
        return X

    def _fit_polynomials(self, X: pd.DataFrame) -> pd.DataFrame:
        k = min(self.interaction_features, 5)
        numeric_original = [c for c, t in self._feature_types.items() if t == "numeric_original" and c in X.columns]
        if len(numeric_original) == 0: return X
        top_cols = numeric_original[:k]
        for col in top_cols:
            X[f"{col}__squared"] = X[col] ** 2
            self._poly_cols.append(col)
        return X

    def _apply_polynomials(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._poly_cols:
            if col in X.columns: X[f"{col}__squared"] = X[col] ** 2
        return X
