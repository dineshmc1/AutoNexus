import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import wandb

class OOFStackingEnsemble:
    def __init__(self, base_model_factories, n_folds=5):
        """
        base_model_factories: List of 3 callable functions that return a fresh, untrained model instance.
        Example: [lambda: XGBClassifier(**params1), lambda: LGBMClassifier(**params2), lambda: GPUDownstreamMLP(**params3)]
        """
        self.base_model_factories = base_model_factories
        self.n_folds = n_folds
        self.meta_learner = LogisticRegression(max_iter=1000, C=1.0)
        self.trained_base_models = []
        
    def fit(self, X, y):
        start_time = time.time()
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        n_classes = len(np.unique(y))
        n_models = len(self.base_model_factories)
        
        # Initialize OOF predictions array: (N_samples, N_classes, N_models)
        oof_preds = np.zeros((len(X), n_classes, n_models))
        
        # 1. Generate Out-Of-Fold Predictions
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for model_idx, factory in enumerate(self.base_model_factories):
                model = factory()
                model.fit(X_train, y_train)
                
                # Get probabilities for meta-features
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_val)
                    # Handle edge case where a fold might miss a class
                    if preds.shape[1] < n_classes:
                        preds = self._pad_probabilities(preds, n_classes)
                else:
                    preds = np.eye(n_classes)[model.predict(X_val)]
                    
                oof_preds[val_idx, :, model_idx] = preds
                
        # Flatten OOF predictions for meta-learner: (N_samples, N_classes * N_models)
        X_meta = oof_preds.reshape(len(X), -1)
        
        # 2. Train Meta-Learner on OOF predictions
        self.meta_learner.fit(X_meta, y)
        
        # 3. Train Base Models on FULL data for final inference
        self.trained_base_models = []
        for factory in self.base_model_factories:
            model = factory()
            model.fit(X, y)
            self.trained_base_models.append(model)
            
        elapsed = time.time() - start_time
        return elapsed

    def predict_proba(self, X):
        # Get predictions from all base models trained on full data
        base_preds = []
        for model in self.trained_base_models:
            if hasattr(model, "predict_proba"):
                base_preds.append(model.predict_proba(X))
            else:
                n_classes = self.meta_learner.classes_.shape[0] if hasattr(self.meta_learner, 'classes_') else len(np.unique(base_preds[0]))
                base_preds.append(np.eye(n_classes)[model.predict(X)])
                
        # Stack and flatten exactly as we did for OOF
        stacked_preds = np.stack(base_preds, axis=-1)
        X_meta = stacked_preds.reshape(len(X), -1)
        
        return self.meta_learner.predict_proba(X_meta)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _pad_probabilities(self, preds, n_classes):
        """Handles edge cases where a specific fold doesn't contain all classes."""
        padded = np.zeros((preds.shape[0], n_classes))
        padded[:, :preds.shape[1]] = preds
        return padded
