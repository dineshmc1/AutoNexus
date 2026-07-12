import optuna
import wandb
import numpy as np
import xgboost as xgb
import lightgbm as lgb

class RegularizedObjective:
    def __init__(self, X_train, y_train, X_val, y_val, model_type):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type
        
        # Penalty configuration: Penalize heavily if gap > 5%
        self.penalty_weight = 0.5 
        self.gap_threshold = 0.05

    def __call__(self, trial):
        # 1. Suggest hyperparameters based on model type
        if self.model_type in ['xgb', 'lgbm']:
            params = self._suggest_tree_params(trial)
            if self.model_type == 'xgb':
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
            else:
                model = lgb.LGBMClassifier(**params)
            
            # 2. Train and Evaluate
            model.fit(self.X_train, self.y_train)
            train_score = model.score(self.X_train, self.y_train)
            val_score = model.score(self.X_val, self.y_val)
                
        elif self.model_type == 'mlp':
            import torch
            import pytorch_lightning as pl
            from torch.utils.data import DataLoader, TensorDataset
            from metaautoml.models.lightning_downstream_mlp import LightningDownstreamMLP
            
            params = self._suggest_mlp_params(trial)
            num_classes = len(np.unique(self.y_train))
            model = LightningDownstreamMLP(
                input_dim=self.X_train.shape[1], 
                num_classes=num_classes, 
                **params
            )
            
            # Convert numpy to PyTorch Tensors
            train_dataset = TensorDataset(
                torch.tensor(self.X_train, dtype=torch.float32),
                torch.tensor(self.y_train, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(self.X_val, dtype=torch.float32),
                torch.tensor(self.y_val, dtype=torch.long)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
            
            # THE MAGIC: PyTorch Lightning Trainer
            # precision='16-mixed' halves VRAM and speeds up training by ~2x
            trainer = pl.Trainer(
                max_epochs=50,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                precision='16-mixed', 
                enable_progress_bar=False, # Disable for HPO speed
                enable_model_summary=False,
                logger=False, # We use W&B at the study level, not per trial
                callbacks=[pl.callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='max')]
            )
            
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # Evaluate for Optuna
            train_score = model.score(self.X_train, self.y_train)
            val_score = model.score(self.X_val, self.y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # 3. Calculate Generalization Gap & Apply Penalty
        gen_gap = train_score - val_score
        penalty = self.penalty_weight * max(0, gen_gap - self.gap_threshold)
        optimized_metric = val_score - penalty

        # 4. Log raw metrics to W&B for dashboard analysis (Phase 5.3 custom logs)
        wandb.log({
            "trial_number": trial.number,
            "model_type": self.model_type,
            "raw_train_score": train_score,
            "raw_val_score": val_score,
            "generalization_gap": gen_gap,
            "penalty_applied": penalty,
            "optimized_metric": optimized_metric
        })

        # Return the penalized metric for Optuna to maximize
        return optimized_metric

    def _suggest_tree_params(self, trial):
        """Strictly constrained search space for Tree models to prevent overfitting."""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),      # L1
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),    # L2
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300)
        }

    def _suggest_mlp_params(self, trial):
        """Strictly constrained search space for Neural models."""
        return {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'n_layers': trial.suggest_int('n_layers', 2, 3),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True), # L2 for NN
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }