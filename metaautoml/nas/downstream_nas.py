import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb

class DownstreamNAS:
    def __init__(self, train_cache_path, val_cache_path):
        self.X_train, self.y_train = self._load_cache(train_cache_path)
        self.X_val, self.y_val = self._load_cache(val_cache_path)
        
    def objective(self, trial):
        # Define Search Space
        model_type = trial.suggest_categorical('model_type', ['xgb', 'lgbm', 'mlp'])
        
        if model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500)
            }
            model = self._build_xgb(params)
        elif model_type == 'mlp':
            # Shallow Neural Network Search
            params = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'n_layers': trial.suggest_int('n_layers', 2, 4)
            }
            model = self._build_mlp(params)
            
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_val, self.y_val)
        return score

    def run_search(self, n_trials=100):
        # Phase 5.3 W&B Integration (Exact schema from reference doc)
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="cv_score",
            wandb_kwargs={"project": "automl-hpo", "name": "downstream_nas"}
        )
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, callbacks=[wandb_callback])
        
        return study.best_trial
