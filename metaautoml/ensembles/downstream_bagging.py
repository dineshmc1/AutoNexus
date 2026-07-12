import time
import wandb
import numpy as np

class DownstreamBagging:
    def ensemble_top_k(self, best_configs, X_train, y_train, k=3):
        models = []
        start_time = time.time()
        
        for config in best_configs[:k]:
            # Bootstrap the cached embeddings (CPU only, very fast)
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot, y_boot = X_train[indices], y_train[indices]
            
            model = self._build_model_from_config(config)
            model.fit(X_boot, y_boot)
            models.append(model)
            
        elapsed = time.time() - start_time
        
        # Phase 5.2 W&B Logging (Exact schema from reference doc)
        wandb.log({
            "model": f"downstream_ensemble_k{k}",
            "accuracy": self._evaluate_ensemble(models),
            "train_time_seconds": elapsed,
            "model_complexity": sum(m.get_complexity() for m in models),
            "multi_objective_score": self._calculate_pareto_score(elapsed, models)
        })
        return models
