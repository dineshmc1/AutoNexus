import time
import wandb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

class AutoMLBaggingRouter:
    def should_bag(self, X, y):
        # Variance Proxy: Train a tiny decision tree, check impurity
        tiny_tree = DecisionTreeClassifier(max_depth=2).fit(X[:1000], y[:1000])
        noise_proxy = 1.0 - tiny_tree.score(X[:1000], y[:1000])
        
        # Condition: If noise is high OR dataset is small (< 5k rows), BAG.
        # Otherwise, skip bagging to save inference time.
        if noise_proxy > 0.3 or len(X) < 5000:
            return True, noise_proxy
        return False, noise_proxy

    def execute_automl(self, X, y):
        start = time.time()
        bag_enabled, noise = self.should_bag(X, y)
        
        if bag_enabled:
            # Use XGBoost/LGBM with native subsampling (Conditionally Pruned)
            params = {'subsample': 0.8, 'colsample_bytree': 0.8, 'n_estimators': 500}
            model = BaggingClassifier(XGBClassifier(**params), n_estimators=5)
        else:
            # Single strong model
            model = XGBClassifier(subsample=1.0, colsample_bytree=1.0)
            
        pareto_score = 0.0 # Placeholder logic to be implemented later
            
        # Log to W&B using Phase 5.2 schema
        wandb.log({
            "model": "automl_bagged" if bag_enabled else "automl_single",
            "accuracy": cross_val_score(model, X, y).mean(),
            "train_time_seconds": time.time() - start,
            "model_complexity": model.get_complexity() if hasattr(model, 'get_complexity') else 0,
            "multi_objective_score": pareto_score,
            "automl_noise_proxy": noise,
            "bagging_triggered": bag_enabled
        })
