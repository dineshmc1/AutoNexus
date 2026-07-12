import wandb
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from metaautoml.ensembles.oof_stacking import OOFStackingEnsemble
from metaautoml.models.lightning_downstream_mlp import LightningDownstreamMLP

def get_top_3_diverse_factories(study, X_train_shape, num_classes):
    """
    Extracts the best XGB, best LGBM, and best MLP from the Optuna study.
    Returns a list of lambda factories to ensure fresh model instances.
    """
    best_xgb = None
    best_lgbm = None
    best_mlp = None
    
    # Sort trials by value (assuming direction='maximize')
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    
    for trial in sorted_trials:
        if trial.state.name == 'COMPLETE':
            params = trial.params
            if 'max_depth' in params and best_xgb is None:
                best_xgb = lambda p=params: xgb.XGBClassifier(**p, use_label_encoder=False, eval_metric='mlogloss')
            elif 'num_leaves' in params and best_lgbm is None: # LGBM specific param
                best_lgbm = lambda p=params: lgb.LGBMClassifier(**p)
            elif 'hidden_dim' in params and best_mlp is None:
                best_mlp = lambda p=params: LightningDownstreamMLP(input_dim=X_train_shape, num_classes=num_classes, **p)
                
        if best_xgb and best_lgbm and best_mlp:
            break
            
    # Fallback if we didn't find all 3 (just duplicate the best available)
    factories = [f for f in [best_xgb, best_lgbm, best_mlp] if f is not None]
    while len(factories) < 3:
        factories.append(factories[0]) 
        
    return factories[:3]

def run_stacking_ensemble(X_train, y_train, X_test, y_test, study):
    # 1. Get diverse model factories
    num_classes = len(np.unique(y_train))
    factories = get_top_3_diverse_factories(study, X_train.shape[1], num_classes)
    
    # 2. Initialize and Fit Stacking Ensemble
    ensemble = OOFStackingEnsemble(base_model_factories=factories, n_folds=5)
    train_time = ensemble.fit(X_train, y_train)
    
    # 3. Evaluate
    test_preds = ensemble.predict(X_test)
    accuracy = np.mean(test_preds == y_test)
    
    # 4. Calculate Complexity (Proxy: sum of parameters/features used)
    # For trees: n_estimators * max_depth. For MLP: hidden_dim * n_layers.
    complexity = 0
    for factory in factories:
        model = factory()
        if hasattr(model, 'n_estimators'):
            complexity += model.n_estimators * model.max_depth
        elif hasattr(model, 'hidden_dim'):
            complexity += model.hidden_dim * model.n_layers
            
    meta_complexity = X_train.shape[1] * 5 * 3 # Rough proxy for Logistic Regression
    
    # 5. Phase 5.2 W&B Logging (EXACT SCHEMA FROM REFERENCE DOC)
    combined_score = accuracy / (train_time * (complexity + meta_complexity + 1e-6)) # Pareto metric
    
    wandb.log({
        "model": "oof_stacking_ensemble",
        "accuracy": accuracy,
        "train_time_seconds": train_time,
        "model_complexity": complexity + meta_complexity,
        "multi_objective_score": combined_score
    })
    
    return ensemble, accuracy
