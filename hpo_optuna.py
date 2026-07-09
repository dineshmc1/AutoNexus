import time
import optuna
import numpy as np
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from multi_objective import calculate_utility_absolute, MODEL_COMPLEXITY
from model_trainer import create_model, baseline_screen

def get_search_space(model_name):
    """Define search space for each model type"""
    if model_name in ['xgb_clf', 'xgb_reg']:
        return {
            'n_estimators': optuna.distributions.IntDistribution(50, 300),
            'max_depth': optuna.distributions.IntDistribution(3, 10),
            'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            'subsample': optuna.distributions.FloatDistribution(0.6, 1.0)
        }
    elif model_name in ['rf', 'rf_reg', 'et_clf', 'et_reg']:
        return {
            'n_estimators': optuna.distributions.IntDistribution(50, 300),
            'max_depth': optuna.distributions.IntDistribution(5, 30),
            'min_samples_split': optuna.distributions.IntDistribution(2, 10)
        }
    elif model_name in ['lgbm_clf', 'lgbm_reg']:
        return {
            'n_estimators': optuna.distributions.IntDistribution(50, 300),
            'max_depth': optuna.distributions.IntDistribution(3, 15),
            'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            'num_leaves': optuna.distributions.IntDistribution(20, 100)
        }
    elif model_name in ['logistic', 'ridge', 'lasso', 'elastic']:
        return {
            'C' if model_name == 'logistic' else 'alpha': optuna.distributions.FloatDistribution(1e-4, 10.0, log=True)
        }
    elif model_name in ['sgd_clf', 'sgd_reg']:
        return {
            'alpha': optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True),
            'penalty': optuna.distributions.CategoricalDistribution(['l2', 'l1', 'elasticnet'])
        }
    # For SVC/SVR, KNN, and others, return empty to fallback to default params
    return {}

def objective(trial, X, y, preprocessor, model_name, problem_type, w1, w2, w3):
    """Optuna objective function that returns multi-objective utility score"""
    # 1. Suggest parameters
    params = {}
    search_space = get_search_space(model_name)
    if not search_space:
        raise optuna.exceptions.TrialPruned("No search space defined for this model.")
        
    for param, dist in search_space.items():
        if isinstance(dist, optuna.distributions.IntDistribution):
            params[param] = trial.suggest_int(param, dist.low, dist.high)
        elif isinstance(dist, optuna.distributions.FloatDistribution):
            params[param] = trial.suggest_float(param, dist.low, dist.high, log=dist.log)
        elif isinstance(dist, optuna.distributions.CategoricalDistribution):
            params[param] = trial.suggest_categorical(param, dist.choices)
    
    # 2. Train model with suggested parameters
    model = create_model(model_name, params)
    model_dict = {model_name: model}
    
    try:
        _, scores = baseline_screen(
            model_dict, preprocessor, X, y, problem_type,
            sample_frac=1.0, cv=3, random_state=42
        )
        if not scores:
            raise optuna.exceptions.TrialPruned()
            
        cv_score = scores[model_name]['score']
        train_time = scores[model_name]['time']
        complexity = MODEL_COMPLEXITY.get(model_name, 3)
        
        # 3. Calculate utility score
        utility = calculate_utility_absolute(cv_score, train_time, complexity, w1, w2, w3, problem_type)
        return utility
    except Exception as e:
        raise optuna.exceptions.TrialPruned()

def run_hpo(X, y, preprocessor, top_models, memory_hparams, problem_type, dataset_id):
    """Run HPO on top models with memory-warm-start"""
    w1, w2, w3 = (0.8, 0.15, 0.05) if problem_type == 'regression' else (0.6, 0.3, 0.1)
    
    best_overall_model = None
    best_overall_utility = -np.inf
    best_params = {}

    for model_name in top_models:
        search_space = get_search_space(model_name)
        if not search_space:
            print(f"[HPO] Skipping {model_name} (no search space defined).")
            continue
            
        # 🚀 USE A PRUNER TO SAVE TIME 🚀
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        
        study = optuna.create_study(
            direction='maximize', 
            study_name=f"hpo_{model_name}_{dataset_id}",
            pruner=pruner
        )
        
        warm_start_used = False
        # --- RESEARCH NOVELTY: MEMORY WARM-START ---
        if model_name in memory_hparams and memory_hparams[model_name]:
            # Ensure warm-started params match search space bounds
            valid_params = {}
            for param, val in memory_hparams[model_name].items():
                if param in search_space:
                    valid_params[param] = val
                    
            if valid_params:
                # Filter out None values to prevent Optuna crashes
                clean_params = {k: v for k, v in valid_params.items() if v is not None}
                print(f"  [HPO] Warm-starting {model_name} with memory params.")
                study.enqueue_trial(clean_params)
                warm_start_used = True
        
        # W&B Callback (logging to the active pipeline run)
        wandb_callback = WeightsAndBiasesCallback(
            metric_name=f"hpo/{model_name}/utility_score"
        )
        
        # Reduce trials from 100 to 30-50. With a Pruner, 30 trials is often better than 100 random ones.
        n_trials = 30 if problem_type == 'classification' else 20
        
        print(f"  [HPO] Running {n_trials} Optuna trials for {model_name}...")
        study.optimize(
            lambda trial: objective(trial, X, y, preprocessor, model_name, problem_type, w1, w2, w3),
            n_trials=n_trials,
            callbacks=[wandb_callback],
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials and study.best_value > best_overall_utility:
            best_overall_utility = study.best_value
            best_overall_model = model_name
            best_params = study.best_params
            
        best_val_to_log = study.best_value if completed_trials else None
            
        wandb.log({
            f"hpo/{model_name}/best_utility": best_val_to_log,
            f"hpo/{model_name}/trials_run": len(study.trials),
            f"hpo/{model_name}/completed_trials": len(completed_trials),
            f"hpo/{model_name}/warm_start_used": warm_start_used
        })
    
    return best_overall_model, best_params
