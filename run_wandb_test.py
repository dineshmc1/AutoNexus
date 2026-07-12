import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from metaautoml.nas.regularized_objective import RegularizedObjective
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# 1. Initialize W&B (Phase 5.3 Schema)
wandb.init(project="automl-hpo", name="regularized_phase_5_3")

# 2. Setup the exact Phase 5.3 W&B Callback from your reference doc
wandb_callback = WeightsAndBiasesCallback(
    metric_name="cv_score", # This maps to the 'optimized_metric' returned by the objective
    wandb_kwargs={"project": "automl-hpo"}
)

# 3. Initialize your data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Optuna Study
study = optuna.create_study(direction='maximize', study_name="regularized_downstream_search")

# 5. Run Optimization with the Callback
objective = RegularizedObjective(X_train, y_train, X_val, y_val, model_type='xgb')

study.optimize(
    objective, 
    n_trials=100, 
    callbacks=[wandb_callback] # Automatically logs every trial's params and cv_score to W&B
)

print(f"Best optimized metric: {study.best_value}")
print(f"Best params: {study.best_params}")
