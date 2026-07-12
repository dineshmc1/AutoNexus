import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import optuna
from sklearn.tree import DecisionTreeClassifier
from metaautoml.ensembles.oof_stacking import OOFStackingEnsemble
from metaautoml.pipelines.stacking_integration import get_top_3_diverse_factories

def test_oof_no_leakage():
    """
    If there is leakage, the meta-learner will achieve 100% accuracy 
    because the base models will memorize the validation fold.
    A leak-free OOF ensemble on random noise should perform at chance level (~50% for binary).
    """
    # 1. Create completely random data (no actual pattern to learn)
    np.random.seed(42)
    X_random = np.random.randn(1000, 10)
    y_random = np.random.randint(0, 2, 1000)
    
    # 2. Use a highly complex model that WILL memorize data if leaked
    # (A deep decision tree with no max_depth limit)
    def memorizing_factory():
        return DecisionTreeClassifier(random_state=42) 
        
    # 3. Run the OOF Stacking
    ensemble = OOFStackingEnsemble(base_model_factories=[memorizing_factory], n_folds=5)
    ensemble.fit(X_random, y_random)
    
    # 4. Evaluate on the SAME random data
    # If leaked: accuracy will be ~1.0 (the meta-learner learned the random noise)
    # If clean: accuracy will be ~0.5 (chance level for binary random noise)
    preds = ensemble.predict(X_random)
    accuracy = np.mean(preds == y_random)
    
    print(f"Leakage Test Accuracy on Random Noise: {accuracy:.4f}")
    
    # Assert that the accuracy is close to chance (0.5), NOT 1.0
    # We allow a small margin (0.6) for random statistical fluctuation
    assert accuracy < 0.60, f"DATA LEAKAGE DETECTED! OOF accuracy on random noise is {accuracy}. The meta-learner is cheating."
    print("[PASS]: No data leakage detected in OOF engine.")


def test_shape_alignment():
    # Assert that X_meta.reshape(len(X), -1) correctly flattens the (N, n_classes, n_models) tensor
    np.random.seed(42)
    N = 10
    n_classes = 3
    n_models = 2
    
    oof_preds = np.random.rand(N, n_classes, n_models)
    X_meta = oof_preds.reshape(N, -1)
    
    assert X_meta.shape == (N, n_classes * n_models), "Flattened shape is incorrect"
    
    for i in range(N):
        for c in range(n_classes):
            for m in range(n_models):
                assert X_meta[i, c * n_models + m] == oof_preds[i, c, m]
    
    print("[PASS]: Shape alignment and flattening logic is correct.")


def test_diversity():
    # Assert that get_top_3_diverse_factories returns at least two different algorithmic families
    study = optuna.create_study(direction='maximize')
    
    # Add trials imitating different models directly
    from optuna.distributions import IntDistribution, FloatDistribution
    trial1 = optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, value=0.9, params={'max_depth': 3}, distributions={'max_depth': IntDistribution(1, 10)})
    trial2 = optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, value=0.85, params={'num_leaves': 31}, distributions={'num_leaves': IntDistribution(10, 50)})
    
    mlp_params = {'hidden_dim': 128, 'n_layers': 2, 'dropout_rate': 0.5, 'weight_decay': 1e-4, 'learning_rate': 1e-3}
    mlp_dists = {
        'hidden_dim': IntDistribution(64, 256),
        'n_layers': IntDistribution(2, 4),
        'dropout_rate': FloatDistribution(0.1, 0.5),
        'weight_decay': FloatDistribution(1e-5, 1e-2),
        'learning_rate': FloatDistribution(1e-4, 1e-2)
    }
    trial3 = optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, value=0.8, params=mlp_params, distributions=mlp_dists)
    
    study.add_trial(trial1)
    study.add_trial(trial2)
    study.add_trial(trial3)
    
    factories = get_top_3_diverse_factories(study, X_train_shape=10, num_classes=2)
    
    assert len(factories) == 3, "Should return exactly 3 factories"
    
    models = [f() for f in factories]
    types = [type(m).__name__ for m in models]
    
    # We should have XGBClassifier, LGBMClassifier, and LightningDownstreamMLP
    assert "XGBClassifier" in types, "Missing XGBoost in diverse models"
    assert "LGBMClassifier" in types, "Missing LightGBM in diverse models"
    assert "LightningDownstreamMLP" in types, "Missing MLP in diverse models"
    
    unique_types = set(types)
    assert len(unique_types) >= 2, "Failed to return at least two different algorithmic families"
    print("[PASS]: Diversity Test - At least two different algorithmic families are present.")

if __name__ == "__main__":
    test_oof_no_leakage()
    test_shape_alignment()
    test_diversity()
