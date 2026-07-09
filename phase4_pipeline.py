import time
import numpy as np
import pandas as pd
import faiss
import random
import os
import json
import argparse
import re

# Custom imports from our pipeline
from dataset_embedding import compute_dataset_embedding
from data_loader import detect_problem_type
from data_loader import load_local_dataset
from data_cleaner import clean
from feature_processing import build_preprocessor
from model_trainer import get_models, baseline_screen
from cold_start import MemoryStore
from config import USE_LLM, USE_WANDB
from paradigm_router import route_paradigm
from dataset_profiler import profile_dataset

def extract_meta_features(X, y) -> dict:
    import numpy as np
    numeric_cols = X.select_dtypes(include=[np.number])
    cat_cols = X.select_dtypes(exclude=[np.number])
    n_samples, n_cols = X.shape
    n_num = numeric_cols.shape[1]
    n_cat = cat_cols.shape[1]
    n_classes = y.nunique()

    return {
        "n_samples":         n_samples,
        "n_features":        n_cols,
        "num_ratio":         n_num / max(n_cols, 1),
        "cat_ratio":         n_cat / max(n_cols, 1),
        "missing_rate":      X.isnull().mean().mean(),
        "skewness_mean":     numeric_cols.skew().mean() if n_num > 0 else 0.0,
        "mean_corr":         numeric_cols.corr().abs().values[np.triu_indices(n_num, k=1)].mean() if n_num > 1 else 0.0,
        "n_classes":         int(n_classes),
        "is_binary":         n_classes == 2,
        "target_entropy":    float(-(y.value_counts(normalize=True) * np.log(y.value_counts(normalize=True) + 1e-10)).sum()),
        "majority_class_ratio": float(y.value_counts(normalize=True).iloc[0])
    }

def sanitize_filename(dataset_id_or_path):
    """Converts a full path or ID into a safe filename string."""
    base = os.path.basename(str(dataset_id_or_path))
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', os.path.splitext(base)[0])
    return safe_name

def run_single_dataset_pipeline(X, y, problem_type, store, encoder, did="local", validate=False, modality="tabular", config=None):
    import numpy as np
    from wandb_logger import log
    
    did_safe = sanitize_filename(did)
    did = did_safe
    
    # 🚨 MULTI-MODAL OVERRIDE LOGIC 🚨
    if modality in ['vision', 'audio', 'text', 'video']:
        print("\n" + "="*50)
        print("🚨 MULTI-MODAL OVERRIDE DETECTED")
        print("="*50)
        print("🛑 Bypassing Tabular FAISS Memory (Incompatible Vector Space)")
        print("🛑 Bypassing Tabular R(D) Router")
        print("🧠 Forcing Paradigm Decision: AutoDL NAS")
        print("="*50 + "\n")
        
        paradigm_decision = "AutoDL"
        r_d_score = 1.0
        llm_score, memory_score, heuristics_score = 1.0, 0.0, 1.0
        warm_params = {}
        best_retrieved_models = []
        w1_def, w2_def, w3_def = (0.6, 0.3, 0.1)
        
    else:
        # 1. Extract meta-features
        from task_encoder import encode_dataset
        raw_vec = compute_dataset_embedding(X, y)
        query_vec = encode_dataset(raw_vec, encoder).reshape(1, -1).astype(np.float32)
    
        print(f"\\n[DEBUG] Raw Meta-Features (10D): {np.round(raw_vec, 4)}")
        print(f"[DEBUG] Learned Embedding (32D) [First 5]: {np.round(query_vec[0][:5], 4)} ...")
    
        meta_features = extract_meta_features(X, y)
    
        # 2. Query FAISS memory
        dists, idxs = store._index.search(query_vec, k=5)
        neighbors = [store.records[i] for i in idxs[0] if i != -1]
    
        # Safety Check: If fewer than 3 valid neighbors found, trigger cold-start heuristics
        if len(neighbors) < 3:
            print("\n⚠️ WARNING: Fewer than 3 valid neighbors found in FAISS. Triggering cold-start fallback heuristics.")
            from heuristics import get_heuristic_suggestions
            best_retrieved_models = get_heuristic_suggestions(meta_features, problem_type)[:3]
            warm_params = {}
        else:
            best_retrieved_models = []
            for n in neighbors[:3]:
                if n.models:
                    best_retrieved_models.extend(n.models)
            # Deduplicate while preserving order
            best_retrieved_models = list(dict.fromkeys(best_retrieved_models))[:3]
            
            best_distance = dists[0][0]
            MAX_DISTANCE_THRESHOLD = 0.50
            if best_distance <= MAX_DISTANCE_THRESHOLD:
                print(f"\n  [Memory] Close match found (L2 Distance: {best_distance:.4f}). Warm-starting...")
                warm_params = neighbors[0].metadata.get("hparams", {})
            else:
                print(f"\n  [Memory] Distant match (L2 Distance: {best_distance:.4f} > {MAX_DISTANCE_THRESHOLD}). Triggering COLD START.")
                warm_params = {}
        
            print(f"\n📂 DATASET: {did} | TARGET: {y.name if hasattr(y, 'name') else 'Unknown'} | TYPE: {problem_type}\n")
            print("🔍 MEMORY RETRIEVAL (Top 3 Similar Past Experiments):")
            print("┌────────────┬────────────┬──────────────┬──────────────────────────────┐")
            print("│ Dataset    │ Distance │ Best Model   │ Warm-Start Hyperparameters   │")
            print("├────────────┼────────────┼──────────────┼──────────────────────────────┤")
            for i, n in enumerate(neighbors[:3]):
                did_name = n.metadata.get("dataset_id", "Unknown")
                sim = dists[0][i]
                b_model = n.models[0] if n.models else "Unknown"
                hp = str(n.metadata.get("hparams", {})).replace("\n", "")[:28]
                print(f"│ {did_name:<10} │ {sim:<10.4f} │ {b_model:<12} │ {hp:<28} │")
            print("└────────────┴────────────┴──────────────┴──────────────────────────────┘")

        # 3. LLM Graceful Fallback
        from llm_suggester import get_llm_suggestions
        try:
            llm_models, llm_reasoning, llm_ok = get_llm_suggestions(meta_features, problem_type, str(did))
            llm_suggestion = llm_reasoning
        except Exception as e:
            llm_suggestion = f"LLM unavailable ({e}). Relying solely on memory retrieval."
            llm_models = []
        
        print(f"\n🤖 LLM SUGGESTION: {llm_suggestion}\n")
        print("⚙️ CONFIGURING PIPELINE...")
    
        # 4. Paradigm Routing
        profile = profile_dataset(did, X, y, problem_type)
        paradigm_decision, r_d_score, llm_score, memory_score, heuristics_score = route_paradigm(
            dataset_profile=profile,
            faiss_store=store,
            query_embedding=query_vec
        )
    
        w1_def, w2_def, w3_def = (0.8, 0.15, 0.05) if problem_type == 'regression' else (0.6, 0.3, 0.1)
    
        print(f"→ Routing Decision: {paradigm_decision} (R(D)={r_d_score:.2f})")
        print(f"→ Multi-Objective Weights: Accuracy={w1_def}, Speed={w2_def}, Complexity={w3_def}")
    
    if paradigm_decision == "AutoML":
        print("\n🚀 Executing Classical ML Pipeline...")
        # ✅ NEW: Activate God-Tier Feature Engineering
        from feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer(
            skew_threshold=1.0,
            rare_threshold=0.01,      # Groups categories < 1% into "_RARE"
            corr_threshold=0.95,      # Drops highly collinear features
            encoding_strategy="target",
            interaction_features=5,
            enable_ratios=True
        )
        
        # Apply engineering (handles text extraction, Yeo-Johnson, outlier capping, etc.)
        X = fe.fit_transform(X, y, problem_type)
        
        # Pass the learned scaler map to the preprocessor for optimal scaling
        preprocessor_cs, _, _ = build_preprocessor(X, scaler_map=fe.get_scalers())
        
        full_search_best_model_name = "NONE"
        best_params = {}
        top_3_shap_features = []
        
        if validate:
            print(f"  [VALIDATION] Running full benchmark across all models...")
            all_models_full = get_models(problem_type)
            _, all_scores = baseline_screen(
                all_models_full, preprocessor_cs, X, y, problem_type,
                sample_frac=1.0, cv=3, random_state=42
            )
            if all_scores:
                from multi_objective import select_best_model_multiobjective
                best_model_by_utility, _ = select_best_model_multiobjective(all_scores, task_type=problem_type, w1=w1_def, w2=w2_def, w3=w3_def)
                full_score = all_scores[best_model_by_utility]['score']
                full_search_best_model_name = best_model_by_utility
                print(f"  [VALIDATION] Full Search Winner: {full_search_best_model_name} (Score: {full_score:.4f})")
                
        # HPO on top models
        agentic_models = llm_models
        memory_models = best_retrieved_models
        
        # Combine them and remove duplicates
        models_to_train = list(set(agentic_models + memory_models))
        
        # CRITICAL: For Regression, ALWAYS ensure XGBoost/LightGBM/RF are in the pool!
        if problem_type == 'regression':
            default_powerhouses = ['xgb_reg', 'lgbm_reg', 'rf_reg']
            for m in default_powerhouses:
                if m not in models_to_train:
                    models_to_train.append(m)
                    
        top_models_hpo = models_to_train
        print(f"  [HPO] Running HPO on Combined Pool: {top_models_hpo}")
        from hpo_optuna import run_hpo
        
        # 🚀 ENSEMBLE OF EXPERTS LOGIC FOR AUTOML 🚀
        # Note: run_hpo currently returns single best. We will re-run top 3 params manually for ensemble.
        # For now, let's assume run_hpo gives us the best. To get top 3, we'd need to modify hpo_optuna.py 
        # to return the study object. For simplicity, we'll use the single best model for now 
        # and focus on the Hybrid ML path ensemble which is more critical for your thesis novelty.
        
        best_hpo_model, best_params = run_hpo(
            X, y, preprocessor_cs, top_models_hpo, warm_params, problem_type, str(did)
        )
        
        if best_hpo_model:
            print(f"  [HPO] Winner: {best_hpo_model} with params {best_params}")
            full_search_best_model_name = best_hpo_model
        
        # XAI SHAP Explanations
        if full_search_best_model_name not in ["NONE", "FAILED"]:
            from shap_explainer import generate_shap_explanations
            print(f"  [SHAP] Generating explanations for {full_search_best_model_name}...")
            try:
                final_model_instance = get_models(problem_type, [full_search_best_model_name])[full_search_best_model_name]
                
                from sklearn.model_selection import train_test_split
                import numpy as np
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.2, random_state=42)
                
                X_train_prep = preprocessor_cs.fit_transform(X_train_split, y_train_split)
                X_test_prep = preprocessor_cs.transform(X_test_split)
                final_model_instance.fit(X_train_prep, y_train_split)
                
                y_pred = final_model_instance.predict(X_test_prep)
                
                from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
                if problem_type == 'classification':
                    if hasattr(final_model_instance, "predict_proba"):
                        y_prob = final_model_instance.predict_proba(X_test_prep)
                        if y_prob.shape[1] == 2:
                            y_prob = y_prob[:, 1]
                    else:
                        y_prob = y_pred
                        
                    if len(y_prob.shape) == 1 or (len(y_prob.shape) == 2 and y_prob.shape[1] == 2):
                        try:
                            roc_auc = roc_auc_score(y_test_split, y_prob)
                            ll = log_loss(y_test_split, y_prob)
                        except:
                            roc_auc, ll = "N/A", "N/A"
                    else:
                        try:
                            roc_auc = roc_auc_score(y_test_split, y_prob, multi_class='ovr')
                            ll = log_loss(y_test_split, y_prob)
                        except:
                            roc_auc, ll = "N/A", "N/A"
                            
                    eval_metrics = {
                        "classification_report": classification_report(y_test_split, y_pred, output_dict=True),
                        "confusion_matrix": confusion_matrix(y_test_split, y_pred).tolist(),
                        "roc_auc": roc_auc,
                        "log_loss": ll
                    }
                else:
                    eval_metrics = {
                        "mae": mean_absolute_error(y_test_split, y_pred),
                        "mse": mean_squared_error(y_test_split, y_pred),
                        "rmse": np.sqrt(mean_squared_error(y_test_split, y_pred)),
                        "r2": r2_score(y_test_split, y_pred)
                    }
                
                total_models_in_catalog = 14
                models_actually_trained = len(all_models_full) if validate else len(top_models_hpo)
                scr = total_models_in_catalog / max(1, models_actually_trained)
                
                if problem_type == 'classification':
                    from sklearn.metrics import accuracy_score
                    final_accuracy = accuracy_score(y_test_split, y_pred)
                    majority_class = y_train_split.value_counts().idxmax()
                    baseline_preds = [majority_class] * len(y_test_split)
                    baseline_acc = accuracy_score(y_test_split, baseline_preds)
                    pr = final_accuracy / max(0.01, baseline_acc)
                else:
                    final_accuracy = eval_metrics['r2']
                    pr = 1.0
                
                tus = scr * pr 
                
                def calculate_ece(y_true, y_prob, n_bins=10):
                    if len(y_prob.shape) > 1: return 0.0
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    ece = 0.0
                    for i in range(n_bins):
                        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
                        if np.sum(mask) > 0:
                            try:
                                bin_acc = np.mean(np.array(y_true)[mask] == np.max(y_true))
                                bin_conf = np.mean(y_prob[mask])
                                ece += np.abs(bin_acc - bin_conf) * (np.sum(mask) / len(y_true))
                            except:
                                pass
                    return ece
                
                if problem_type == 'classification' and (len(y_prob.shape) == 1):
                    ece = calculate_ece(y_test_split, y_prob)
                else:
                    ece = 0.0
                    
                system_metrics = {
                    "confidence_score_C_D": round(float(r_d_score), 4),
                    "expected_calibration_error_ECE": round(ece, 4),
                    "search_compression_ratio_SCR": round(scr, 2),
                    "performance_retention_PR": round(pr, 2),
                    "transfer_utility_score_TUS": round(tus, 2)
                }

                if hasattr(preprocessor_cs, 'get_feature_names_out'):
                    feature_names = list(preprocessor_cs.get_feature_names_out())
                else:
                    # Fallback for AutoDL bypass or FunctionTransformer
                    feature_names = X_train_prep.columns.tolist() if hasattr(X_train_prep, 'columns') else [f"feat_{i}" for i in range(X_train_prep.shape[1])]
                
                dense_X = X_train_prep.toarray() if hasattr(X_train_prep, 'toarray') else X_train_prep
                X_train_df = pd.DataFrame(dense_X, columns=feature_names)
                
                dense_X_test = X_test_prep.toarray() if hasattr(X_test_prep, 'toarray') else X_test_prep
                X_test_df = pd.DataFrame(dense_X_test, columns=feature_names)
                
                success, top_3_shap_features = generate_shap_explanations(
                    model=final_model_instance, X_train=X_train_df, X_test=X_test_df, 
                    model_name=full_search_best_model_name, dataset_id=str(did)
                )
            except Exception as e:
                import traceback
                print(f"  [SHAP/Metrics] Failed: {e}")
                traceback.print_exc()
                
        # Phase 5.6: LLM Explainability Report
        try:
            from llm_explainer import generate_comprehensive_report
            master_context = {
                "paradigm_routing": {
                    "decision": paradigm_decision,
                    "R_D_score": round(float(r_d_score), 4),
                    "llm_signal": round(float(llm_score), 4),
                    "memory_signal": round(float(memory_score), 4),
                    "heuristic_signal": round(float(heuristics_score), 4)
                },
                "dataset_profile": profile,
                "training_and_hpo": {
                    "final_model": full_search_best_model_name,
                    "best_hpo_params": best_params,
                },
                "shap_interpretability": {
                    "top_3_features": top_3_shap_features,
                    "model_type": "Tree-based" if full_search_best_model_name in ['rf', 'gb', 'xgb_clf', 'xgb_reg', 'lgbm_clf', 'lgbm_reg', 'et_clf', 'et_reg'] else "Linear/Black-box"
                },
                "evaluation_metrics": eval_metrics if 'eval_metrics' in locals() else {},
                "system_metrics": system_metrics if 'system_metrics' in locals() else {}
            }
            generate_comprehensive_report(master_context, str(did))
        except Exception as e:
            print(f"  [Phase 5.6 Report] Failed: {e}")
            
        # Store in FAISS memory mapping using MemoryStore
        if full_search_best_model_name not in ["NONE", "FAILED"]:
            print(f"\\n[Memory] Saving execution results to FAISS...")
            metadata = {
                "dataset_id": str(did),
                "problem_type": problem_type,
                "hparams": {full_search_best_model_name: best_params}
            }
            # Save the current dataset's result
            store.add(str(did), query_vec[0], [full_search_best_model_name], metadata)
            store.build_index()
            
            # Save to disk
            MEMORY_INDEX_PATH = "memory_store.faiss"
            MEMORY_META_PATH  = "memory_store.pkl"
            store.save_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
            print("✅ Run successfully added to FAISS Memory Store!")

            # ─── Advanced Notebook Generator ─────────────────────────────────
            try:
                from notebook_generator import generate_advanced_notebook
                if full_search_best_model_name not in ["NONE", "FAILED"]:
                    results_dict = {
                        "X": X, "y": y, "y_test": None, "y_pred": None, # Test metrics aren't neatly localized here without re-evaluating
                        "final_accuracy": best_params.get("utility_score", 0),
                        "paradigm": "AutoML",
                        "modality": "tabular"
                    }
                    nb_path = f"reports/{did}_advanced_analysis.ipynb"
                    if config:
                        generate_advanced_notebook(config, results_dict, nb_path)
            except Exception as nb_err:
                print(f"  [Notebook Generator] Failed: {nb_err}")

            return {
                "score": final_accuracy if 'final_accuracy' in locals() else best_params.get("utility_score", 0),
                "models_evaluated": models_actually_trained if 'models_actually_trained' in locals() else 1,
                "eval_metrics": eval_metrics if 'eval_metrics' in locals() else {}
            }
            
    elif paradigm_decision == "AutoDL":
        print("\n🧠 Executing Hybrid ML-on-Embeddings Pipeline...")
        try:
            # 1. Prepare Data
            nas_prep, _, _ = build_preprocessor(X)
            X_nas_prep = nas_prep.fit_transform(X, y)
            X_train_numpy = X_nas_prep.toarray() if hasattr(X_nas_prep, 'toarray') else np.array(X_nas_prep)
            # 🚨 CRITICAL FIX: LABEL ENCODING 🚨
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(np.array(y)) if problem_type == 'classification' else np.array(y)
            is_clf = (problem_type == 'classification')
            
            # 2. FIRST SPLIT: Isolate TRUE TEST SET (20%) - NEVER TOUCHED DURING TRAINING
            from sklearn.model_selection import train_test_split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_train_numpy, y_encoded, test_size=0.2, random_state=42, 
                stratify=y_encoded if is_clf else None
            )
            
            # 3. SECOND SPLIT: Train vs Validation (80/20 of remaining = 64% / 16%)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, 
                stratify=y_temp if is_clf else None
            )
            print(f"  [Split] Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")

            # --- Warm-start DL Memory ---
            query_embedding = None
            dl_memory = None
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                from dl_faiss_memory import ModalityFAISSMemory
                
                scaler_warm = StandardScaler()
                X_scaled_warm = scaler_warm.fit_transform(X_train_numpy)
                n_comp = min(100, X_scaled_warm.shape[1], X_scaled_warm.shape[0])
                pca_warm = PCA(n_components=n_comp)
                X_pca_warm = pca_warm.fit_transform(X_scaled_warm)
                
                if X_pca_warm.shape[1] < 100:
                    pad_width = 100 - X_pca_warm.shape[1]
                    X_pca_warm = np.pad(X_pca_warm, ((0, 0), (0, pad_width)), mode='constant')
                
                query_embedding = X_pca_warm[0]
                dl_memory = ModalityFAISSMemory(modality)
            except Exception as e:
                print(f"  [DL Memory] Failed to init: {e}")

            # 3. Define Search Space for Classical Models on Embeddings
            import optuna
            from xgboost import XGBClassifier, XGBRegressor
            from lightgbm import LGBMClassifier, LGBMRegressor
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
            
            def objective_hybrid(trial, X_tr, y_tr, X_v, y_v, n_features):
                # Dynamic search bounds based on actual PCA dimensionality
                max_possible_depth = min(int(np.log2(n_features)) + 2, 15) 
                
                model_type = trial.suggest_categorical('model', ['xgb', 'lgbm', 'hgb'])
                
                if model_type == 'xgb':
                    params = {
                        'max_depth': trial.suggest_int('max_depth', 2, max_possible_depth), 
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'min_child_weight': 20,
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 20.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 20.0),
                        'random_state': 42,
                        'verbosity': 0
                    }
                    if problem_type == 'classification':
                        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
                    else:
                        model = XGBRegressor(**params, eval_metric='rmse')
                        
                elif model_type == 'lgbm':
                    params = {
                        'max_depth': trial.suggest_int('max_depth', 2, max_possible_depth),
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'min_child_samples': 20,
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 20.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 20.0),
                        'random_state': 42,
                        'verbose': -1
                    }
                    if problem_type == 'classification':
                        model = LGBMClassifier(**params)
                    else:
                        model = LGBMRegressor(**params)
                        
                elif model_type == 'hgb':
                    params = {
                        'max_iter': trial.suggest_int('max_iter', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 2, max_possible_depth),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 20.0),
                        'random_state': 42,
                    }
                    if problem_type == 'classification':
                        model = HistGradientBoostingClassifier(**params)
                    else:
                        model = HistGradientBoostingRegressor(**params)
                        
                if problem_type == 'classification' and len(X_tr) < 5000:
                    import torch
                    idx = torch.randperm(X_tr.shape[0]).numpy()
                    lam = np.random.beta(0.2, 0.2)  # Mixup alpha
                    X_tr_aug = lam * X_tr + (1 - lam) * X_tr[idx]
                    y_tr_aug = y_tr  # Keep original labels for mixup
                    model.fit(X_tr_aug, y_tr)
                else:
                    model.fit(X_tr, y_tr)
                    
                preds = model.predict(X_v) # <-- EVALUATE ON VAL, NOT TRAIN
                
                if problem_type == 'classification':
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_v, preds)
                else:
                    from sklearn.metrics import mean_squared_error
                    score = -mean_squared_error(y_v, preds) # Negative for minimization
                    
                return score

            # 4. Run Optuna HPO (Reduced trials since these models are fast)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            nas_study = optuna.create_study(direction='maximize', study_name=f"hybrid_ml_{did}")
            nas_study.optimize(lambda trial: objective_hybrid(trial, X_tr, y_tr, X_val, y_val, X_tr.shape[1]), n_trials=20)
            
            # 🚀 ENSEMBLE OF EXPERTS LOGIC 🚀
            # Get the top 3 trials from the study
            completed_trials = [t for t in nas_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:3]
            
            ensemble_preds_test = []
            ensemble_preds_tr = []
            ensemble_preds_val = []
            print(f"\n🤖 Training Ensemble of {len(top_trials)} Experts...")
            
            for i, trial in enumerate(top_trials):
                params = trial.params.copy()
                model_name = params.pop('model', 'xgb')
                
                # Re-instantiate the best model type with these specific params
                if model_name == 'xgb':
                    expert_model = XGBClassifier(**params, use_label_encoder=False) if problem_type=='classification' else XGBRegressor(**params)
                elif model_name == 'lgbm':
                    expert_model = LGBMClassifier(**params) if problem_type=='classification' else LGBMRegressor(**params)
                elif model_name == 'rf':
                    expert_model = RandomForestClassifier(**params) if problem_type=='classification' else RandomForestRegressor(**params)
                elif model_name == 'hgb':
                    expert_model = HistGradientBoostingClassifier(**params) if problem_type=='classification' else HistGradientBoostingRegressor(**params)
                    
                if problem_type == 'classification':
                    from sklearn.calibration import CalibratedClassifierCV
                    calibrated_expert = CalibratedClassifierCV(expert_model, cv=5, method='isotonic')
                    try:
                        calibrated_expert.fit(X_temp, y_temp)
                        expert_model = calibrated_expert
                    except Exception as e:
                        print(f"  [Calibration] Failed (likely due to class imbalance in CV): {e}. Falling back to uncalibrated expert.")
                        expert_model.fit(X_temp, y_temp)
                else:
                    expert_model.fit(X_temp, y_temp)
                
                def get_preds(X_data):
                    if problem_type == 'classification':
                        if hasattr(expert_model, 'predict_proba'):
                            return expert_model.predict_proba(X_data)
                        else:
                            p = expert_model.predict(X_data)
                            n_classes = len(np.unique(y_temp))
                            p_proba = np.zeros((len(p), n_classes))
                            p_proba[np.arange(len(p)), p.astype(int)] = 1.0
                            return p_proba
                    else:
                        return expert_model.predict(X_data)
                    
                ensemble_preds_test.append(get_preds(X_test))
                ensemble_preds_tr.append(get_preds(X_tr))
                ensemble_preds_val.append(get_preds(X_val))
                print(f"  ✅ Expert {i+1} trained ({model_name}).")

            # Combine Predictions
            if problem_type == 'classification':
                # Soft Voting: Average the probabilities
                final_preds_prob = np.mean(ensemble_preds_test, axis=0)
                final_y_pred = np.argmax(final_preds_prob, axis=1)
                
                tr_preds_prob = np.mean(ensemble_preds_tr, axis=0)
                tr_y_pred = np.argmax(tr_preds_prob, axis=1)
                
                val_preds_prob = np.mean(ensemble_preds_val, axis=0)
                val_y_pred = np.argmax(val_preds_prob, axis=1)
                
                # Calculate Metrics based on averaged probabilities
                from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
                train_acc = accuracy_score(y_tr, tr_y_pred)
                val_acc = accuracy_score(y_val, val_y_pred)
                final_acc = accuracy_score(y_test, final_y_pred)
                try:
                    if final_preds_prob.shape[1] == 2:
                        final_roc = roc_auc_score(y_test, final_preds_prob[:, 1])
                    else:
                        final_roc = roc_auc_score(y_test, final_preds_prob, multi_class='ovr')
                except:
                    final_roc = "N/A"
                    
                if 'le' in locals():
                    target_names = [str(c) for c in le.classes_]
                    class_names = target_names
                else:
                    target_names = None
                    class_names = [str(c) for c in np.unique(y_temp)]
                    
                clf_report_str = classification_report(y_test, final_y_pred, target_names=target_names)
                conf_matrix = confusion_matrix(y_test, final_y_pred).tolist()
                print(f"\n✅ TRUE TEST ACCURACY: {final_acc * 100:.2f}% | ROC-AUC: {final_roc}")
                print("\n📊 CLASSIFICATION REPORT:")
                print(clf_report_str)
                out_dim = len(np.unique(y_temp))
            else:
                # Regression: Average the values
                final_y_pred = np.mean(ensemble_preds_test, axis=0)
                tr_y_pred = np.mean(ensemble_preds_tr, axis=0)
                val_y_pred = np.mean(ensemble_preds_val, axis=0)
                
                from sklearn.metrics import r2_score, mean_absolute_error
                train_acc = r2_score(y_tr, tr_y_pred)
                val_acc = r2_score(y_val, val_y_pred)
                final_acc = r2_score(y_test, final_y_pred)
                final_mae = mean_absolute_error(y_test, final_y_pred)
                print(f"  📊 Ensemble MAE: {final_mae:.4f}")
                print(f"\n✅ TRUE TEST R²: {final_acc:.4f}")
                clf_report_str = "N/A"
                conf_matrix = []
                out_dim = 1
                class_names = []

            # 5. Generate SHAP Explanations (Native support for Tree models!)
            import matplotlib.pyplot as plt
            try:
                import shap
                # Use the first expert as the representative for SHAP analysis
                try:
                    explainer = shap.TreeExplainer(expert_model)
                except Exception:
                    # Fallback: Use the underlying estimator inside CalibratedClassifierCV
                    if hasattr(expert_model, 'estimator'):
                        explainer = shap.TreeExplainer(expert_model.estimator)
                    else:
                        print("  [SHAP] Skipping: Model type not supported.")
                        explainer = None
                
                if explainer is not None:
                    shap_values = explainer.shap_values(X_test[:100]) # Sample for speed
                    shap.summary_plot(shap_values, X_test[:100], show=False)
                    plt.title("SHAP Explanation for Embedding Features")
                    import os
                    os.makedirs("reports", exist_ok=True)
                    plt.savefig(f"reports/{did}_shap_embeddings.png")
                    plt.close()
                    print("  [SHAP] Successfully generated embedding explanations.")
            except Exception as e:
                print(f"  [SHAP] Failed: {e}")

            # 6. Save to Modality-Specific FAISS (Same logic as before)
            try:
                if dl_memory is not None and query_embedding is not None:
                    dl_memory.add(
                        embedding_100d=query_embedding,
                        dataset_name=str(did),
                        best_params=nas_study.best_params,
                        accuracy=float(final_acc) if final_acc is not None else 0.0
                    )
            except Exception as mem_err:
                print(f"  [DL Memory] Failed to save: {mem_err}")

            # 9. Generate Report & Notebook (Same logic as before)
            dl_context = {
                "paradigm_routing": {
                    "decision": "AutoDL (Hybrid ML)",
                    "R_D_score": r_d_score,
                    "modality": modality,
                    "extractor_used": "CLIP" if modality in ["vision", "video"] else
                                      "SentenceTransformer" if modality == "text" else
                                      "Librosa MFCC"
                },
                "dataset": {
                    "id": did,
                    "n_samples": int(X_train_numpy.shape[0]),
                    "n_features_after_pca": int(X_train_numpy.shape[1]),
                    "n_classes": int(out_dim),
                    "class_names": class_names
                },
                "hybrid_ml_results": {
                    "ensemble_size": len(top_trials),
                    "models_used": [t.params.get('model', 'xgb') for t in top_trials],
                    "best_ensemble_accuracy": round(float(final_acc), 4)
                },
                "final_performance": {
                    "test_accuracy": round(float(final_acc), 4) if final_acc is not None else None,
                    "classification_report": clf_report_str,
                    "confusion_matrix": conf_matrix
                }
            }

            try:
                from llm_explainer import generate_comprehensive_report
                generate_comprehensive_report(dl_context, did)
            except Exception as report_err:
                print(f"  [LLM Report] Failed to generate AutoDL report: {report_err}")

            try:
                from notebook_generator import generate_advanced_notebook
                results_dict = {
                    "X": X_test, "y": y_test, "y_test": y_test, "y_pred": final_y_pred,
                    "final_accuracy": final_acc,
                    "paradigm": "AutoDL (Hybrid ML)",
                    "modality": modality
                }
                nb_path = f"reports/{did}_advanced_analysis.ipynb"
                if config:
                    generate_advanced_notebook(config, results_dict, nb_path)
            except Exception as nb_err:
                print(f"  [Notebook Generator] Failed: {nb_err}")

            result_score = final_acc if 'final_acc' in locals() and final_acc is not None else 0.0
            return {
                "score": result_score,
                "train_score": train_acc if 'train_acc' in locals() else 0.0,
                "val_score": val_acc if 'val_acc' in locals() else 0.0,
                "models_evaluated": len(nas_study.trials) if 'nas_study' in locals() else 20
            }

        except Exception as e:
            print(f"  [AutoDL NAS] Failed: {e}")
            import traceback; traceback.print_exc()
            return {"score": 0.0, "models_evaluated": 0}


def main():
    parser = argparse.ArgumentParser(description='Run Phase 4 Pipeline')
    parser.add_argument('--validate', action='store_true', help='Run full baseline screening for validation')
    args = parser.parse_args()

    MEMORY_INDEX_PATH = "memory_store.faiss"
    MEMORY_META_PATH  = "memory_store.pkl"
    
    store = MemoryStore()
    if os.path.exists(MEMORY_INDEX_PATH):
        loaded = store.load_index(MEMORY_INDEX_PATH, MEMORY_META_PATH)
        print(f"✅ Loaded {loaded} existing records from disk")
    else:
        print("⚠️ Memory store not found. Creating a new empty store.")
        
    print("🤖 Loading pre-trained Task Encoder...")
    from task_encoder import SiameseEncoder, TaskEncoderConfig
    import torch
    cfg = TaskEncoderConfig(input_dim=10, hidden_dim=64, output_dim=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SiameseEncoder(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim).to(device)
    encoder_path = cfg.encoder_save_path
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        encoder.eval()
    else:
        print(f"⚠️ Task encoder model not found at {encoder_path}. Using un-trained encoder.")
        encoder.eval()

    print("\n" + "="*50)
    print("META-AUTOML UNIVERSAL INGESTION")
    print("="*50)
    
    user_input = input("Enter path to dataset (CSV file OR Image/Audio/Text Folder): ").strip()
    
    from onboarding_agent import OnboardingAgent
    agent = OnboardingAgent()
    config = agent.run(user_input)
    
    if not config:
        print("❌ Failed to process input. Exiting.")
        return
        
    modality = config['modality']
    
    # ==========================================
    # PATH A: MULTI-MODAL (FOLDERS)
    # ==========================================
    if modality in ['vision', 'text', 'audio', 'video']:
        print("🚀 Bypassing Agentic DataAgent. Routing to Multi-Modal Embedder...")
        
        from multimodal_extractor import UniversalEmbedder
        
        embedder = UniversalEmbedder(device=device, batch_size=32, domain=config['domain'])
        X, y = embedder.embed_directory(user_input, modality)
        
        print(f"✅ Embedding Extraction Complete! Shape: {X.shape}")
        
        problem_type = 'classification'
        run_single_dataset_pipeline(
            X, y, problem_type, store, encoder, 
            did=os.path.basename(user_input), 
            validate=args.validate,
            modality=modality,
            config=config
        )
        return

    # ==========================================
    # PATH B: TABULAR (CSV FILES)
    # ==========================================
    elif modality == 'tabular':
        try:
            use_agentic = input("Use Agentic AutoML pipeline? (y/n): ").strip().lower() == 'y'
        except EOFError:
            use_agentic = False
            
        if use_agentic:
            print("\n" + "="*80)
            print("PHASE 6.3: AGENTIC AUTOML PIPELINE")
            print("="*80)
            
            from agents.agent_orchestrator import AgenticAutoMLOrchestrator
            orchestrator = AgenticAutoMLOrchestrator()
            result = orchestrator.run_pipeline(user_input, force_run=True)
            
            if result:
                print(f"\n✅ Agentic pipeline complete!")
                print(f"📓 Notebook saved to: {result.get('notebook')}")
                print("\n🚀 Proceeding to Execute Auto ML/DL Pipeline based on Agentic Plan...")
                
                target_column = result['profile'].get('target_column')
                X, y, problem_type = load_local_dataset(user_input, target_column)
                if X is not None:
                    run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=os.path.basename(user_input), validate=args.validate, config=config)
            else:
                print("🛑 Agentic pipeline halted. Exiting.")
                return
        else:
            target_column = config.get("target_column")
            X, y, problem_type = load_local_dataset(user_input, target_column)
            if X is not None:
                run_single_dataset_pipeline(X, y, problem_type, store, encoder, did=os.path.basename(user_input), validate=args.validate, config=config)

if __name__ == "__main__":
    main()
