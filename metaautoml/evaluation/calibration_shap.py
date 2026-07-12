import shap
import matplotlib.pyplot as plt
import wandb

class AutoDLEvaluator:
    def evaluate_final_model(self, model, X_val, y_val):
        # 1. Phase 5.4: Confidence + ECE
        confidences = model.predict_proba(X_val).max(axis=1)
        predictions = model.predict(X_val)
        ece = self._calculate_ece(confidences, predictions, y_val)
        
        wandb.log({
            "confidence_score": confidences.mean(),
            "actual_score": (predictions == y_val).mean(),
            "ece": ece,
            "calibration_gap": abs(confidences.mean() - (predictions == y_val).mean())
        })
        
        # 2. Phase 5.5: SHAP Explainability
        # Note: SHAP for tree models on embeddings is highly effective
        explainer = shap.TreeExplainer(model) 
        shap_values = explainer.shap_values(X_val)
        
        plt.figure()
        shap.summary_plot(shap_values, X_val, show=False)
        wandb.log({"shap_summary": wandb.Image(plt)})
        plt.close()
