import os
import time
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

from train_model import (
    load_orders_data,
    prepare_order_features,
    get_chronological_split_indices,
    get_gradient_boosting_model
)

OPTIMAL_THRESHOLD = 0.19

def generate_template_explanation(top_features: list[tuple[str, float]]) -> str:
    """Fallback plain-English explanation generator using raw SHAP feature contributions."""
    feature_names = [f[0].replace('_', ' ') for f in top_features]
    if len(feature_names) == 3:
        return f"High return risk driven primarily by {feature_names[0]}, {feature_names[1]}, and {feature_names[2]}."
    elif len(feature_names) == 2:
        return f"High return risk driven primarily by {feature_names[0]} and {feature_names[1]}."
    elif len(feature_names) == 1:
        return f"High return risk driven primarily by {feature_names[0]}."
    return "High return risk detected based on historical order characteristics."

def explain_with_llm(top_features: list[tuple[str, float]], api_key: str = None) -> str:
    """
    Attempts to call Anthropic API (claude-sonnet-4-6) to turn top 3 SHAP features into a single concise sentence.
    Retries once on failure, then gracefully falls back to template explanation.
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
    if not api_key:
        # No API key present -> immediate fallback to template
        return generate_template_explanation(top_features)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        feature_desc = ", ".join([f"{name} (SHAP value: {val:+.3f})" for name, val in top_features])
        prompt = (
            f"You are an AI e-commerce risk analyst. Summarize the following top risk factors for an order into "
            f"EXACTLY ONE professional, clear plain-English sentence for a merchant dashboard:\n"
            f"Risk factors: {feature_desc}\n"
            f"Output only the one sentence explanation without conversational filler."
        )
        
        # Try once with retries
        for attempt in range(2):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                explanation = response.content[0].text.strip()
                return explanation
            except Exception as retry_err:
                if attempt == 0:
                    time.sleep(1) # wait 1s before retrying
                    continue
                else:
                    print(f"  [LLM API Warning] Anthropic API failed after retry ({retry_err}). Falling back to template explanation.")
                    return generate_template_explanation(top_features)
                    
    except ImportError:
        print("  [LLM API Warning] Anthropic SDK not found. Falling back to template explanation.")
        return generate_template_explanation(top_features)

def compute_shap_explanations(model, X_test: pd.DataFrame, threshold: float = OPTIMAL_THRESHOLD):
    """
    Runs SHAP explainer on high-risk flagged orders.
    Returns top 3 contributing features per flagged order.
    """
    # Create TreeExplainer on base LightGBM model
    # CalibratedClassifierCV wraps estimator in calibrated_classifiers_
    if hasattr(model, 'calibrated_classifiers_'):
        base_estimator = model.calibrated_classifiers_[0].estimator
    else:
        base_estimator = model
        
    explainer = shap.TreeExplainer(base_estimator)
    
    # Calculate probabilities
    probs = model.predict_proba(X_test)[:, 1]
    flagged_indices = np.where(probs >= threshold)[0]
    
    print(f"Total Test Orders: {len(X_test)} | Flagged High-Risk (>= {threshold}): {len(flagged_indices)}")
    
    if len(flagged_indices) == 0:
        return []

    X_flagged = X_test.iloc[flagged_indices]
    shap_values = explainer.shap_values(X_flagged)
    
    # Handle single output array vs multi-class list output in SHAP
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # positive class
        
    explanations_summary = []
    
    for i, idx in enumerate(flagged_indices):
        order_shap = shap_values[i]
        feature_names = X_test.columns
        
        # Get top 3 positive contributing SHAP features
        top_3_idx = np.argsort(order_shap)[::-1][:3]
        top_features = [(feature_names[j], float(order_shap[j])) for j in top_3_idx]
        
        # Narrate explanation
        narrative = explain_with_llm(top_features)
        
        explanations_summary.append({
            'test_index': idx,
            'risk_score': probs[idx],
            'top_features': top_features,
            'narrative': narrative
        })
        
    return explanations_summary

def main():
    print("=== Phase 6: Explainability Layer (SHAP + LLM Narration) ===")
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_orders_data(dataset_path)
    
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, test_idx = get_chronological_split_indices(orders_df)
    
    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train, y_test = target_vector.iloc[train_idx], target_vector.iloc[test_idx]
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / max(1, pos_count)
    
    base_model, _ = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    
    print("Training base LightGBM model and Isotonic Calibrator...")
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    print(f"\nComputing SHAP explanations for orders flagged high-risk at threshold = {OPTIMAL_THRESHOLD}...")
    explanations = compute_shap_explanations(calibrated_model, X_test, threshold=OPTIMAL_THRESHOLD)
    
    print("\n--- Sample Explanations (First 5 Flagged Orders) ---")
    for exp in explanations[:5]:
        print(f"Order Index: {exp['test_index']} | Risk Score: {exp['risk_score']:.3f}")
        print(f"  Top 3 SHAP Features: {exp['top_features']}")
        print(f"  Narrative: \"{exp['narrative']}\"\n")
        
    print("Phase 6 (Explainability Layer) complete.")

if __name__ == "__main__":
    main()
