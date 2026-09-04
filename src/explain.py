import os
import time

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

from train_model import (
    get_chronological_split_indices,
    get_gradient_boosting_model,
    load_orders_data,
    prepare_order_features,
)

OPTIMAL_THRESHOLD = 0.19

def format_feature_for_human(feature_name: str, feature_value: float) -> str:
    """
    Translates raw feature column names and values into unambiguous human-readable business terms.
    Fixes the classic dummy-variable interpretability bug where dummy=0 (e.g. Prepaid=False -> COD)
    is mislabeled in naive SHAP outputs.
    """
    if feature_name == "payment_method_Prepaid":
        return "Payment Method: COD" if not bool(feature_value) else "Payment Method: Prepaid"
    elif feature_name == "is_size_sensitive":
        return "Size-Sensitive Category" if bool(feature_value) else "Non-Size-Sensitive Category"
    elif feature_name == "customer_past_orders":
        if int(feature_value) == 0:
            return "First-Time Customer (0 past orders)"
        return f"{int(feature_value)} past orders"
    elif feature_name == "customer_past_return_rate":
        return f"Historical Return Rate ({feature_value * 100:.1f}%)"
    elif feature_name == "discount_pct":
        return f"High Discount ({feature_value * 100:.0f}%)"
    elif feature_name == "delivery_days":
        return f"Long Delivery Time ({int(feature_value)} days)"
    elif feature_name == "days_to_purchase":
        if int(feature_value) <= 1:
            return "Impulse Purchase (<= 1 day deliberation)"
        return f"{int(feature_value)} days deliberation"
    elif feature_name.startswith("category_"):
        cat_name = feature_name.replace("category_", "")
        return f"Category: {cat_name}" if bool(feature_value) else f"Category: Not {cat_name}"
    elif feature_name.startswith("region_"):
        region_name = feature_name.replace("region_", "")
        return f"Region: {region_name}"
    
    return f"{feature_name.replace('_', ' ')} ({feature_value})"

def generate_template_explanation(top_features: list[tuple[str, float]]) -> str:
    """Fallback plain-English explanation generator using mapped business features."""
    feature_descriptions = [f[0] for f in top_features]
    if len(feature_descriptions) == 3:
        return f"High return risk driven primarily by {feature_descriptions[0]}, {feature_descriptions[1]}, and {feature_descriptions[2]}."
    elif len(feature_descriptions) == 2:
        return f"High return risk driven primarily by {feature_descriptions[0]} and {feature_descriptions[1]}."
    elif len(feature_descriptions) == 1:
        return f"High return risk driven primarily by {feature_descriptions[0]}."
    return "High return risk detected based on historical order characteristics."

def explain_with_llm(top_features: list[tuple[str, float]], api_key: str | None = None) -> tuple[str, str]:
    """Generate narration using Anthropic first, Groq second, or a local template."""
    feature_desc = ", ".join([f"{name} (SHAP contribution: {val:+.3f})" for name, val in top_features])
    prompt = (
        f"You are an AI e-commerce risk analyst for Razorpay. Summarize the following top risk factors for an order into "
        f"EXACTLY ONE professional, clear plain-English sentence for a merchant dashboard:\n"
        f"Risk factors: {feature_desc}\n"
        f"Output only the one sentence explanation without conversational filler."
    )

    anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            for attempt in range(2):
                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=100,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text.strip(), "Anthropic API (claude-sonnet-4-6)"
                except Exception as retry_err:  # noqa: BLE001
                    if attempt == 0:
                        time.sleep(1)
                    else:
                        print(f"  [LLM API Warning] Anthropic API call failed after retry: {retry_err}")
        except ImportError:
            print("  [LLM API Warning] Anthropic SDK not found; trying Groq.")
        except Exception as anthropic_err:  # noqa: BLE001
            print(f"  [LLM API Warning] Anthropic setup failed: {anthropic_err}")

    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            response = client.chat.completions.create(
                model="groq/compound-mini",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Groq response contained no message content")
            return content.strip(), "Groq API (groq/compound-mini)"
        except ImportError:
            print("  [LLM API Warning] OpenAI SDK not found; using template fallback.")
        except Exception as groq_err:  # noqa: BLE001
            print(f"  [LLM API Warning] Groq API call failed: {groq_err}")

    fallback_mode = "Template Fallback (No API Key in Env)" if not (anthropic_key or groq_key) else "Template Fallback (API Calls Failed)"
    return generate_template_explanation(top_features), fallback_mode

def compute_shap_explanations(model, X_test: pd.DataFrame, threshold: float = OPTIMAL_THRESHOLD):
    """
    Runs SHAP explainer on high-risk flagged orders.
    Maps one-hot dummy variables to true business meanings before generating explanations.
    """
    if hasattr(model, 'calibrated_classifiers_'):
        base_estimator = model.calibrated_classifiers_[0].estimator
    else:
        base_estimator = model
        
    explainer = shap.TreeExplainer(base_estimator)
    
    probs = model.predict_proba(X_test)[:, 1]
    flagged_indices = np.where(probs >= threshold)[0]
    
    print(f"Total Test Orders: {len(X_test)} | Flagged High-Risk (>= {threshold}): {len(flagged_indices)}")
    
    if len(flagged_indices) == 0:
        return []

    X_flagged = X_test.iloc[flagged_indices]
    shap_values = explainer.shap_values(X_flagged)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    explanations_summary = []
    
    for i, idx in enumerate(flagged_indices):
        order_shap = shap_values[i]
        order_row = X_flagged.iloc[i]
        feature_names = X_test.columns
        
        # Get top 3 positive contributing SHAP features
        top_3_idx = np.argsort(order_shap)[::-1][:3]
        
        # Map raw feature + value to human business term
        top_features = []
        for j in top_3_idx:
            raw_name = feature_names[j]
            raw_val = order_row[raw_name]
            human_term = format_feature_for_human(raw_name, raw_val)
            shap_val = float(order_shap[j])
            top_features.append((human_term, shap_val, raw_name, raw_val))
        
        # Format for narration
        narration_features = [(tf[0], tf[1]) for tf in top_features]
        narrative, mode = explain_with_llm(narration_features)
        
        explanations_summary.append({
            'test_index': idx,
            'risk_score': probs[idx],
            'top_features': top_features,
            'narrative': narrative,
            'narration_mode': mode
        })
        
    return explanations_summary

def main():
    print("=== Phase 6: Explainability Layer (SHAP + LLM Narration) ===")
    
    anthropic_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    groq_key_present = bool(os.environ.get("GROQ_API_KEY"))
    print(f"Anthropic API Key Status: {'PRESENT' if anthropic_key_present else 'NOT SET'}")
    print(f"Groq API Key Status: {'PRESENT' if groq_key_present else 'NOT SET'}")
    
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_orders_data(dataset_path)
    
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, test_idx = get_chronological_split_indices(orders_df)
    
    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train, _y_test = target_vector.iloc[train_idx], target_vector.iloc[test_idx]
    
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
        print(f"Order Index: {exp['test_index']} | Risk Score: {exp['risk_score']:.3f} | Mode: {exp['narration_mode']}")
        print("  Top 3 SHAP Features (Human-Mapped):")
        for tf in exp['top_features']:
            print(f"    - {tf[0]:<40} (Raw: {tf[2]}={tf[3]}, SHAP: {tf[1]:+.3f})")
        print(f"  Narrative: \"{exp['narrative']}\"\n")
        
    print("Phase 6 (Explainability Layer) complete.")

if __name__ == "__main__":
    main()
