import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from train_model import (
    get_chronological_split_indices,
    get_gradient_boosting_model,
    load_orders_data,
    prepare_order_features,
)

OPTIMAL_THRESHOLD = 0.19

def evaluate_order_risk(order_dict: dict, calibrated_model, feature_columns: list[str]) -> dict:
    """
    Evaluates risk score for an incoming order and applies business fallback rules.
    
    Rule-Based Cold-Start Fallback:
    If customer_past_orders == 0, the model lacks historical return signal and may assign an 
    overly optimistic score. Any first-time customer placing a Cash-on-Delivery (COD) order 
    is automatically routed to 'Manual Review' rather than blindly trusted.
    """
    # Ensure optional categorical columns exist in order_dict with defaults
    default_order = {
        'category': 'Apparel',
        'payment_method': 'COD',
        'region': 'North',
        'is_size_sensitive': 1,
        'price': 1000.0,
        'discount_pct': 0.0,
        'customer_past_orders': 0,
        'customer_past_return_rate': 0.0,
        'delivery_days': 5,
        'days_to_purchase': 2
    }
    merged_order = {**default_order, **order_dict}
    order_df = pd.DataFrame([merged_order])
    
    # Identify present categorical columns to encode
    cat_cols = [c for c in ['category', 'payment_method', 'region'] if c in order_df.columns]
    encoded_df = pd.get_dummies(order_df, columns=cat_cols)
    
    # Reindex columns to match feature matrix exactly
    for col in feature_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    encoded_df = encoded_df[feature_columns]
    
    # Model probability
    probs_array = np.asarray(calibrated_model.predict_proba(encoded_df))
    model_prob = float(probs_array[:, 1][0])
    
    # Apply Business Fallback Rule (Cold Start Guard)
    is_cold_start = (order_dict.get('customer_past_orders', 0) == 0)
    is_cod = (order_dict.get('payment_method') == 'COD')
    
    if is_cold_start and is_cod:
        decision = "MANUAL_REVIEW"
        reason = "Cold-start fallback rule triggered: First-time customer with COD payment."
    elif model_prob >= OPTIMAL_THRESHOLD:
        decision = "FLAG_HIGH_RISK"
        reason = f"Calibrated model probability ({model_prob:.3f}) exceeds threshold ({OPTIMAL_THRESHOLD})."
    else:
        decision = "PASS_LOW_RISK"
        reason = f"Calibrated model probability ({model_prob:.3f}) below threshold ({OPTIMAL_THRESHOLD})."
        
    return {
        'order_id': order_dict.get('order_id', 'UNKNOWN'),
        'model_prob': model_prob,
        'decision': decision,
        'reason': reason,
        'is_cold_start': is_cold_start
    }

def main():
    print("=== Phase 7: Failure Case #1 - Cold Start Analysis ===")
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_orders_data(dataset_path)
    
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, _test_idx = get_chronological_split_indices(orders_df)
    
    X_train = feature_matrix.iloc[train_idx]
    y_train = target_vector.iloc[train_idx]
    
    pos_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
    base_model, _ = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Construct a real Cold-Start Test Order
    cold_start_order = {
        'order_id': 'COLD_START_999',
        'category': 'Apparel',
        'is_size_sensitive': 1,
        'price': 2500.0,
        'discount_pct': 0.40,
        'customer_id': 9999,
        'customer_past_orders': 0,        # Zero past history!
        'customer_past_return_rate': 0.0, # No past return signal
        'payment_method': 'COD',          # High risk combination
        'region': 'North',
        'delivery_days': 8,
        'days_to_purchase': 0             # Impulse buy
    }
    
    print("\nEvaluating Cold-Start Order through Raw Pipeline...")
    raw_result = evaluate_order_risk(cold_start_order, calibrated_model, feature_matrix.columns)
    
    print(f"  Order ID     : {raw_result['order_id']}")
    print(f"  Model Prob   : {raw_result['model_prob']:.4f}")
    print(f"  Final Action : {raw_result['decision']}")
    print(f"  Rationale    : {raw_result['reason']}")
    
    print("\nPhase 7 (Cold Start Failure Analysis & Rule Fix) complete.")

if __name__ == "__main__":
    main()
