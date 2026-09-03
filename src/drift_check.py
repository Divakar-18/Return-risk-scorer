import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score

from train_model import (
    get_chronological_split_indices,
    get_gradient_boosting_model,
    load_orders_data,
    prepare_order_features,
)

PR_AUC_FLOOR_PERCENTAGE = 0.85  # Flag recalibration if recent window PR-AUC drops > 15% from validation baseline

def check_concept_drift(baseline_val_pr_auc: float, recent_window_pr_auc: float, floor_ratio: float = PR_AUC_FLOOR_PERCENTAGE) -> dict:
    """
    Monitors model performance across rolling time windows.
    Triggers an alert if recent PR-AUC falls below the defined operational floor.
    """
    minimum_allowed_pr_auc = baseline_val_pr_auc * floor_ratio
    performance_drop = baseline_val_pr_auc - recent_window_pr_auc
    percentage_drop = (performance_drop / baseline_val_pr_auc) * 100
    
    if recent_window_pr_auc < minimum_allowed_pr_auc:
        status = "RECALIBRATION_NEEDED"
        alert_msg = (
            f"WARNING: Concept drift detected! Recent window PR-AUC ({recent_window_pr_auc:.4f}) "
            f"dropped by {percentage_drop:.1f}% below the minimum operational floor ({minimum_allowed_pr_auc:.4f}). "
            f"Model retrain / recalibration required."
        )
    else:
        status = "HEALTHY"
        alert_msg = f"Model performance is healthy. Recent PR-AUC ({recent_window_pr_auc:.4f}) meets operational floor ({minimum_allowed_pr_auc:.4f})."
        
    return {
        'status': status,
        'baseline_val_pr_auc': baseline_val_pr_auc,
        'recent_window_pr_auc': recent_window_pr_auc,
        'minimum_allowed_pr_auc': minimum_allowed_pr_auc,
        'percentage_drop': percentage_drop,
        'alert_msg': alert_msg
    }

def main():
    print("=== Phase 8: Failure Case #2 - Concept Drift Analysis & Monitoring ===")
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_orders_data(dataset_path)
    
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, test_idx = get_chronological_split_indices(orders_df)
    
    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train, y_test = target_vector.iloc[train_idx], target_vector.iloc[test_idx]
    
    # Train base model & calibrator on training set
    pos_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
    base_model, _ = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Early Validation Window (First half of test set)
    half_test_len = len(X_test) // 2
    X_val_early, X_test_late = X_test.iloc[:half_test_len], X_test.iloc[half_test_len:]
    y_val_early, y_test_late = y_test.iloc[:half_test_len], y_test.iloc[half_test_len:]
    
    # Calculate PR-AUC across both chronological windows
    early_probs = calibrated_model.predict_proba(X_val_early)[:, 1]
    late_probs = calibrated_model.predict_proba(X_test_late)[:, 1]
    
    early_pr_auc = average_precision_score(y_val_early, early_probs)
    late_pr_auc = average_precision_score(y_test_late, late_probs)
    
    print("\nEvaluating Chronological Performance Across Windows:")
    print(f"  Early Validation Window PR-AUC : {early_pr_auc:.4f}")
    print(f"  Late Recent Window PR-AUC       : {late_pr_auc:.4f}")
    print(f"  Performance Change             : {(late_pr_auc - early_pr_auc):+.4f}")
    
    # Run Monitoring Drift Check
    print("\nRunning Automated Drift Monitoring Check...")
    drift_report = check_concept_drift(early_pr_auc, late_pr_auc)
    
    print(f"  Status        : {drift_report['status']}")
    print(f"  Alert Message : {drift_report['alert_msg']}")
    
    print("\nPhase 8 (Concept Drift Monitoring) complete.")

if __name__ == "__main__":
    main()
