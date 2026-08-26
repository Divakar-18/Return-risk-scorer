import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from train_model import (
    load_orders_data,
    prepare_order_features,
    get_chronological_split_indices,
    get_gradient_boosting_model
)

# Explicit Cost Matrix (in INR ₹)
COST_FALSE_POSITIVE = 50.0  # Friction / lost margin on a good order unnecessarily flagged
COST_FALSE_NEGATIVE = 250.0 # Return shipping + restocking + depreciated item value on a missed return
COST_TRUE_POSITIVE = 0.0    # Successfully flagged order (mitigated risk)
COST_TRUE_NEGATIVE = 0.0    # Smooth checkout, no friction

def compute_total_cost(y_true, y_probs, threshold: float) -> float:
    """Computes total operational cost for a given decision threshold."""
    preds = (y_probs >= threshold).astype(int)
    
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    
    total_cost = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)
    return total_cost

def plot_cost_curve(thresholds, costs, optimal_thresh, optimal_cost, default_cost, save_path="data/threshold_cost_curve.png"):
    """Plots operational cost vs decision threshold curve."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, costs, label="Total Cost (₹)", color="purple", linewidth=2)
    plt.axvline(x=optimal_thresh, color="green", linestyle="--", label=f"Optimal Threshold ({optimal_thresh:.2f})")
    plt.axvline(x=0.5, color="red", linestyle=":", label="Default Threshold (0.50)")
    
    plt.scatter([optimal_thresh], [optimal_cost], color="green", s=100, zorder=5)
    plt.scatter([0.5], [default_cost], color="red", s=100, zorder=5)
    
    plt.xlabel("Decision Threshold")
    plt.ylabel("Total Expected Cost (₹)")
    plt.title("Cost-Sensitive Threshold Analysis")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Cost curve plot saved to '{save_path}'")

def main():
    print("=== Phase 5: Cost-Sensitive Threshold Analysis ===")
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_orders_data(dataset_path)
    
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, test_idx = get_chronological_split_indices(orders_df)
    
    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train, y_test = target_vector.iloc[train_idx], target_vector.iloc[test_idx]
    
    # Train base LightGBM and calibrate with Isotonic
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / max(1, pos_count)
    
    base_model, _ = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    
    print("Fitting base LightGBM model and Isotonic Calibrator...")
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    calibrated_test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Sweep thresholds from 0.00 to 1.00 in steps of 0.01
    thresholds = np.linspace(0.0, 1.0, 101)
    costs = [compute_total_cost(y_test, calibrated_test_probs, th) for th in thresholds]
    
    min_cost_idx = np.argmin(costs)
    optimal_threshold = thresholds[min_cost_idx]
    optimal_cost = costs[min_cost_idx]
    
    default_idx = int(0.5 * 100)
    default_cost = costs[default_idx]
    
    num_test_orders = len(y_test)
    cost_per_1k_optimal = (optimal_cost / num_test_orders) * 1000
    cost_per_1k_default = (default_cost / num_test_orders) * 1000
    savings_per_1k = cost_per_1k_default - cost_per_1k_optimal
    
    print("\n=== Cost Analysis Results (Test Set: 1,000 Orders) ===")
    print(f"  Default Threshold (0.50) Total Cost : INR {default_cost:,.2f}  (INR {cost_per_1k_default:,.2f} per 1,000 orders)")
    print(f"  Optimal Threshold ({optimal_threshold:.2f}) Total Cost : INR {optimal_cost:,.2f}  (INR {cost_per_1k_optimal:,.2f} per 1,000 orders)")
    print(f"  Net Savings per 1,000 Orders        : INR {savings_per_1k:,.2f} ({(savings_per_1k / cost_per_1k_default) * 100:.1f}% cost reduction)")
    
    plot_cost_curve(thresholds, costs, optimal_threshold, optimal_cost, default_cost)
    
    print("\nPhase 5 (Threshold Analysis) complete.")

if __name__ == "__main__":
    main()
