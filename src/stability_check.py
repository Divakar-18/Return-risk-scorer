import os
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

SEEDS = [42, 1042, 2042]

def get_top_shap_features(dataset_path: str, seed: int, top_n: int = 10) -> list[str]:
    """Retrains the risk model under a specific seed and dataset, returning top N SHAP features."""
    orders_df = load_orders_data(dataset_path)
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, test_idx = get_chronological_split_indices(orders_df)
    
    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train = target_vector.iloc[train_idx]
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / max(1, pos_count)
    
    base_model, _ = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    # Set model seed
    if hasattr(base_model, 'random_state'):
        base_model.random_state = seed
        
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Compute mean absolute SHAP values across test set
    base_estimator = calibrated_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_estimator)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_names = X_test.columns
    
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    return top_features

def compute_top_k_overlap(feature_lists: list[list[str]], top_k: int = 10) -> float:
    """Computes the mean pairwise Jaccard overlap percentage across seed runs."""
    sets = [set(fl[:top_k]) for fl in feature_lists]
    pairwise_overlaps = []
    
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            intersection = len(sets[i].intersection(sets[j]))
            union = len(sets[i].union(sets[j]))
            overlap_pct = (intersection / union) * 100
            pairwise_overlaps.append(overlap_pct)
            
    return float(np.mean(pairwise_overlaps))

def main():
    print("=== Phase 9: Feature Stability Analysis Across Random Seeds ===")
    
    seed_top_features = {}
    feature_lists = []
    
    for seed in SEEDS:
        filename = "synthetic_orders.csv" if seed == 42 else f"synthetic_orders_seed_{seed}.csv"
        dataset_path = os.path.join("data", filename)
        
        print(f"\nEvaluating SHAP Feature Importance for Seed {seed}...")
        top_10 = get_top_shap_features(dataset_path, seed=seed, top_n=10)
        seed_top_features[seed] = top_10
        feature_lists.append(top_10)
        
        print(f"  Seed {seed} Top-5 Features: {top_10[:5]}")
        
    stability_overlap_pct = compute_top_k_overlap(feature_lists, top_k=10)
    
    print("\n=== Feature Stability Summary ===")
    print(f"  Top-10 Feature Overlap Across Seeds: {stability_overlap_pct:.1f}%")
    if stability_overlap_pct >= 80.0:
        print("  Verdict: HIGHLY STABLE. Model feature attribution is robust to random seed variations.")
    else:
        print("  Verdict: SEED SENSITIVE. Model feature attribution varies noticeably across random seeds.")
        
    print("\nPhase 9 (Feature Stability Check) complete.")

if __name__ == "__main__":
    main()
