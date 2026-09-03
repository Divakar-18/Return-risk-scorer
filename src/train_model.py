import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_orders_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset, guarding against missing files."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"ERROR: Dataset not found at '{filepath}'.\n"
            f"Please run 'python src/generate_data.py' first to generate the synthetic data."
        )
    return pd.read_csv(filepath)

def prepare_order_features(orders_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepares features and target for modeling."""
    features_df = orders_df.drop(columns=['order_id', 'order_date', 'customer_id', 'returned'])
    features_df = pd.get_dummies(features_df, columns=['category', 'payment_method', 'region'], drop_first=True)
    target_series = orders_df['returned']
    return features_df, target_series

def get_chronological_split_indices(orders_df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Splits the data chronologically to respect the time dimension.
    Returns indices for train and test sets.
    """
    sorted_df = orders_df.sort_values('order_date').reset_index(drop=True)
    split_cutoff_index = int(len(sorted_df) * (1 - test_ratio))
    
    training_indices = sorted_df.index[:split_cutoff_index]
    testing_indices = sorted_df.index[split_cutoff_index:]
    
    return training_indices, testing_indices

def get_gradient_boosting_model(scale_pos_weight: float = 1.0):
    """
    Attempts to instantiate LightGBM. If not installed or fails, falls back gracefully to XGBoost.
    """
    try:
        import lightgbm as lgb
        print("Using LightGBM classifier as primary boosted model.")
        # Light hyperparameter tuning for defensibility (controlled depth and learning rate to prevent overfitting)
        return lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        ), "LightGBM"
    except ImportError:
        print("WARNING: LightGBM is not installed. Falling back gracefully to XGBoost...")
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            ), "XGBoost"
        except ImportError:
            raise ImportError(
                "CRITICAL: Neither LightGBM nor XGBoost could be loaded. "
                "Please ensure at least one is installed via requirements.txt."
            )

def run_time_series_cv(model, feature_matrix: pd.DataFrame, target_vector: pd.Series, n_splits: int = 5):
    """
    Executes a n-fold Time-Series Cross Validation (expanding window) and computes PR-AUC for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pr_auc_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(feature_matrix), 1):
        X_fold_train, X_fold_val = feature_matrix.iloc[train_idx], feature_matrix.iloc[val_idx]
        y_fold_train, y_fold_val = target_vector.iloc[train_idx], target_vector.iloc[val_idx]
        
        # Clone / fit model on fold
        model.fit(X_fold_train, y_fold_train)
        val_probs = model.predict_proba(X_fold_val)[:, 1]
        
        fold_pr_auc = average_precision_score(y_fold_val, val_probs)
        pr_auc_scores.append(fold_pr_auc)
        print(f"  Fold {fold_idx} PR-AUC: {fold_pr_auc:.4f}")
        
    return np.mean(pr_auc_scores), np.std(pr_auc_scores)

def main():
    print("=== Phase 3: Main Model Training & Cross Validation ===")
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_orders_data(dataset_path)
    
    print("Preparing features...")
    feature_matrix, target_vector = prepare_order_features(orders_df)
    
    print("Splitting data chronologically...")
    train_idx, test_idx = get_chronological_split_indices(orders_df)
    
    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train, y_test = target_vector.iloc[train_idx], target_vector.iloc[test_idx]
    
    print(f"  Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # 1. Baseline Model
    print("\n--- 1. Baseline Model (Logistic Regression) ---")
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_probs = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_preds = baseline_pipeline.predict(X_test)
    
    baseline_prec = precision_score(y_test, baseline_preds)
    baseline_rec = recall_score(y_test, baseline_preds)
    baseline_f1 = f1_score(y_test, baseline_preds)
    baseline_pr_auc = average_precision_score(y_test, baseline_probs)
    
    print(f"  Precision: {baseline_prec:.3f} | Recall: {baseline_rec:.3f} | F1: {baseline_f1:.3f} | PR-AUC: {baseline_pr_auc:.3f}")
    
    # 2. Main Boosted Model
    print("\n--- 2. Main Boosted Risk Scorer ---")
    # Calculate scale_pos_weight for class imbalance handling
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / max(1, pos_count)
    
    risk_scorer, model_name = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    
    # Run 5-fold Time Series Cross Validation
    print(f"\nRunning 5-Fold Time-Series Cross Validation for {model_name}...")
    mean_cv_pr_auc, std_cv_pr_auc = run_time_series_cv(risk_scorer, X_train, y_train, n_splits=5)
    print(f"  {model_name} 5-Fold Time-Series CV PR-AUC: {mean_cv_pr_auc:.4f} ± {std_cv_pr_auc:.4f}")
    
    # Fit main model on full training set
    print(f"\nFitting final {model_name} model on full training dataset...")
    risk_scorer.fit(X_train, y_train)
    
    boosted_probs = risk_scorer.predict_proba(X_test)[:, 1]
    boosted_preds = risk_scorer.predict(X_test)
    
    boosted_prec = precision_score(y_test, boosted_preds)
    boosted_rec = recall_score(y_test, boosted_preds)
    boosted_f1 = f1_score(y_test, boosted_preds)
    boosted_pr_auc = average_precision_score(y_test, boosted_probs)
    
    print(f"\nEvaluating {model_name} on Held-Out Test Set:")
    print(f"  Precision: {boosted_prec:.3f}")
    print(f"  Recall:    {boosted_rec:.3f}")
    print(f"  F1 Score:  {boosted_f1:.3f}")
    print(f"  PR-AUC:    {boosted_pr_auc:.3f}")
    
    print("\n=== Model Comparison Summary ===")
    print(f"Baseline (LogReg) PR-AUC : {baseline_pr_auc:.4f}")
    print(f"{model_name:<23} PR-AUC : {boosted_pr_auc:.4f}")
    print(f"PR-AUC Improvement       : +{(boosted_pr_auc - baseline_pr_auc):.4f}")

if __name__ == "__main__":
    main()
