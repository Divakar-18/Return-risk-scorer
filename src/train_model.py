import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset, guarding against missing files."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"ERROR: Dataset not found at '{filepath}'.\n"
            f"Please run 'python src/generate_data.py' first to generate the synthetic data."
        )
    return pd.read_csv(filepath)

def prepare_features(orders_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepares features and target for modeling."""
    # Convert dates to numerical values if needed, or drop
    # For now, drop order_date, order_id, customer_id as they are identifiers/dates
    # One-hot encode categorical variables
    features = orders_df.drop(columns=['order_id', 'order_date', 'customer_id', 'returned'])
    features = pd.get_dummies(features, columns=['category', 'payment_method', 'region'], drop_first=True)
    target = orders_df['returned']
    return features, target

def time_based_split(orders_df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits the data chronologically to respect the time dimension.
    Returns indices for train and test sets to be applied after feature engineering.
    """
    # Ensure it's sorted by date
    orders_df = orders_df.sort_values('order_date').reset_index(drop=True)
    split_idx = int(len(orders_df) * (1 - test_size))
    
    train_indices = orders_df.index[:split_idx]
    test_indices = orders_df.index[split_idx:]
    
    return train_indices, test_indices

def main():
    print("Loading data...")
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_data(dataset_path)
    
    print("Preparing features...")
    X, y = prepare_features(orders_df)
    
    print("Splitting data chronologically...")
    train_idx, test_idx = time_based_split(orders_df)
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set:     {len(X_test)} samples")
    
    print("\nTraining Logistic Regression Baseline...")
    # Baseline: Logistic Regression with balanced class weights
    # Balanced weights heavily penalize missing the minority class (returns)
    # We use a pipeline with StandardScaler to ensure convergence and fair regularization
    baseline_model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    baseline_model.fit(X_train, y_train)
    
    print("Evaluating Baseline Model on Test Set...")
    y_pred = baseline_model.predict(X_test)
    y_prob = baseline_model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)
    
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  PR-AUC:    {pr_auc:.3f}")
    
    print("\nPhase 2 (Baseline) complete.")

if __name__ == "__main__":
    main()
