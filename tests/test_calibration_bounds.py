import os
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import (
    get_chronological_split_indices,
    get_gradient_boosting_model,
    load_orders_data,
    prepare_order_features,
)


def test_calibrated_probabilities_within_valid_bounds():
    """Verifies that calibrated output probabilities are strictly bounded within [0.0, 1.0]."""
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    assert os.path.exists(dataset_path), "Data file must exist for calibration bounds test"
    
    orders_df = load_orders_data(dataset_path)
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, test_idx = get_chronological_split_indices(orders_df)
    
    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train = target_vector.iloc[train_idx]
    
    pos_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
    base_model, _ = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]
    
    assert np.all(calibrated_probs >= 0.0), f"Found probability < 0.0: min is {np.min(calibrated_probs)}"
    assert np.all(calibrated_probs <= 1.0), f"Found probability > 1.0: max is {np.max(calibrated_probs)}"
    assert not np.isnan(calibrated_probs).any(), "Calibrated probabilities contain NaN values"
    assert not np.isinf(calibrated_probs).any(), "Calibrated probabilities contain Inf values"
