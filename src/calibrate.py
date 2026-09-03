import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# Import existing functions from train_model
from train_model import (
    get_chronological_split_indices,
    get_gradient_boosting_model,
    load_orders_data,
    prepare_order_features,
)


def evaluate_calibration(y_true, y_prob, name="Model"):
    """Computes key calibration metrics: Brier Score Loss and Log Loss."""
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    print(f"  {name:<25} | Brier Score: {brier:.4f} | Log Loss: {logloss:.4f}")
    return brier, logloss


def plot_reliability_diagrams(y_test, raw_probs, isotonic_probs, sigmoid_probs, save_path="data/calibration_curves.png"):
    """Plots reliability diagrams comparing uncalibrated raw model with isotonic and sigmoid calibrations."""
    plt.figure(figsize=(10, 6))

    # Perfectly calibrated reference line
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.7)

    # 1. Uncalibrated Raw Model
    fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(y_test, raw_probs, n_bins=10)
    plt.plot(mean_predicted_value_raw, fraction_of_positives_raw, "s-", label="Raw LightGBM", color="red")

    # 2. Isotonic Calibration
    fraction_of_positives_iso, mean_predicted_value_iso = calibration_curve(y_test, isotonic_probs, n_bins=10)
    plt.plot(mean_predicted_value_iso, fraction_of_positives_iso, "o-", label="Isotonic Calibration", color="blue")

    # 3. Sigmoid Calibration
    fraction_of_positives_sig, mean_predicted_value_sig = calibration_curve(y_test, sigmoid_probs, n_bins=10)
    plt.plot(mean_predicted_value_sig, fraction_of_positives_sig, "^-", label="Sigmoid Calibration", color="green")

    plt.ylabel("Fraction of Positives (Actual Return Rate)")
    plt.xlabel("Mean Predicted Probability")
    plt.title("Reliability Diagram (Calibration Curves)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nReliability diagram saved to '{save_path}'")


def main():
    print("=== Phase 4: Model Calibration Analysis ===")
    dataset_path = os.path.join("data", "synthetic_orders.csv")
    orders_df = load_orders_data(dataset_path)

    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, test_idx = get_chronological_split_indices(orders_df)

    X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
    y_train, y_test = target_vector.iloc[train_idx], target_vector.iloc[test_idx]

    # Split train further into train_base and val for calibration fit/selection
    val_split_idx = int(len(X_train) * 0.75)
    X_train_base, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_train_base, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]

    pos_count = np.sum(y_train_base == 1)
    neg_count = np.sum(y_train_base == 0)
    pos_weight = neg_count / max(1, pos_count)

    # 1. Fit Base Raw LightGBM Model
    raw_model, _model_name = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    raw_model.fit(X_train_base, y_train_base)
    raw_val_probs = raw_model.predict_proba(X_val)[:, 1]
    raw_test_probs = raw_model.predict_proba(X_test)[:, 1]

    print("\nEvaluating Base Model Calibration on Validation Set...")
    _brier_raw, _ = evaluate_calibration(y_val, raw_val_probs, "Raw LightGBM")

    # 2. Fit Isotonic Calibration
    print("\nFitting Isotonic Calibrator on Validation Set...")
    calibrated_iso = CalibratedClassifierCV(estimator=raw_model, method="isotonic", cv=3)
    calibrated_iso.fit(X_val, y_val)
    iso_val_probs = calibrated_iso.predict_proba(X_val)[:, 1]
    iso_test_probs = calibrated_iso.predict_proba(X_test)[:, 1]
    brier_iso, _ = evaluate_calibration(y_val, iso_val_probs, "Isotonic Calibration")

    # 3. Fit Sigmoid (Platt Scaling) Calibrator on Validation Set
    print("\nFitting Sigmoid (Platt Scaling) Calibrator on Validation Set...")
    calibrated_sig = CalibratedClassifierCV(estimator=raw_model, method="sigmoid", cv=3)
    calibrated_sig.fit(X_val, y_val)
    sig_val_probs = calibrated_sig.predict_proba(X_val)[:, 1]
    sig_test_probs = calibrated_sig.predict_proba(X_test)[:, 1]
    brier_sig, _ = evaluate_calibration(y_val, sig_val_probs, "Sigmoid Calibration")

    # Pick the best performing calibrator based on Brier Score Loss on validation set
    if brier_iso < brier_sig:
        best_method = "Isotonic"
    else:
        best_method = "Sigmoid"

    print(f"\nSelection: {best_method} Calibration performed better on validation data (Lower Brier Score).")

    print("\n=== Final Test Set Calibration Performance ===")
    evaluate_calibration(y_test, raw_test_probs, "Raw LightGBM (Test)")
    evaluate_calibration(y_test, iso_test_probs, "Isotonic (Test)")
    evaluate_calibration(y_test, sig_test_probs, "Sigmoid (Test)")

    # Plot reliability diagram
    plot_reliability_diagrams(y_test, raw_test_probs, iso_test_probs, sig_test_probs)

    print("\nPhase 4 (Calibration) complete.")


if __name__ == "__main__":
    main()
