import numpy as np


def compute_expected_cost_by_threshold(y_true: np.ndarray, y_probs: np.ndarray, threshold: float, cost_fp: float = 50.0, cost_fn: float = 250.0) -> float:
    """Computes total expected cost for a specific decision threshold."""
    preds = (y_probs >= threshold).astype(int)
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    return float((fp * cost_fp) + (fn * cost_fn))

def test_cost_minimization_optimal_threshold():
    """Verifies that cost-minimization function correctly selects the minimum cost threshold."""
    # Synthetic target: 10 orders, 3 returns (indices 2, 5, 8)
    y_true = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    y_probs = np.array([0.05, 0.10, 0.25, 0.12, 0.08, 0.30, 0.02, 0.15, 0.22, 0.05])
    
    thresholds = np.linspace(0.0, 1.0, 101)
    costs = [compute_expected_cost_by_threshold(y_true, y_probs, th) for th in thresholds]
    
    optimal_threshold = thresholds[np.argmin(costs)]
    
    # At threshold between 0.16 and 0.22, all 3 returns (probs 0.25, 0.30, 0.22) are flagged (0 FN), and 0 non-returns are flagged (0 FP). Cost = 0.
    assert 0.15 <= optimal_threshold <= 0.23, f"Expected optimal threshold near 0.16-0.22, got {optimal_threshold}"
    assert np.min(costs) == 0.0, f"Expected min cost of 0.0 on separable toy curve, got {np.min(costs)}"
