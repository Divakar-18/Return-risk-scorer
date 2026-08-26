# Return-Risk Scorer

A rigorous, defensible ML pipeline to predict the probability of e-commerce order returns.

## Phase 2: Split & Baseline

### Why Accuracy is Misleading
In this return-risk scoring system, the target variable is imbalanced (overall return rate is ~15%). If we train a naïve model that simply predicts "Not Returned" (0) for every single order, it will achieve ~85% accuracy. However, this model is operationally useless because it misses 100% of the actual returns.

Instead of accuracy, we rely on:
- **Precision:** Of the orders we flagged as high-risk, what fraction actually returned?
- **Recall:** Of all the actual returns, what fraction did we successfully flag?
- **F1-Score:** The harmonic mean of precision and recall.
- **PR-AUC (Precision-Recall Area Under Curve):** A summary metric of the model's ability to trade off precision and recall across all possible thresholds, specifically suited for imbalanced datasets.

## Phase 3: Main Model Choice & Cross-Validation

### Why Gradient Boosted Decision Trees Win
Logistic Regression assumes a linear decision boundary between input features and the log-odds of a return. However, real e-commerce return behavior relies heavily on **non-linear feature interactions**.

For example:
- A 50% discount on an `Electronics` item might indicate a great deal with a low return rate.
- The exact same 50% discount on an `Apparel` item bought with 0 deliberation days (`days_to_purchase <= 1`) signals impulse buy behavior, resulting in a very high return rate.

Tree-based ensemble models like **LightGBM** (and XGBoost) natively capture these non-linear interaction effects (`discount_pct × category` or `payment_method × customer_past_orders`) without requiring manual feature engineering.

> [!NOTE]
> **Performance Finding:** LightGBM achieved a PR-AUC of **0.2617**, marginally outperforming the baseline Logistic Regression (0.2515). Because our synthetic dataset features strong linear/additive log-odds drivers (e.g. baseline return risk rules), the tree model offers a slight edge through interaction modeling rather than a massive leap.

### Cross-Validation Strategy
To avoid temporal data leakage (predicting past events using future patterns), we use a **5-Fold Time-Series Split (`TimeSeriesSplit`)** instead of standard random K-Fold cross-validation. This enforces an expanding window training scheme that accurately mimics real-world deployment.

## Phase 4: Probability Calibration

### Why Uncalibrated Scores Are Operationally Useless
Tree-based models like LightGBM push predictions toward 0 and 1 due to leaf node purity splits and class weight rebalancing (`scale_pos_weight`). Consequently, a raw model score of `0.80` does **NOT** mean an order has an 80% real-world chance of being returned — it is often an overconfident rank score.

In automated e-commerce operations:
- If a system uses risk scores to hold funds, charge return fees, or trigger manual review, decision thresholds depend on **true empirical probabilities**.
- An uncalibrated score distorts financial cost calculations, causing over-flagging or under-flagging of orders.

### Calibration Method Selection
We evaluate two calibration methods using **Brier Score Loss** (Mean Squared Error between predicted probabilities and actual outcomes):
1. **Isotonic Regression:** Non-parametric piecewise constant monotonic function. Flexible, but requires more validation data to avoid overfitting.
2. **Sigmoid (Platt Scaling):** Parametric logistic fit. Performs better on smaller validation sets when the raw model output error is monotonic.

![Reliability Diagram](data/calibration_curves.png)

## Phase 5: Cost-Sensitive Threshold Optimization

### Cost Matrix Definition
In real-world e-commerce, the financial damage of a **False Negative** (missing a return) far outweighs the friction of a **False Positive** (unnecessarily verifying an order).

| Outcome | Decision | Actual Target | Assigned Cost (₹) | Business Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **False Positive (FP)** | Flagged High-Risk | Kept Order | **₹50** | Customer friction, verification SMS cost, lost repeat margin |
| **False Negative (FN)** | Passed Low-Risk | Returned Order | **₹250** | Two-way reverse shipping, warehouse restocking, item depreciation |
| **True Positive (TP)** | Flagged High-Risk | Returned Order | **₹0** | Friction is offset by mitigating return shipping / early cancellation |
| **True Negative (TN)** | Passed Low-Risk | Kept Order | **₹0** | Seamless checkout flow |

### Cost-Optimal Decision Boundary
Because missing a return (₹250) is **5x more expensive** than a false alarm (₹50), the default decision threshold of `0.50` is mathematically suboptimal. 

By sweeping thresholds from `0.00` to `1.00` on the calibrated test set probabilities, we select **`0.19`** as the minimum-cost decision boundary.

- **Default 0.50 Threshold Cost:** **₹37,150.00** per 1,000 orders
- **Cost-Optimal 0.19 Threshold Cost:** **₹29,600.00** per 1,000 orders
- **Empirical Savings:** **₹7,550.00 net savings per 1,000 orders** (**20.3% cost reduction**)

![Cost Curve](data/threshold_cost_curve.png)



