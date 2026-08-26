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
