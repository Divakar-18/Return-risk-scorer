import os
import sys
from html import escape

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.dirname(__file__))

from explain import explain_with_llm, format_feature_for_human  # noqa: E402
from predict import evaluate_order_risk  # noqa: E402
from train_model import (  # noqa: E402
    get_chronological_split_indices,
    get_gradient_boosting_model,
    load_orders_data,
    prepare_order_features,
)


OPTIMAL_THRESHOLD = 0.19
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_orders.csv")


@st.cache_resource
def load_calibrated_pipeline():
    orders_df = load_orders_data(DATASET_PATH)
    feature_matrix, target_vector = prepare_order_features(orders_df)
    train_idx, _test_idx = get_chronological_split_indices(orders_df)
    X_train = feature_matrix.iloc[train_idx]
    y_train = target_vector.iloc[train_idx]
    pos_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
    base_model, _model_name = get_gradient_boosting_model(scale_pos_weight=pos_weight)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method="isotonic", cv=5
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model, feature_matrix.columns.tolist()


def get_shap_features(calibrated_model, order: dict, feature_columns: list[str]):
    order_df = pd.DataFrame([order])
    encoded_order = pd.get_dummies(
        order_df, columns=["category", "payment_method", "region"]
    )
    for column in feature_columns:
        if column not in encoded_order:
            encoded_order[column] = 0
    encoded_order = encoded_order[feature_columns]

    base_estimator = calibrated_model.calibrated_classifiers_[0].estimator
    shap_values = shap.TreeExplainer(base_estimator).shap_values(encoded_order)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values[0]
    top_indices = np.argsort(values)[::-1][:3]
    return [
        (
            format_feature_for_human(feature_columns[index], encoded_order.iloc[0, index]),
            float(values[index]),
        )
        for index in top_indices
    ]


st.set_page_config(page_title="Return Risk Scorer", page_icon="R", layout="centered")
st.markdown(
    """
    <style>
    :root {
        --accent: #31566b;
        --text: #2f3a40;
        --muted: #707b82;
        --line: #d9dee2;
        --approve: #4f7d61;
        --danger: #a55353;
    }
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {
        color: var(--accent);
        font-weight: 600;
    }
    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 2rem 2rem 1.5rem;
    }
    [data-testid="stForm"] label {
        color: var(--muted);
        font-size: 0.88rem;
    }
    [data-testid="stForm"] [data-testid="stVerticalBlock"] {
        gap: 1rem;
    }
    [data-testid="stForm"] button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
        margin-top: 1rem;
    }
    .risk-card {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 2rem 2rem 1.75rem;
        margin-top: 2rem;
        background-color: #ffffff !important;
        color: var(--text) !important;
    }
    .risk-card h3 {
        margin: 0 0 2rem;
        color: var(--accent) !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .risk-card p {
        color: var(--text) !important;
        font-weight: 400;
        margin: 1.25rem 0 0;
    }
    .risk-card .risk-score + .assessment-label {
        margin-top: 1.75rem;
    }
    .risk-card .decision-badge + p {
        margin-top: 1.75rem;
    }
    .risk-card small {
        color: var(--muted) !important;
        display: block;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }
    .risk-score {
        color: var(--text) !important;
        font-size: 3.25rem;
        font-weight: 700;
        line-height: 1;
        margin-top: 0.5rem;
    }
    .decision-badge {
        display: inline-block;
        border: 1px solid;
        border-radius: 4px;
        color: var(--text) !important;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 0.25rem 0.55rem;
    }
    .decision-approve { border-color: var(--approve); color: var(--approve) !important; }
    .decision-review, .decision-flag { border-color: var(--danger); color: var(--danger) !important; }
    .assessment-label {
        color: var(--muted) !important;
        font-size: 0.85rem;
        font-weight: 400;
        margin: 0;
    }
    .detail-label {
        color: var(--muted) !important;
        font-size: 0.85rem;
        font-weight: 400;
    }
    .detail-label.emphasis,
    .detail-value.emphasis {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Return Risk Scorer")
st.caption(f"Calibrated model with cold-start protection | Threshold: {OPTIMAL_THRESHOLD:.2f}")
st.image("data/architecture_diagram.png", use_container_width=True)

with st.form("order_risk_form"):
    category = st.selectbox("Category", ["Apparel", "Beauty", "Electronics", "Footwear", "Home"])
    price = st.number_input("Price", min_value=0.0, value=1000.0, step=50.0)
    discount_pct = st.number_input(
        "Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0
    ) / 100
    customer_past_orders = st.number_input(
        "Customer past orders", min_value=0, value=0, step=1
    )
    customer_past_return_rate = st.number_input(
        "Customer past return rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0
    ) / 100
    payment_method = st.selectbox("Payment method", ["COD", "Prepaid"])
    delivery_days = st.number_input("Delivery days", min_value=1, value=5, step=1)
    days_to_purchase = st.number_input("Days to purchase", min_value=0, value=2, step=1)
    submitted = st.form_submit_button("Score order", type="primary")

if submitted:
    with st.spinner("Loading model and scoring order..."):
        calibrated_model, feature_columns = load_calibrated_pipeline()
        order = {
            "category": category,
            "is_size_sensitive": int(category in {"Apparel", "Footwear"}),
            "price": price,
            "discount_pct": discount_pct,
            "customer_past_orders": customer_past_orders,
            "customer_past_return_rate": customer_past_return_rate,
            "payment_method": payment_method,
            "region": "North",
            "delivery_days": delivery_days,
            "days_to_purchase": days_to_purchase,
        }
        result = evaluate_order_risk(order, calibrated_model, feature_columns)
        shap_features = get_shap_features(calibrated_model, order, feature_columns)
        narrative, narration_mode = explain_with_llm(shap_features)

    decision_map = {
        "PASS_LOW_RISK": "APPROVE",
        "MANUAL_REVIEW": "MANUAL_REVIEW",
        "FLAG_HIGH_RISK": "FLAG",
    }
    decision = decision_map[result["decision"]]
    decision_class = {
        "APPROVE": "decision-approve",
        "MANUAL_REVIEW": "decision-review",
        "FLAG": "decision-flag",
    }[decision]
    st.markdown(
        f"""
        <section class="risk-card">
            <h3>Risk assessment</h3>
            <p class="assessment-label">Risk score</p>
            <div class="risk-score">{result['model_prob']:.3f}</div>
            <p class="assessment-label">Decision</p>
            <div class="decision-badge {decision_class}">{escape(decision)}</div>
            <p><span class="detail-label">Calibrated probability:</span> <span class="detail-value emphasis">{result['model_prob']:.1%}</span></p>
            <p><span class="detail-label emphasis">Reason:</span> {escape(result['reason'])}</p>
            <p><span class="detail-label emphasis">Top-3 explanation:</span></p>
            <p>{escape(narrative)}</p>
            <small>Narration: {escape(narration_mode)}</small>
        </section>
        """,
        unsafe_allow_html=True,
    )