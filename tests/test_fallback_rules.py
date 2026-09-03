import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from predict import evaluate_order_risk


class MockCalibratedModel:
    """Mock model returning a fixed low probability to test rule override behavior."""
    def predict_proba(self, X):
        # Always return low risk probability (0.05) as numpy array
        return np.array([[0.95, 0.05]])

def test_cold_start_fallback_rule_override():
    """Verifies that cold-start COD orders are forced to MANUAL_REVIEW regardless of low model score."""
    mock_model = MockCalibratedModel()
    feature_cols = [
        'is_size_sensitive', 'price', 'discount_pct', 'customer_past_orders',
        'customer_past_return_rate', 'delivery_days', 'days_to_purchase',
        'category_Beauty', 'category_Electronics', 'category_Footwear', 'category_Home',
        'payment_method_Prepaid', 'region_North', 'region_South', 'region_West'
    ]
    
    # Cold-start COD order (customer_past_orders == 0)
    cold_start_cod_order = {
        'order_id': 'TEST_COLD_001',
        'category': 'Electronics',
        'price': 100.0,
        'customer_past_orders': 0,
        'customer_past_return_rate': 0.0,
        'payment_method': 'COD'
    }
    
    result = evaluate_order_risk(cold_start_cod_order, mock_model, feature_cols)
    
    assert result['decision'] == "MANUAL_REVIEW", f"Expected MANUAL_REVIEW, got {result['decision']}"
    assert result['model_prob'] == 0.05, f"Expected model probability 0.05, got {result['model_prob']}"
    assert "Cold-start fallback rule triggered" in result['reason']

def test_repeat_customer_low_risk_passes():
    """Verifies that a repeat customer with low risk probability passes normally."""
    mock_model = MockCalibratedModel()
    feature_cols = [
        'is_size_sensitive', 'price', 'discount_pct', 'customer_past_orders',
        'customer_past_return_rate', 'delivery_days', 'days_to_purchase',
        'category_Beauty', 'category_Electronics', 'category_Footwear', 'category_Home',
        'payment_method_Prepaid', 'region_North', 'region_South', 'region_West'
    ]
    
    repeat_customer_order = {
        'order_id': 'TEST_REPEAT_002',
        'category': 'Electronics',
        'price': 100.0,
        'customer_past_orders': 5,
        'customer_past_return_rate': 0.0,
        'payment_method': 'Prepaid'
    }
    
    result = evaluate_order_risk(repeat_customer_order, mock_model, feature_cols)
    
    assert result['decision'] == "PASS_LOW_RISK", f"Expected PASS_LOW_RISK, got {result['decision']}"
