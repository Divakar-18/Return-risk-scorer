# Synthetic E-Commerce Return Data

This directory contains synthetically generated e-commerce orders designed to train and evaluate the Return-Risk Scorer. 

## Label Generation Rules

The `returned` target variable (0 = Kept, 1 = Returned) is generated using a log-odds probabilistic formulation with the following realistic e-commerce drivers:

1. **Size-Sensitive Categories:**
   - Categories like `Apparel` and `Footwear` inherently carry higher return risk due to fit issues.
   - **Effect:** Increases log-odds of a return by +0.8.

2. **Impulse Buying:**
   - Defined as high discount (`discount_pct > 0.3`) paired with a short deliberation time (`days_to_purchase <= 1`).
   - Customers buying heavily discounted items on impulse often experience buyer's remorse.
   - **Effect:** Increases log-odds of a return by +1.2.

3. **Low Commitment (Cold Start + COD):**
   - Customers with no prior history (`customer_past_orders == 0`) choosing Cash on Delivery (`payment_method == 'COD'`).
   - These orders have zero sunk cost for the customer and high refusal-at-doorstep rates.
   - **Effect:** Increases log-odds of a return by +1.5.

4. **Historical Return Behavior:**
   - For repeat customers (`customer_past_orders > 0`), their past return rate (`customer_past_return_rate`) is the strongest predictor.
   - **Effect:** Scales directly with past return rate (adds up to ~+3.0 log-odds for serial returners).

5. **Fulfillment Friction:**
   - Orders taking more than 7 days to deliver (`delivery_days > 7`).
   - Slower deliveries increase the chance the customer found an alternative locally or lost interest.
   - **Effect:** Increases log-odds of a return by +0.4.

## Data Characteristics
- **Volume:** 5,000 samples per seed.
- **Noise:** Realistic normally distributed noise (`N(0, 0.5)`) is added to the log-odds before logistic conversion to ensure the classes are not perfectly separable.
- **Target Distribution:** Designed to yield a highly realistic overall return rate of **12% – 18%**.

## Files
- `synthetic_orders.csv`: Primary dataset generated with seed 42.
- `synthetic_orders_seed_1042.csv`: Auxiliary dataset for stability testing.
- `synthetic_orders_seed_2042.csv`: Auxiliary dataset for stability testing.
