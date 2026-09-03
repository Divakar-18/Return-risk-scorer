import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_synthetic_data(seed: int, num_samples: int = 4000) -> pd.DataFrame:
    """
    Generates synthetic e-commerce order data with realistic return drivers.
    
    Args:
        seed: Random seed for reproducibility.
        num_samples: Number of orders to generate.
        
    Returns:
        DataFrame containing the synthetic orders and return targets.
    """
    np.random.seed(seed)
    
    # Base configuration
    categories = ['Apparel', 'Electronics', 'Home', 'Beauty', 'Footwear']
    size_sensitive_cats = ['Apparel', 'Footwear']
    payment_methods = ['COD', 'Prepaid']
    regions = ['North', 'South', 'East', 'West']
    
    # 1. Basic Order Info
    order_id = np.arange(1, num_samples + 1)
    
    # Dates spanning 6 months
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    date_offsets = np.random.randint(0, 180, num_samples)
    order_date = [start_date + timedelta(days=int(d)) for d in date_offsets]
    
    # 2. Product Features
    category = np.random.choice(categories, num_samples, p=[0.4, 0.2, 0.15, 0.1, 0.15])
    is_size_sensitive = np.isin(category, size_sensitive_cats).astype(int)
    
    # Prices: log-normal distribution to simulate real e-commerce prices
    price = np.round(np.random.lognormal(mean=7, sigma=1, size=num_samples), 2)
    # Discount percentage: most are 0, some up to 80%
    discount_pct = np.clip(np.random.normal(loc=0.1, scale=0.2, size=num_samples), 0, 0.8)
    # Zero out some discounts completely
    discount_pct[np.random.rand(num_samples) > 0.6] = 0.0
    
    # 3. Customer Features
    # Simulate some repeat customers by picking from a smaller pool
    customer_id = np.random.randint(1000, 1000 + int(num_samples * 0.7), num_samples)
    
    # We need realistic past orders and return rates.
    # To keep the script self-contained per row for simplicity, we simulate past stats directly.
    # A Poisson distribution for past orders.
    customer_past_orders = np.random.poisson(lam=2, size=num_samples)
    
    # Past return rate: typically between 0 and 1. If past_orders is 0, return rate is 0.
    customer_past_return_rate = np.random.beta(a=1, b=4, size=num_samples)
    customer_past_return_rate[customer_past_orders == 0] = 0.0
    
    # 4. Transaction Features
    payment_method = np.random.choice(payment_methods, num_samples, p=[0.3, 0.7])
    region = np.random.choice(regions, num_samples)
    
    # Delivery days: 1 to 14 days
    delivery_days = np.random.randint(1, 15, num_samples)
    
    # Days to purchase: proxy for impulse buying (0 = same day, 30 = long deliberation)
    days_to_purchase = np.random.exponential(scale=3, size=num_samples).astype(int)
    
    # 5. Calculate Return Probability (Log-odds formulation)
    # Base log-odds (intercept) for ~5% base return rate
    log_odds = np.full(num_samples, -2.9)
    
    # Rule A: Size-sensitive categories increase return odds
    log_odds += is_size_sensitive * 0.8
    
    # Rule B: High discount + short days_to_purchase (Impulse buy)
    impulse_buy = (discount_pct > 0.3) & (days_to_purchase <= 1)
    log_odds += impulse_buy.astype(int) * 1.2
    
    # Rule C: COD + first-time customer (low commitment)
    low_commitment = (payment_method == 'COD') & (customer_past_orders == 0)
    log_odds += low_commitment.astype(int) * 1.5
    
    # Rule D: Customer's past return behavior is highly predictive
    # For repeat customers, heavily weight their past return rate
    is_repeat = (customer_past_orders > 0)
    log_odds += is_repeat * (customer_past_return_rate * 3.5 - 0.5) 
    
    # Rule E: Long delivery days slightly increase returns (frustration/found elsewhere)
    log_odds += (delivery_days > 7).astype(int) * 0.4
    
    # Add realistic noise
    noise = np.random.normal(0, 0.5, num_samples)
    log_odds += noise
    
    # Convert log-odds to probability
    prob_return = 1 / (1 + np.exp(-log_odds))
    
    # Sample actual return outcome
    returned = np.random.binomial(1, prob_return)
    
    # Create DataFrame
    df = pd.DataFrame({
        'order_id': order_id,
        'order_date': order_date,
        'category': category,
        'is_size_sensitive': is_size_sensitive,
        'price': price,
        'discount_pct': discount_pct,
        'customer_id': customer_id,
        'customer_past_orders': customer_past_orders,
        'customer_past_return_rate': customer_past_return_rate,
        'payment_method': payment_method,
        'region': region,
        'delivery_days': delivery_days,
        'days_to_purchase': days_to_purchase,
        'returned': returned
    })
    
    # Sort chronologically to make time-based splitting easier later
    df = df.sort_values('order_date').reset_index(drop=True)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce orders for return risk modeling.")
    parser.add_argument('--output_dir', type=str, default='data', help="Directory to save the generated CSV files.")
    parser.add_argument('--num_samples', type=int, default=5000, help="Number of samples to generate per seed.")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Handle missing output directory gracefully as per requirements
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' does not exist. Creating it now.")
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"ERROR: Failed to create output directory '{output_dir}'.")
            print(f"Exception: {e}")
            raise SystemExit(1)
            
    # Seeds: 42 (primary), 1042 and 2042 (for Phase 9 stability testing)
    seeds = [42, 1042, 2042]
    
    for seed in seeds:
        print(f"Generating data for seed {seed}...")
        df = generate_synthetic_data(seed=seed, num_samples=args.num_samples)
        
        return_rate = df['returned'].mean() * 100
        print(f"  Generated {len(df)} rows. Target return rate: {return_rate:.2f}%")
        
        # We need a primary file and auxiliary files
        filename = "synthetic_orders.csv" if seed == 42 else f"synthetic_orders_seed_{seed}.csv"
        filepath = os.path.join(output_dir, filename)
        
        try:
            df.to_csv(filepath, index=False)
            print(f"  Saved to {filepath}")
        except OSError as e:
            print(f"ERROR: Failed to save data to '{filepath}'.")
            print(f"Exception: {e}")
            raise SystemExit(1)
            
    print("\nData generation complete.")

if __name__ == "__main__":
    main()
