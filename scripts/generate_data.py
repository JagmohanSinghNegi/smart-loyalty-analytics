"""
Generate synthetic transaction data for the smart loyalty project.
Creates realistic customer purchase patterns with repeat customers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
num_transactions = 500
num_customers = 50
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

# Products list
products = [
    'Apple', 'Banana', 'Orange', 'Mango', 'Grape',
    'Milk', 'Cheese', 'Yogurt', 'Bread', 'Eggs',
    'Chicken', 'Beef', 'Fish', 'Rice', 'Pasta',
    'Tomato', 'Lettuce', 'Carrot', 'Potato', 'Onion',
    'Coffee', 'Tea', 'Juice', 'Water', 'Soda',
    'Soap', 'Shampoo', 'Toothpaste', 'Deodorant', 'Lotion'
]

def generate_date_range():
    """Generate random date between start and end."""
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    return start_date + timedelta(days=random_days)

def generate_products():
    """Generate 1-3 random products for a transaction."""
    num_products = random.randint(1, 3)
    selected_products = random.sample(products, num_products)
    return ';'.join(selected_products)

def generate_amount():
    """Generate random transaction amount."""
    return round(random.uniform(5.0, 100.0), 2)

# Generate transactions with repeat customers (to simulate loyalty)
transactions = []
transaction_id = 1

# Create customer profiles (some are loyal, some are one-time)
customer_loyalty_prob = {}
for cid in range(1001, 1001 + num_customers):
    # 40% loyal (high repeat rate), 30% occasional, 30% one-time
    rand = random.random()
    if rand < 0.4:
        customer_loyalty_prob[cid] = 0.7  # Loyal customers buy often
    elif rand < 0.7:
        customer_loyalty_prob[cid] = 0.3  # Occasional customers
    else:
        customer_loyalty_prob[cid] = 0.1  # One-time customers

# Generate transaction distribution
customer_transactions = {}
remaining = num_transactions

for cid in range(1001, 1001 + num_customers):
    if remaining <= 0:
        break
    
    # Allocate transactions proportionally to loyalty probability
    loyalty_prob = customer_loyalty_prob[cid]
    # Loyal customers get 3-8 transactions, occasional get 2-4, one-time get 1
    if loyalty_prob > 0.5:
        num_trans = random.randint(3, 8)
    elif loyalty_prob > 0.2:
        num_trans = random.randint(2, 4)
    else:
        num_trans = random.randint(1, 2)
    
    num_trans = min(num_trans, remaining)
    customer_transactions[cid] = num_trans
    remaining -= num_trans

# Generate actual transactions
for cid, num_trans in customer_transactions.items():
    for _ in range(num_trans):
        transactions.append({
            'transaction_id': transaction_id,
            'customer_id': cid,
            'date': generate_date_range(),
            'product_id': random.randint(1000, 1029),
            'product_name': random.choice(products),
            'amount': generate_amount()
        })
        transaction_id += 1

# Create DataFrame and sort by date
df = pd.DataFrame(transactions)
df = df.sort_values('date').reset_index(drop=True)

# Save to CSV
output_path = 'data/raw/sample.csv'
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} synthetic transactions for {len(customer_transactions)} customers")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Saved to: {output_path}")
print(f"\nSample transactions:")
print(df.head(10))

# Summary statistics
print(f"\nCustomer Statistics:")
customer_counts = df['customer_id'].value_counts()
print(f"Avg transactions per customer: {customer_counts.mean():.1f}")
print(f"Min transactions per customer: {customer_counts.min()}")
print(f"Max transactions per customer: {customer_counts.max()}")
print(f"Total revenue: ${df['amount'].sum():.2f}")
print(f"Avg transaction value: ${df['amount'].mean():.2f}")
