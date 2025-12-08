"""Cleaning utilities for transaction datasets.

This module provides `clean_transactions(input_csv, output_csv)` which:
- parses dates
- removes duplicates
- drops rows missing `customer_id` or `product_name`
- aggregates line items so each transaction row contains a semicolon-separated list of products and the total amount
- writes cleaned CSV with columns: transaction_id, customer_id, date, products, amount

The `__main__` block runs a small test using `data/raw/sample.csv` and writes to `data/cleaned/sample_cleaned.csv`.
"""
from typing import Optional
import os
import pandas as pd


def clean_transactions(input_csv: str, output_csv: str) -> pd.DataFrame:
    """Read transactions from `input_csv`, clean and aggregate, then write `output_csv`.

    Args:
        input_csv: Path to raw CSV file with columns [transaction_id, customer_id, date, product_id, product_name, amount].
        output_csv: Path where cleaned CSV will be written. Directory will be created if needed.

    Returns:
        Cleaned pandas DataFrame with columns [transaction_id, customer_id, date, products, amount].
    """
    df = pd.read_csv(input_csv)

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Drop rows missing customer_id or product_name
    df = df.dropna(subset=['customer_id', 'product_name'])

    # Ensure amount numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

    # Aggregate line items to transactions
    agg = df.groupby(['transaction_id', 'customer_id', 'date'], dropna=False).agg({
        'product_name': lambda x: ';'.join(x.astype(str)),
        'amount': 'sum'
    }).reset_index()

    # Rename product column to products
    agg = agg.rename(columns={'product_name': 'products'})

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write CSV with specified columns order
    agg[['transaction_id', 'customer_id', 'date', 'products', 'amount']].to_csv(output_csv, index=False)

    return agg


if __name__ == '__main__':
    # Quick unit-test style run
    sample_in = os.path.join('data', 'raw', 'sample.csv')
    sample_out = os.path.join('data', 'cleaned', 'sample_cleaned.csv')
    print(f"Running cleaning on '{sample_in}' -> '{sample_out}'")
    cleaned = clean_transactions(sample_in, sample_out)
    print('Cleaned rows:', len(cleaned))
    print(cleaned.head().to_string(index=False))
