"""Simple basket co-occurrence recommender.

Function:
- recommend_for_product(product_name, top_n=5)

Implementation uses cleaned transactions in `data/cleaned/*.csv`.
Handles both formats:
1. Single product per row: uses customer purchase history
2. Multiple products per row: uses within-transaction co-occurrence
"""
from typing import List
import os
import pandas as pd
from collections import Counter


def _load_cleaned(path: str = os.path.join('data', 'cleaned', 'sample_cleaned.csv')) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def recommend_for_product(product_name: str, top_n: int = 5) -> List[str]:
    """Recommend products that co-occur with `product_name`.
    
    Uses two strategies:
    1. If products column has semicolon-separated values: co-occurrence within transactions
    2. Otherwise: products bought by same customers who bought product_name
    
    Returns a list of product names sorted by co-occurrence count (descending).
    """
    df = _load_cleaned()
    if df.empty:
        return []

    # Ensure product_name is case-insensitive
    product_name_lower = product_name.lower()
    
    # Strategy 1: Check if products column has multiple products (semicolon-separated)
    if ';' in df['products'].astype(str).str.cat():
        # Split products into lists
        df['products_list'] = df['products'].astype(str).str.split(';')
        
        # Find transactions containing the product
        mask = df['products_list'].apply(
            lambda lst: product_name_lower in [p.strip().lower() for p in lst]
        )
        tx_with = df[mask]
        
        if not tx_with.empty:
            # Collect other products appearing in these transactions
            co_products = Counter()
            for lst in tx_with['products_list']:
                for p in lst:
                    p_clean = p.strip()
                    if p_clean.lower() != product_name_lower:
                        co_products[p_clean] += 1
            
            # Return top_n most common
            recommendations = [prod for prod, _ in co_products.most_common(top_n)]
            return recommendations
    
    # Strategy 2: Customer-based co-occurrence (default for single product per row)
    # Find customers who bought the target product
    target_mask = df['products'].astype(str).str.lower() == product_name_lower
    target_customers = df[target_mask]['customer_id'].unique()
    
    if len(target_customers) == 0:
        return []
    
    # Find other products these customers bought
    other_purchases = df[df['customer_id'].isin(target_customers) & ~target_mask]
    
    if other_purchases.empty:
        return []
    
    # Count occurrences of each product
    product_counts = other_purchases['products'].astype(str).str.lower().value_counts()
    
    # Return top_n products
    recommendations = [prod for prod in product_counts.head(top_n).index]
    return recommendations
