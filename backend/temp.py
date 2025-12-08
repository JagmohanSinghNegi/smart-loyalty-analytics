import pandas as pd
import os

# Ensure folders exist
os.makedirs("../models", exist_ok=True)

# Load cleaned transaction data
data_path = "../data/cleaned/sample_cleaned.csv"
df = pd.read_csv(data_path)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Calculate RFM
reference_date = df['date'].max()

rfm = df.groupby('customer_id').agg({
    'date': lambda x: (reference_date - x.max()).days,  # Recency
    'order_id': 'count',                                # Frequency
    'amount': 'sum'                                     # Monetary value
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Save RFM features
output_path = "../models/rfm_features.csv"
rfm.to_csv(output_path, index=True)

print(f"✅ RFM features saved successfully at {output_path}")
