"""RFM feature computation utilities.

Provides `compute_rfm(cleaned_csv, output_csv, reference_date=None)` which computes
Recency (days since last purchase), Frequency (number of purchases), Monetary (total spend)
per customer from a cleaned transaction CSV.

If `reference_date` is None, the function uses the maximum `date` value in the dataset.
RFM values are min-max normalized (recency inverted so higher is better) and combined
into an `rfm_score` using weights R=0.3, F=0.4, M=0.3.

The function writes the result to `output_csv`. When run as `__main__` a small example
reads `data/cleaned/sample_cleaned.csv` (if present) and writes `models/rfm_features.csv`.
"""
from typing import Optional
import os
import pandas as pd


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to range [0,1].

    If the series has constant values, returns a series of zeros.
    """
    min_v = series.min()
    max_v = series.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(0.0, index=series.index)
    return (series - min_v) / (max_v - min_v)


def compute_rfm(cleaned_csv: str, output_csv: str, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Compute RFM features for each customer from a cleaned transactions CSV.

    Args:
        cleaned_csv: Path to cleaned transactions CSV with columns ['transaction_id','customer_id','date','products','amount'].
        output_csv: Path where the RFM features CSV will be written.
        reference_date: Optional reference date (pd.Timestamp or string). If None, uses max(date) from data.

    Returns:
        DataFrame with columns ['customer_id','recency','frequency','monetary','rfm_score'] indexed by integer.
    """
    df = pd.read_csv(cleaned_csv, parse_dates=['date'], dayfirst=False)

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if reference_date is None:
        ref_date = df['date'].max()
    else:
        ref_date = pd.to_datetime(reference_date)

    if pd.isna(ref_date):
        raise ValueError('Reference date is NaT; ensure `date` column contains valid dates or pass `reference_date`')

    # Compute per-customer metrics
    grouped = df.groupby('customer_id').agg(
        last_date=('date', 'max'),
        frequency=('transaction_id', 'nunique'),
        monetary=('amount', 'sum')
    ).reset_index()

    # Recency in days: days since last purchase
    grouped['recency'] = (pd.to_datetime(ref_date) - pd.to_datetime(grouped['last_date'])).dt.days

    # If frequency is NaN (shouldn't be), fill 0
    grouped['frequency'] = grouped['frequency'].fillna(0).astype(int)
    grouped['monetary'] = grouped['monetary'].fillna(0.0)

    # Normalization: recency should be inverted (lower recency = better)
    recency_norm = 1.0 - _min_max_normalize(grouped['recency'])
    frequency_norm = _min_max_normalize(grouped['frequency'])
    monetary_norm = _min_max_normalize(grouped['monetary'])

    # Weighted RFM score
    R_WEIGHT = 0.3
    F_WEIGHT = 0.4
    M_WEIGHT = 0.3

    grouped['rfm_score'] = (
        recency_norm * R_WEIGHT +
        frequency_norm * F_WEIGHT +
        monetary_norm * M_WEIGHT
    )

    # Keep only requested columns
    result = grouped[['customer_id', 'recency', 'frequency', 'monetary', 'rfm_score']]

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    result.to_csv(output_csv, index=False)

    # Also save a copy to models/rfm_features.csv for convention
    try:
        models_path = os.path.join('models', 'rfm_features.csv')
        models_dir = os.path.dirname(models_path)
        if models_dir and not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        result.to_csv(models_path, index=False)
    except Exception:
        # Non-fatal: continue even if saving to models/ fails
        pass

    return result


if __name__ == '__main__':
    # Small example usage
    import sys

    cleaned = os.path.join('data', 'cleaned', 'sample_cleaned.csv')
    out = os.path.join('models', 'rfm_features.csv')

    if not os.path.exists(cleaned):
        print(f"Cleaned file '{cleaned}' not found. Run the cleaning script first: python notebooks\\cleaning.py")
        sys.exit(1)

    print(f"Computing RFM from '{cleaned}' -> '{out}'")
    df_rfm = compute_rfm(cleaned, out)
    print('Computed RFM for customers:', len(df_rfm))
    print(df_rfm.head().to_string(index=False))
