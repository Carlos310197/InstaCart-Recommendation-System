from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLS = [
    'up_total_orders', 'up_order_rate', 'up_avg_position', 'up_orders_since_last', 'up_streak',
    'user_total_orders', 'user_avg_days_since_prior', 'user_reorder_ratio',
    'prod_total_purchases', 'prod_reorder_prob', 'aisle_id', 'department_id',
    'order_dow', 'order_hour_of_day', 'days_since_prior_order',
]


def _feature_paths(feature_dir: Path) -> dict:
    return {
        'user_features': feature_dir / 'user_features.parquet',
        'prod_features': feature_dir / 'prod_features.parquet',
        'up': feature_dir / 'up.parquet',
        'user_prior': feature_dir / 'user_prior.parquet',
    }


def load_feature_tables(feature_dir: Path) -> dict:
    paths = _feature_paths(feature_dir)
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            'Missing feature tables. Run scripts/build_features.py to generate them. Missing: '
            + ', '.join(missing)
        )
    return {
        'user_features': pd.read_parquet(paths['user_features']),
        'prod_features': pd.read_parquet(paths['prod_features']),
        'up': pd.read_parquet(paths['up']),
        'user_prior': pd.read_parquet(paths['user_prior']),
    }


def build_scoring_frame(order_rows: pd.DataFrame, tables):
    candidates = order_rows[['order_id', 'user_id']].merge(tables['user_prior'], on='user_id', how='left')
    if candidates.empty:
        return candidates

    df = candidates.merge(tables['up'], on=['user_id', 'product_id'], how='left')
    df = df.merge(tables['user_features'], on='user_id', how='left')
    df = df.merge(tables['prod_features'], on='product_id', how='left')
    df = df.merge(order_rows, on=['order_id', 'user_id'], how='left')
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0.0).astype(np.float32)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).astype(np.float32)
    return df


def select_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return (
        df.sort_values(['order_id', 'pred'], ascending=[True, False])
          .groupby('order_id')
          .head(k)[['order_id', 'product_id', 'pred']]
    )
