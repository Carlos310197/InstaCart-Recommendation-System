from pathlib import Path
import os
import numpy as np
import pandas as pd


def compute_streak(order_numbers):
    order_numbers = np.sort(order_numbers)
    if order_numbers.size == 0:
        return 0
    streak = 1
    for i in range(order_numbers.size - 2, -1, -1):
        if order_numbers[i] == order_numbers[i + 1] - 1:
            streak += 1
        else:
            break
    return streak


def build_feature_tables_from_raw(data_dir: Path) -> dict:
    orders = pd.read_csv(data_dir / 'orders.csv')
    prior = pd.read_csv(data_dir / 'order_products__prior.csv')
    products = pd.read_csv(data_dir / 'products.csv')

    prior = prior.merge(orders[['order_id', 'user_id', 'order_number']], on='order_id', how='left')
    prior['order_number'] = prior['order_number'].astype(np.int16)

    user_features = orders.groupby('user_id').agg(
        user_total_orders=('order_number', 'max'),
        user_avg_days_since_prior=('days_since_prior_order', 'mean'),
    ).reset_index()
    user_reorder = prior.groupby('user_id')['reordered'].mean().reset_index(name='user_reorder_ratio')
    user_features = user_features.merge(user_reorder, on='user_id', how='left')
    user_features['user_avg_days_since_prior'] = user_features['user_avg_days_since_prior'].fillna(0.0).astype(np.float32)
    user_features['user_reorder_ratio'] = user_features['user_reorder_ratio'].fillna(0.0).astype(np.float32)

    prod_features = prior.groupby('product_id').agg(
        prod_total_purchases=('order_id', 'count'),
        prod_reorder_prob=('reordered', 'mean'),
    ).reset_index()
    prod_features['prod_total_purchases'] = prod_features['prod_total_purchases'].astype(np.int32)
    prod_features['prod_reorder_prob'] = prod_features['prod_reorder_prob'].astype(np.float32)
    prod_features = prod_features.merge(products[['product_id', 'aisle_id', 'department_id']], on='product_id', how='left')

    up = prior.groupby(['user_id', 'product_id']).agg(
        up_total_orders=('order_id', 'count'),
        up_avg_position=('add_to_cart_order', 'mean'),
        up_last_order_number=('order_number', 'max'),
    ).reset_index()
    up['up_avg_position'] = up['up_avg_position'].astype(np.float32)
    up = up.merge(user_features[['user_id', 'user_total_orders']], on='user_id', how='left')
    up['up_order_rate'] = up['up_total_orders'] / up['user_total_orders']
    up['up_orders_since_last'] = up['user_total_orders'] - up['up_last_order_number']
    up['up_orders_since_last'] = up['up_orders_since_last'].astype(np.int16)
    up['up_order_rate'] = up['up_order_rate'].astype(np.float32)
    up_streak = prior.groupby(['user_id', 'product_id'])['order_number'].apply(compute_streak).reset_index(name='up_streak')
    up = up.merge(up_streak, on=['user_id', 'product_id'], how='left')
    up['up_streak'] = up['up_streak'].fillna(0).astype(np.int16)
    up = up.drop(columns=['user_total_orders'])

    user_prior = prior[['user_id', 'product_id']].drop_duplicates()

    return {
        'user_features': user_features,
        'prod_features': prod_features,
        'up': up,
        'user_prior': user_prior,
    }


def save_feature_tables(tables: dict, feature_dir: Path) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    tables['user_features'].to_parquet(feature_dir / 'user_features.parquet', index=False)
    tables['prod_features'].to_parquet(feature_dir / 'prod_features.parquet', index=False)
    tables['up'].to_parquet(feature_dir / 'up.parquet', index=False)
    tables['user_prior'].to_parquet(feature_dir / 'user_prior.parquet', index=False)


def main() -> None:
    data_dir = Path(os.getenv('DATA_DIR', Path(__file__).resolve().parents[1] / 'data'))
    feature_dir = Path(os.getenv('FEATURE_DIR', data_dir / 'features'))

    tables = build_feature_tables_from_raw(data_dir)
    save_feature_tables(tables, feature_dir)
    print(f"Saved feature tables to {feature_dir}")


if __name__ == '__main__':
    main()
