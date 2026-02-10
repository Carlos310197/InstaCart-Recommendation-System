from pathlib import Path
import os
from flask import Blueprint, request, jsonify
import pandas as pd
import xgboost as xgb

from utils import FEATURE_COLS, load_feature_tables, build_scoring_frame, prepare_features, select_topk

predict_bp = Blueprint('predict', __name__)

DATA_DIR = Path(os.getenv('DATA_DIR', Path(__file__).resolve().parents[3] / 'data'))
FEATURE_DIR = Path(os.getenv('FEATURE_DIR', DATA_DIR / 'features'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', Path(__file__).resolve().parents[3] / 'opt' / 'xgb_recommender.json'))

tables = load_feature_tables(FEATURE_DIR)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)


def _parse_orders(payload: dict) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    if 'orders' in payload:
        return pd.DataFrame(payload['orders'])
    return pd.DataFrame([payload])


@predict_bp.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    k = int(payload.get('k', 10))
    order_rows = _parse_orders(payload)
    if order_rows.empty:
        return jsonify({'error': 'orders payload is required'}), 400

    required_cols = {'order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'}
    missing = required_cols.difference(order_rows.columns)
    if missing:
        return jsonify({'error': f'missing required fields: {sorted(missing)}'}), 400

    df = build_scoring_frame(order_rows, tables)
    if df.empty:
        return jsonify({'recommendations': []})

    df = prepare_features(df)
    preds = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df['pred'] = preds

    topk = select_topk(df, k)
    return jsonify({'recommendations': topk.to_dict(orient='records')})