# Instacart Recommendation System

## Video of the demo

[![Demo video (GIF preview)](assets/demo_video.gif)](assets/demo_video.mov)

## Project structure

- **Challenge prompt**: [Challenge.md](Challenge.md)
- **Python deps**: [requirements.txt](requirements.txt)
- **Data files**: [data/](data/)
- **Feature build script**: [preprocessing/build_features.py](preprocessing/build_features.py)
- **Model artifact**: [opt/xgb_recommender.json](opt/xgb_recommender.json)

### Backend (Flask + Gunicorn)
- App entry: [backend/src/app.py](backend/src/app.py)
- Routes:
  - Health: [backend/src/routes/health.py](backend/src/routes/health.py)
  - Predictions: [backend/src/routes/predict.py](backend/src/routes/predict.py)
- Utilities: [backend/src/utils.py](backend/src/utils.py)

### Frontend (Streamlit)
- App UI: [frontend/streamlit_app.py](frontend/streamlit_app.py)
  - Uses [`frontend.streamlit_app.get_last_products`](frontend/streamlit_app.py) to show recent items.

### Notebooks
- **EDA**: [notebooks/EDA.ipynb](notebooks/EDA.ipynb)  
  - Data profiling and visual exploration of orders/products.
- **Baseline recommender**: [notebooks/Baseline_Recommender.ipynb](notebooks/Baseline_Recommender.ipynb)  
  - Builds a simple next-basket baseline and explores top products + user-product lists.
- **Two-stage model**: [notebooks/TwoStage_Recommender.ipynb](notebooks/TwoStage_Recommender.ipynb)  
  - Candidate generation + feature engineering + XGBoost model.
  - This notebook saves the model into the opt folder.
  - If needed, you can run this notebook for this purpose.

---

## Environment setup

Python should be already installed in your current OS. Then follow up these steps for running the solution

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### 2) Install dependencies (uv)
```bash
pip install -r requirements.txt
```

### 3) Ensure data files exist
Data is expected under [data/](data/), including:
- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`

Download the data from Kaggle: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

---

## Build features (optional, for model training)

Features are used in the XGBoost model. Build them with:

```bash
python preprocessing/build_features.py
```

This writes feature tables into [data/features/](data/features/) using [preprocessing/build_features.py](preprocessing/build_features.py).

---

## Run the backend

In one terminal, from the project root, run:

```bash
python backend/src/app.py
```

If you prefer Gunicorn:

```bash
gunicorn -c backend/gunicorn.conf.py backend.src.app:app
```

- Health endpoint: `/health` (see [backend/src/routes/health.py](backend/src/routes/health.py))
- Predict endpoint: `/predict` (see [backend/src/routes/predict.py](backend/src/routes/predict.py))

---

## Run the Streamlit app

In one terminal, from the project root, run:

```bash
streamlit run frontend/streamlit_app.py
```

The UI:
- Loads `orders.csv`, `order_products__train.csv`, and `products.csv`.
- Shows “Last products bought” using [`frontend.streamlit_app.get_last_products`](frontend/streamlit_app.py).
- Calls the backend `/predict` endpoint to display “Top recommendations”.

