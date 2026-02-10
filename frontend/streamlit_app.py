import json
from pathlib import Path
from urllib import request, error

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Instacart Recommender", layout="centered")

st.title("Instacart Recommender")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@st.cache_data
def load_orders() -> pd.DataFrame:
    path = DATA_DIR / "orders.csv"
    orders = pd.read_csv(
        path,
        usecols=["order_id", "user_id", "order_number", "eval_set"],
        dtype={
            "order_id": "int64",
            "user_id": "int64",
            "order_number": "int16",
            "eval_set": "category",
        },
    )
    return orders[orders["eval_set"].isin(["train", "test"])].copy()


@st.cache_data
def load_order_products_train() -> pd.DataFrame:
    path = DATA_DIR / "order_products__train.csv"
    return pd.read_csv(
        path,
        usecols=["order_id", "product_id", "add_to_cart_order"],
        dtype={"order_id": "int64", "product_id": "int64", "add_to_cart_order": "int16"},
    )


@st.cache_data
def load_products() -> pd.DataFrame:
    path = DATA_DIR / "products.csv"
    return pd.read_csv(
        path,
        usecols=["product_id", "product_name"],
        dtype={"product_id": "int64", "product_name": "string"},
    )


def get_last_products(
    user_id: int,
    orders_df: pd.DataFrame,
    order_products_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    user_orders = orders_df[orders_df["user_id"] == user_id]
    if user_orders.empty:
        return pd.DataFrame(columns=["product_id", "product_name"])

    merged = user_orders.merge(order_products_df, on="order_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["product_id", "product_name"])

    merged = merged.merge(products_df, on="product_id", how="left")
    merged = merged.sort_values(["order_number", "add_to_cart_order"], ascending=[False, True])
    merged = merged.drop_duplicates(subset=["product_id"], keep="first")
    return merged[["product_id", "product_name"]]


try:
    orders_df = load_orders()
    order_products_df = load_order_products_train()
    products_df = load_products()
except Exception as exc:
    st.error(f"Failed to load data files: {exc}")
    st.stop()

product_lookup = products_df.set_index("product_id")["product_name"].to_dict()
next_order_id = int(orders_df["order_id"].max()) + 1 if not orders_df.empty else 1

with st.sidebar:
    st.subheader("Backend")
    backend_url = st.text_input("Base URL", value="http://localhost:8000")
    if st.button("Check health"):
        try:
            with request.urlopen(f"{backend_url}/health", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            st.success(f"Status: {payload.get('status', 'unknown')}")
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

st.subheader("Order snapshot")

with st.form("predict"):
    st.caption(f"Assigned order ID: {next_order_id}")

    col1, col2 = st.columns(2)
    user_id = col1.number_input("User ID", min_value=1, step=1)
    days_since = col2.number_input("Days since prior order", min_value=0, step=1)

    dow_labels = {
        0: "Saturday",
        1: "Sunday",
        2: "Monday",
        3: "Tuesday",
        4: "Wednesday",
        5: "Thursday",
        6: "Friday",
    }
    order_dow = st.radio(
        "Order day of week",
        options=list(dow_labels.keys()),
        format_func=lambda value: f"{value}: {dow_labels[value]}",
        horizontal=True,
    )

    order_hour = st.radio(
        "Order hour (0-23)",
        options=list(range(24)),
        horizontal=True,
    )

    k = st.radio(
        "Top K recommendations",
        options=[5, 10, 15],
        horizontal=True,
    )

    submitted = st.form_submit_button("Get recommendations")

if submitted:
    last_products = get_last_products(int(user_id), orders_df, order_products_df, products_df)
    last_products_count = len(last_products.index)
    if last_products_count == 0:
        st.info("No products found for that user in train/test orders.")

    payload = {
        "k": int(k),
        "order_id": next_order_id,
        "user_id": int(user_id),
        "order_dow": int(order_dow),
        "order_hour_of_day": int(order_hour),
        "days_since_prior_order": int(days_since),
    }

    try:
        req = request.Request(
            f"{backend_url}/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        recs = data.get("recommendations", [])

        recs_df = pd.DataFrame(recs)
        if not recs_df.empty:
            recs_df["product_name"] = recs_df["product_id"].map(product_lookup)
            recs_df = recs_df[["product_id", "product_name"]]

        st.subheader("Last products bought")
        if last_products_count == 0:
            st.write("No matching products found.")
        else:
            st.caption(f"Showing {last_products_count} unique products.")
            st.dataframe(last_products, use_container_width=True)

        st.subheader("Top recommendations")
        if recs_df.empty:
            st.write("No recommendations returned.")
        else:
            st.dataframe(recs_df, use_container_width=True)

        match_count = 0
        if not recs_df.empty and last_products_count > 0:
            rec_ids = set(recs_df["product_id"].dropna().astype(int).tolist())
            last_ids = set(last_products["product_id"].dropna().astype(int).tolist())
            match_count = len(rec_ids.intersection(last_ids))

        st.info(f"{match_count} of {len(recs_df)} products were bought in the last order!")
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            message = detail.get("error", str(exc))
        except Exception:
            message = str(exc)
        st.error(f"Prediction failed: {message}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
