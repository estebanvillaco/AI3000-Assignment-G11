#!/usr/bin/env python3
"""
ai_stock_recommender.py

Single-file AI-powered stock recommender (no TensorFlow).
- Auto-detects columns in your CSV
- Trains RandomForest to predict daily Units Sold
- Iteratively forecasts next N days per Store+Product
- Computes safety stock, reorder point, and recommended weekly reorder
- Produces plots and CSV outputs

Requirements:
  pip install pandas numpy scikit-learn matplotlib

Usage:
  python ai_stock_recommender.py --data /path/to/retail_store_inventory.csv
"""

import argparse
from cmath import rect
import math
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Utilities: Column autodetection
# ---------------------------
DATE_KEYWORDS = ["date", "day"]
PRODUCT_KEYWORDS = ["product", "sku", "item", "article"]
STORE_KEYWORDS = ["store", "shop", "outlet"]
SALES_KEYWORDS = ["units sold", "units_sold", "units", "sales", "qty", "quantity"]
INVENTORY_KEYWORDS = ["inventory", "stock", "inventory level", "stock_level"]
PRICE_KEYWORDS = ["price", "unit_price", "selling_price"]
SEASON_KEYWORDS = ["season", "seasonality"]
HOLIDAY_KEYWORDS = ["holiday", "promotion", "promo"]


def find_column(columns, keywords):
    lower = [c.lower() for c in columns]
    for kw in keywords:
        # allow spaces, underscores equivalently
        for i, name in enumerate(lower):
            if kw in name:
                return columns[i]
    # fallback: try partial matches
    for i, name in enumerate(lower):
        for kw in keywords:
            if kw.replace("_", "") in name.replace("_", ""):
                return columns[i]
    return None


def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    detected = {}
    detected['date'] = find_column(cols, DATE_KEYWORDS)
    detected['product'] = find_column(cols, PRODUCT_KEYWORDS)
    detected['store'] = find_column(cols, STORE_KEYWORDS)
    detected['sales'] = find_column(cols, SALES_KEYWORDS)
    detected['inventory'] = find_column(cols, INVENTORY_KEYWORDS)
    detected['price'] = find_column(cols, PRICE_KEYWORDS)
    detected['season'] = find_column(cols, SEASON_KEYWORDS)
    detected['holiday'] = find_column(cols, HOLIDAY_KEYWORDS)
    return detected


# ---------------------------
# Preprocessing & feature engineering
# ---------------------------
def preprocess_load(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda c: c.strip())
    detected = detect_columns(df)
    # Basic logging
    print("Detected columns:")
    for k, v in detected.items():
        print(f"  {k:10s}: {v}")
    # Validate required fields
    if detected['date'] is None:
        raise ValueError("Could not detect a date column. Ensure your CSV has a date-like column.")
    if detected['product'] is None:
        raise ValueError("Could not detect a product column.")
    if detected['sales'] is None:
        raise ValueError("Could not detect a sales/units column.")

    # parse date
    df[detected['date']] = pd.to_datetime(df[detected['date']], errors='coerce')
    # drop rows without date or sales
    df = df.dropna(subset=[detected['date'], detected['sales']])
    # Ensure sales numeric
    df[detected['sales']] = pd.to_numeric(df[detected['sales']], errors='coerce').fillna(0).astype(float)

    # If store missing, create a default store
    if detected['store'] is None:
        df['__STORE__'] = 'SINGLE_STORE'
        detected['store'] = '__STORE__'

    # Inventory optional
    if detected['inventory'] is None:
        df['__INV__'] = 0
        detected['inventory'] = '__INV__'
    else:
        df[detected['inventory']] = pd.to_numeric(df[detected['inventory']], errors='coerce').fillna(0)

    # Price optional
    if detected['price'] is None:
        df['__PRICE__'] = df[detected['sales']].median()  # fallback
        detected['price'] = '__PRICE__'
    else:
        df[detected['price']] = pd.to_numeric(df[detected['price']], errors='coerce').fillna(df[detected['price']].median())

    # Season optional
    if detected['season'] is None:
        # derive season from month
        df['month'] = df[detected['date']].dt.month
        def month_to_season(m):
            if m in [12, 1, 2]: return 'Winter'
            if m in [3, 4, 5]: return 'Spring'
            if m in [6, 7, 8]: return 'Summer'
            return 'Autumn'
        df['__SEASON__'] = df['month'].apply(month_to_season)
        detected['season'] = '__SEASON__'
    else:
        df[detected['season']] = df[detected['season']].fillna('Unknown').astype(str)

    # Holiday/promo optional
    if detected['holiday'] is None:
        df['__HOL__'] = 0
        detected['holiday'] = '__HOL__'
    else:
        df[detected['holiday']] = pd.to_numeric(df[detected['holiday']], errors='coerce').fillna(0)

    # normalize column names in df for easier reference
    df = df.rename(columns={
        detected['date']: 'Date',
        detected['product']: 'Product',
        detected['store']: 'Store',
        detected['sales']: 'Units_Sold',
        detected['inventory']: 'Inventory_Level',
        detected['price']: 'Price',
        detected['season']: 'Seasonality',
        detected['holiday']: 'Holiday_Promo'
    })

    # ensure expected columns exist
    for c in ['Date', 'Product', 'Store', 'Units_Sold', 'Inventory_Level', 'Price', 'Seasonality', 'Holiday_Promo']:
        if c not in df.columns:
            df[c] = 0

    # add weekday, month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def create_time_features(df):
    # For each Store+Product, ensure continuous daily index and fill gaps
    groups = []
    for (s, p), g in df.groupby(['Store', 'Product']):
        g = g.set_index('Date').sort_index()
        # create daily index between min and max
        idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
        g = g.reindex(idx)
        # bring back Store/Product info
        g['Store'] = s
        g['Product'] = p
        # fill static or periodic columns
        for col in ['Price', 'Seasonality', 'Holiday_Promo']:
            if col in g.columns:
                g[col] = g[col].ffill().bfill().fillna(method='ffill').fillna(0)
            else:
                g[col] = 0
        # numeric fills
        for col in ['Units_Sold', 'Inventory_Level']:
            if col in g.columns:
                g[col] = g[col].fillna(0)
            else:
                g[col] = 0
        g['DayOfWeek'] = g.index.dayofweek
        g['Month'] = g.index.month
        g = g.reset_index().rename(columns={'index': 'Date'})
        groups.append(g)
    big = pd.concat(groups, ignore_index=True)
    # create lag and rolling features per Store+Product
    big = big.sort_values(['Store', 'Product', 'Date']).reset_index(drop=True)
    big['lag_1'] = big.groupby(['Store', 'Product'])['Units_Sold'].shift(1).fillna(0)
    big['lag_7'] = big.groupby(['Store', 'Product'])['Units_Sold'].shift(7).fillna(0)
    big['ma_7'] = big.groupby(['Store', 'Product'])['Units_Sold'].transform(lambda x: x.rolling(7, min_periods=1).mean()).fillna(0)
    big['ma_30'] = big.groupby(['Store', 'Product'])['Units_Sold'].transform(lambda x: x.rolling(30, min_periods=1).mean()).fillna(0)
    # drop early rows where not enough history if desired (but we keep for model)
    return big


# ---------------------------
# Model training
# ---------------------------
def encode_categories(df, cols):
    enc_maps = {}
    for c in cols:
        df[c] = df[c].astype(str)
        cats = list(df[c].unique())
        mapping = {v: i for i, v in enumerate(cats)}
        enc_maps[c] = mapping
        df[c + '_code'] = df[c].map(mapping).fillna(-1).astype(int)
    return enc_maps


def build_model_and_train(df, feature_cols, target_col='Units_Sold'):
    X = df[feature_cols].fillna(0.0)
    y = df[target_col].astype(float)
    # train-test split by time: keep last 15% for test
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    model = RandomForestRegressor(n_estimators=150, max_depth=18, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Model evaluation on hold-out (time-split): MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
    return model


# ---------------------------
# Iterative forecast per Store+Product
# ---------------------------
def iterative_forecast(model, df_recent, feature_cols, horizon=30):
    """
    df_recent: a DataFrame containing the most recent continuous rows per Store+Product (sorted)
    We'll perform iterative forecasts horizon days ahead per Store+Product by updating lag features
    """
    preds = []
    for (s, p), g in df_recent.groupby(['Store', 'Product']):
        g = g.sort_values('Date').reset_index(drop=True)
        # take the last row as seed
        last = g.iloc[-1:].copy().reset_index(drop=True)
        # maintain recent history series to update lags (we'll use a simple list)
        recent_sales = list(g['Units_Sold'].values[-30:].astype(float))  # up to 30 last days
        for day in range(1, horizon + 1):
            # build a feature row representing the next day
            next_row = last.copy()
            next_date = pd.to_datetime(next_row.at[0, 'Date']) + pd.Timedelta(days=1)
            next_row.at[0, 'Date'] = next_date
            next_row.at[0, 'DayOfWeek'] = next_date.dayofweek
            next_row.at[0, 'Month'] = next_date.month
            # update lags/rolling based on recent_sales
            lag_1 = recent_sales[-1] if len(recent_sales) >= 1 else 0.0
            lag_7 = recent_sales[-7] if len(recent_sales) >= 7 else (recent_sales[0] if recent_sales else 0.0)
            ma_7 = float(np.mean(recent_sales[-7:])) if len(recent_sales) >= 1 else 0.0
            ma_30 = float(np.mean(recent_sales[-30:])) if len(recent_sales) >= 1 else 0.0
            next_row.at[0, 'lag_1'] = lag_1
            next_row.at[0, 'lag_7'] = lag_7
            next_row.at[0, 'ma_7'] = ma_7
            next_row.at[0, 'ma_30'] = ma_30
            # seasonal and price assumed constant from last known
            # prepare feature vector
            Xf = next_row[feature_cols].fillna(0.0).values.reshape(1, -1)
            # predict
            pred = model.predict(Xf)[0]
            # store prediction
            preds.append({
                'Store': s, 'Product': p, 'Date': next_date, 'predicted_units_sold': float(pred)
            })
            # update recent_sales with predicted value to simulate next-step lags
            recent_sales.append(float(pred))
            if len(recent_sales) > 60:
                recent_sales = recent_sales[-60:]
            # update last to new day (so next iteration increments)
            last = next_row
    return pd.DataFrame(preds)


# ---------------------------
# Stock recommendation logic
# ---------------------------
def compute_recommendations(df_hist_stats, df_preds_agg, safety_margin=0.12, lead_time_days=7, review_days=7, service_level=0.95):
    """
    df_hist_stats: historical aggregations per Store+Product including avg_actual_daily, std_actual_daily, avg_inventory, avg_price
    df_preds_agg: aggregated predictions per Store+Product (avg_pred_daily)
    """
    stats = pd.merge(df_hist_stats, df_preds_agg, on=['Store', 'Product'], how='left').fillna(0.0)
    recs = []
    # z-score mapping for common service levels
    z_map = {0.90: 1.282, 0.95: 1.645, 0.98: 2.055, 0.99: 2.326}
    z = z_map.get(round(service_level, 2), 1.645)
    for _, r in stats.iterrows():
        avg_pred = r['avg_pred_daily'] if r['avg_pred_daily'] > 0 else r['avg_actual_daily']
        sigma = r['std_actual_daily'] if not pd.isna(r['std_actual_daily']) else 0.0
        safety_stock = z * sigma * math.sqrt(lead_time_days)
        reorder_point = avg_pred * lead_time_days + safety_stock
        target_weekly = avg_pred * review_days * (1.0 + safety_margin)
        current_inventory = r['avg_inventory']
        recommended_order = max(0, int(round(target_weekly - current_inventory)))
        # EOQ fallback (simple assumptions)
        D = max(1.0, avg_pred * 365.0)
        unit_price = r['avg_price'] if r['avg_price'] > 0 else 1.0
        order_cost = 50.0
        carrying_rate = 0.2
        H = unit_price * carrying_rate
        eoq = int(round(math.sqrt(2.0 * D * order_cost / max(0.01, H))))
        final_order = max(recommended_order, eoq) if recommended_order > 0 else 0
        recs.append({
            'Store': r['Store'],
            'Product': r['Product'],
            'avg_actual_daily': round(r['avg_actual_daily'], 3),
            'avg_pred_daily': round(avg_pred, 3),
            'std_actual_daily': round(sigma, 3),
            'avg_inventory': int(round(current_inventory)),
            'safety_stock': int(round(safety_stock)),
            'reorder_point': int(round(reorder_point)),
            'recommended_weekly_stock': int(round(target_weekly)),
            'recommended_order': int(recommended_order),
            'eoq': int(eoq),
            'final_order_qty': int(final_order)
        })
    return pd.DataFrame(recs)


# ---------------------------
# Plotting functions
# ---------------------------
def plot_monthly_trend(df):
    monthly = df.groupby(df['Date'].dt.to_period('M'))['Units_Sold'].sum().reset_index()
    monthly['month_str'] = monthly['Date'].astype(str)
    plt.figure(figsize=(12, 5))
    plt.plot(monthly['month_str'], monthly['Units_Sold'], marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.title("Monthly total Units Sold")
    plt.ylabel("Total units sold")
    plt.tight_layout()
    plt.show()


def plot_seasonality(df):
    if 'Seasonality' in df.columns:
        s = df.groupby('Seasonality')['Units_Sold'].mean().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        plt.bar(s.index, s.values)
        plt.title("Average Units Sold by Season")
        plt.ylabel("Avg units sold")
        plt.tight_layout()
        plt.show()


def plot_top_predictions(preds_agg, top_n=10):
    top = preds_agg.groupby('Product')['avg_pred_daily'].mean().nlargest(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(top.index.astype(str), top.values)
    plt.title(f"Top {top_n} products by predicted avg daily demand")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main orchestration
# ---------------------------
def main(args):
    print("Loading and preprocessing data...", flush=True)
    df = preprocess_load(args.data)

    print("Creating time features and lags...", flush=True)
    big = create_time_features(df)

    # encode categories and prepare features
    enc = encode_categories(big, ['Store', 'Product', 'Seasonality'])
    feature_cols = ['Price', 'Inventory_Level', 'DayOfWeek', 'Month',
                    'lag_1', 'lag_7', 'ma_7', 'ma_30',
                    'Store_code', 'Product_code', 'Seasonality_code', 'Holiday_Promo']
    for c in feature_cols:
        if c not in big.columns:
            big[c] = 0.0

    print("Training RandomForest model...", flush=True)
    model = build_model_and_train(big, feature_cols, target_col='Units_Sold')

    # Forecast next N days
    recent = big.groupby(['Store', 'Product']).tail(60).reset_index(drop=True)
    preds = iterative_forecast(model, recent, feature_cols, horizon=args.horizon)

    if preds.empty:
        print("⚠️ No predictions generated. Exiting main().", flush=True)
        return

    # Aggregate predictions
    preds_agg = preds.groupby(['Store', 'Product']).agg(
        avg_pred_daily=('predicted_units_sold', 'mean'),
        sample_pred_days=('predicted_units_sold', 'count')
    ).reset_index()

    # Historical stats
    hist_stats = big.groupby(['Store', 'Product']).agg(
        avg_actual_daily=('Units_Sold', 'mean'),
        std_actual_daily=('Units_Sold', 'std'),
        avg_inventory=('Inventory_Level', 'mean'),
        avg_price=('Price', 'mean'),
        sample_days=('Units_Sold', 'count')
    ).reset_index()

    print("Computing stock recommendations...", flush=True)
    recs = compute_recommendations(hist_stats, preds_agg,
                                   safety_margin=args.safety_margin,
                                   lead_time_days=args.lead_time,
                                   review_days=args.review_days,
                                   service_level=args.service_level)

    # Save CSV
    recs.to_csv("stock_recommendations.csv", index=False)
    preds.to_csv("predictions_next30days.csv", index=False)
    print("✅ Saved CSVs: predictions_next30days.csv, stock_recommendations.csv", flush=True)

    # ---------------------------
    # Display Top Restock Recommendations
    # ---------------------------
    recommendations_sorted = recs.sort_values('final_order_qty', ascending=False)
    top10 = recommendations_sorted.head(10)

    print("\n===== TOP 10 RESTOCK RECOMMENDATIONS =====", flush=True)
    for _, row in top10.iterrows():
        store = row['Store']
        product = row['Product']
        qty = int(row['final_order_qty'])
        print(f"Store {store} - Product {product} → {'🔁 Restock '+str(qty) if qty>0 else '✅ No restock needed'}", flush=True)
    print("==========================================\n", flush=True)

    print("📊 Full Restock Summary:", flush=True)
    for _, row in recommendations_sorted.iterrows():
        store = row['Store']
        product = row['Product']
        qty = int(row['final_order_qty'])
        status = f"🔁 Restock {qty}" if qty > 0 else "✅ No restock needed"
        print(f"{store} | {product} → {status}", flush=True)

    # Optional: plots
    try:
        plot_monthly_trend(big)
        plot_seasonality(big)
        plot_top_predictions(preds_agg, top_n=12)
    except Exception as e:
        print("Plotting failed:", e, flush=True)

    print("\n✅ Done.", flush=True)


if __name__ == "__main__":
    class Args:
        data = "/Users/estebanvillacorta/Documents/USN/3rd year/AI3000/G11 Assignment/retail_store_inventory.csv"
        horizon = 30
        lead_time = 7
        review_days = 7
        safety_margin = 0.12
        service_level = 0.95

    args = Args()
    print(f"📁 Using dataset: {args.data}", flush=True)
    try:
        main(args)
    except Exception as exc:
        print("❌ Error:", exc, flush=True)
        sys.exit(1)

