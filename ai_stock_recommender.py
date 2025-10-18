#!/usr/bin/env python3
"""
AI in Retail Business - Simple Stock Recommender
------------------------------------------------
This script predicts daily product sales using a Decision Tree model.
It helps identify which products may need restocking soon.

Author: [Your Group Name]
Course: AI3000R - Artificial Intelligence for Business Applications
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# === 1. Load and prepare data ===
data = pd.read_csv("retail_store_inventory.csv")

# Basic cleaning
data.columns = [c.strip().lower() for c in data.columns]

# Detect and rename key columns
date_col = [c for c in data.columns if "date" in c][0]
product_col = [c for c in data.columns if "product" in c or "item" in c][0]
sales_col = [c for c in data.columns if "sale" in c or "unit" in c][0]
inventory_col = [c for c in data.columns if "invent" in c or "stock" in c][0]

df = data[[date_col, product_col, sales_col, inventory_col]].copy()
df.columns = ["Date", "Product", "Units_Sold", "Inventory_Level"]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna()

# === 2. Create simple features ===
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

# Encode product as numbers
df["Product_Code"] = df["Product"].astype("category").cat.codes

# === 3. Split into train/test sets ===
train = df.iloc[:-30]
test = df.iloc[-30:]

X_train = train[["Product_Code", "Inventory_Level", "DayOfWeek", "Month"]]
y_train = train["Units_Sold"]
X_test = test[["Product_Code", "Inventory_Level", "DayOfWeek", "Month"]]
y_test = test["Units_Sold"]

# === 4. Train Decision Tree model ===
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === 5. Evaluate model ===
preds = model.predict(X_test)
print("Model evaluation:")
print("  MAE:", round(mean_absolute_error(y_test, preds), 2))
print("  R²:", round(r2_score(y_test, preds), 3))

# === 6. Simple restock recommendation ===
recent = df.groupby("Product").tail(1)
recent["Predicted_Sales"] = model.predict(
    recent[["Product_Code", "Inventory_Level", "DayOfWeek", "Month"]]
)

# if predicted sales > current stock → restock
recent["Restock_Recommended"] = recent["Predicted_Sales"] > recent["Inventory_Level"]
recent["Recommended_Order"] = (
    (recent["Predicted_Sales"] - recent["Inventory_Level"]).clip(lower=0).round()
)

top = recent.sort_values("Recommended_Order", ascending=False).head(5)
print("\nTop 5 Products to Restock:")
print(top[["Product", "Inventory_Level", "Predicted_Sales", "Recommended_Order"]])

# === 7. Save results ===
top.to_csv("simple_stock_recommendations.csv", index=False)
print("\nSaved results to simple_stock_recommendations.csv")
