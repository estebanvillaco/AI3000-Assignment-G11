#!/usr/bin/env python3
"""
AI in Retail Business - Smarter Stock Recommender (Seasonal Menu Version)
------------------------------------------------------------------------
This version predicts product demand and restock recommendations
for a selected season (Winter, Spring, Summer, Autumn).

The program displays a menu so you can choose which season to analyze.
The printed output is written in simple language for business users.
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# === 1. Load and prepare data ===
data = pd.read_csv("retail_store_inventory.csv")

# Normalize column names
data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

# Detect and rename key columns
date_col = "date"
product_col = "product_id"
sales_col = "units_sold"
inventory_col = "inventory_level"
forecast_col = "demand_forecast"

# Extract relevant features
df = data[[date_col, product_col, sales_col, inventory_col, forecast_col]].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna()

# === 2. Add season feature ===
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["month"] = df["date"].dt.month
df["season"] = df["month"].apply(get_season)
df["day_of_week"] = df["date"].dt.dayofweek
df["product_code"] = df["product_id"].astype("category").cat.codes

# === 3. Season selection menu ===
print("=== AI Stock Recommender ===")
print("Select a season to analyze:")
print("1. Winter")
print("2. Spring")
print("3. Summer")
print("4. Autumn")

choice = input("Enter your choice (1-4): ").strip()

season_map = {"1": "Winter", "2": "Spring", "3": "Summer", "4": "Autumn"}
season = season_map.get(choice)

if not season:
    print("\nInvalid choice. Please run the program again and choose a number from 1 to 4.")
    exit()

print(f"\n SEASONAL ANALYSIS: {season}")
print("=" * 70)

# === 4. Filter data for selected season ===
season_data = df[df["season"] == season]
if len(season_data) < 50:
    print(f"Not enough data for {season} — please choose another season.\n")
    exit()

# === 5. Train/test split ===
train = season_data.iloc[:-30]
test = season_data.iloc[-30:]

X_train = train[["product_code", "inventory_level", "day_of_week", "month"]]
y_train = train["units_sold"]
X_test = test[["product_code", "inventory_level", "day_of_week", "month"]]
y_test = test["units_sold"]

# === 6. Train Decision Tree model ===
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluate model ===
preds = model.predict(X_test)
mae = round(mean_absolute_error(y_test, preds), 2)
r2 = round(r2_score(y_test, preds), 3)

print("Model Performance Summary:")
print(f"• Average prediction error (MAE): {mae} units")
print(f"• Model accuracy (R²): {r2}")
print("This means the model explains about {:.0f}% of sales variation.\n".format(r2 * 100))

# === 8. Predict next-day demand ===
recent = season_data.groupby("product_id").tail(1).copy()
recent["predicted_sales"] = recent["demand_forecast"]
recent.loc[recent["predicted_sales"].isna(), "predicted_sales"] = model.predict(
    recent[["product_code", "inventory_level", "day_of_week", "month"]]
)

# === 9. Restock logic ===
recent["days_of_coverage"] = recent["inventory_level"] / (recent["predicted_sales"] + 1)
recent["restock_recommended"] = recent["days_of_coverage"] < 7
recent["recommended_order"] = (
    (recent["predicted_sales"] * 14 - recent["inventory_level"]).clip(lower=0).round()
)
recent["sales_to_stock_ratio"] = recent["predicted_sales"] / (recent["inventory_level"] + 1)

# === 10. Show results ===
top_sales = recent.sort_values("predicted_sales", ascending=False).head(5)
top_combined = recent.sort_values("sales_to_stock_ratio", ascending=False).head(5)

print("Top 5 Bestselling Products (Predicted Highest Demand):")
for _, row in top_sales.iterrows():
    print(f"• {row['product_id']}: Expected to sell about {row['predicted_sales']:.0f} units soon, "
          f"{int(row['inventory_level'])} in stock → recommend ordering {int(row['recommended_order'])} more.")
print()

print("Products Selling Fast and May Need Restocking Soon:")
for _, row in top_combined.iterrows():
    print(f"• {row['product_id']}: Selling quickly ({row['predicted_sales']:.0f} units/day) "
          f"with only {int(row['inventory_level'])} in stock → suggest ordering {int(row['recommended_order'])} more.")
print()

print("Business Insight Summary:")
print(f"This analysis focuses on {season.lower()} demand patterns.")
print("It helps ensure you have enough stock for seasonal demand peaks, keeping about two weeks of stock ready.")
print("This helps avoid sellouts during high-demand periods while reducing unnecessary overstock.\n")

# === 11. Save seasonal results ===
output_file = f"smart_stock_recommendations_{season.lower()}.csv"
top_combined.to_csv(output_file, index=False)
print(f"Detailed {season} results saved to: {output_file}\n")
