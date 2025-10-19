# AI in Retail Business - Simple Stock Recommender

This Python script predicts daily product sales for a retail store and provides simple restocking recommendations based on predicted demand. It uses a **Decision Tree Regressor** to identify products that may need replenishing soon.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Requirements](#requirements)
* [Usage](#usage)
* [How It Works](#how-it-works)
* [Output](#output)
* [Author & Course](#author--course)

---

## Overview

Retail businesses need to maintain optimal inventory levels to avoid overstocking or stockouts. This script uses historical sales and inventory data to:

* Predict daily units sold for each product.
* Recommend which products should be restocked.
* Suggest quantities to order based on predicted demand.

---

## Features

* Loads historical sales and inventory data from a CSV file.
* Cleans and preprocesses data, including date parsing and categorical encoding.
* Creates simple features such as **Day of Week** and **Month**.
* Trains a **Decision Tree Regressor** to predict units sold.
* Evaluates model performance using **MAE** and **R²** metrics.
* Provides **top 5 restocking recommendations** with suggested order quantities.
* Saves the recommendations to a CSV file.

---

## Requirements

* Python 3.x
* Libraries:

  * `pandas`
  * `scikit-learn`

Install dependencies using:

```bash
pip install pandas scikit-learn
```

---

## Usage

### 1. Prepare your CSV file

* Place your retail data CSV file in the same folder as the script.
* Ensure the CSV contains at least the following columns (names can vary slightly):

  * Date of sale (e.g., `Date`, `SaleDate`)
  * Product name (e.g., `Product`, `Item`)
  * Units sold (e.g., `Units_Sold`, `Sales`)
  * Inventory level (e.g., `Inventory`, `Stock`)


### 2. Run the script

Open a terminal, navigate to the folder containing the script, and run:

```bash
python3 simple_stock_recommender.py
```

## How It Works

1. **Data Preparation:**

   * Load historical sales and inventory data.
   * Standardize column names and handle missing values.
   * Extract features such as day of week, month, and product codes.

2. **Model Training:**

   * Split data into training (all but last 30 days) and testing (last 30 days).
   * Train a `DecisionTreeRegressor` on the features:
     `Product_Code`, `Inventory_Level`, `DayOfWeek`, `Month`.

3. **Prediction & Evaluation:**

   * Predict units sold for the test set.
   * Evaluate predictions using **Mean Absolute Error (MAE)** and **R² score**.

4. **Restock Recommendation:**

   * Predict sales for the most recent date of each product.
   * Compare predicted sales to current inventory.
   * Recommend restocking for products where predicted sales exceed inventory.

---

## Output

The script prints:

* Model performance metrics: MAE and R².
* Top 5 products that need restocking with columns:

  * `Product`
  * `Inventory_Level`
  * `Predicted_Sales`
  * `Recommended_Order`

It also saves the recommendations to `simple_stock_recommendations.csv`.


---

## Author & Course

**Author:** Esteban Villacorta, Jonas Ambaya, Adrian Lafjell Ed (Group 11) 
**Course:** AI3000R - Artificial Intelligence for Business Applications
