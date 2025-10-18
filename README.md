# AI-Powered Stock Recommender

This project is a single-file Python application that predicts daily units sold for products in retail stores and provides stock replenishment recommendations. It uses a Random Forest model and computes safety stock, reorder points, and recommended orders.

---

## Features

* Auto-detects relevant columns in your CSV file.
* Trains a Random Forest to predict daily units sold.
* Iteratively forecasts the next N days per Store+Product.
* Computes safety stock, reorder points, and weekly reorder recommendations.
* Generates plots and saves CSV outputs:

  * `predictions_next30days.csv`
  * `stock_recommendations.csv`

---

## Requirements

* Python 3.9+
* Conda (recommended but optional)
* Required Python packages:

  ```bash
  pip install pandas numpy scikit-learn matplotlib
  ```

---

## Usage

1. Clone the repository or copy the project folder locally.

2. Place your inventory CSV in a known path. The CSV must include at least:

   * Date column (e.g., `Date`)
   * Product column (e.g., `Product`, `SKU`)
   * Units Sold column (e.g., `Units_Sold`, `Quantity`)

3. Modify the `data` path in `ai_stock_recommender.py` depending on your OS:

   **Mac / Linux**

   ```python
   data = "/Users/yourusername/path/to/retail_store_inventory.csv"
   ```

   **Windows**

   ```python
   data = "C:\\Users\\yourusername\\path\\to\\retail_store_inventory.csv"
   ```

   **Tip:** Use raw strings `r"..."` or double backslashes `\\` for Windows paths.

4. (Optional) Create a Conda environment for isolation:

   ```bash
   conda create -n ai_stock python=3.10
   conda activate ai_stock
   pip install pandas numpy scikit-learn matplotlib
   ```

5. Run the recommender:

   ```bash
   python ai_stock_recommender.py
   ```

---

## Outputs

* `predictions_next30days.csv`: Iterative daily sales forecasts per Store+Product.
* `stock_recommendations.csv`: Recommended reorder quantities, EOQ, safety stock, etc.
* Console prints top 10 actionable restock recommendations and a full summary.
* Optional plots showing:

  * Monthly sales trends
  * Seasonality
  * Top products by predicted demand

---

## Notes

* Designed for university/learning purposes.
* The CSV path **must be updated** to match your environment.
* Make sure your CSV has a continuous daily time series per product if possible.
