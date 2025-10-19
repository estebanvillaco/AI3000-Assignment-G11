# AI in Retail Business – Smarter Stock Recommender (Seasonal Menu Version)

This project predicts product demand and provides restock recommendations for retail stores, based on seasonal trends. It is designed to be user-friendly for business users and helps optimize inventory management.

---

## Features

* **Seasonal Analysis:** Choose from Winter, Spring, Summer, or Autumn to analyze seasonal sales patterns.
* **Demand Prediction:** Uses a Decision Tree Regressor to forecast daily product sales.
* **Restock Recommendations:** Identifies fast-selling products and calculates suggested reorder quantities.
* **Business Insights:** Provides easy-to-understand summaries of sales trends and inventory recommendations.
* **Export Results:** Saves detailed seasonal recommendations to a CSV file for further analysis.

---

## Requirements

* Python 3.8+
* `pandas`
* `scikit-learn`

Install dependencies using pip:

```bash
pip install pandas scikit-learn
```

---

## Usage

1. Place your retail data CSV file as `retail_store_inventory.csv` in the project directory.
2. Run the program:

```bash
python smart_stock_recommender.py
```

3. Select a season when prompted:

```
1. Winter
2. Spring
3. Summer
4. Autumn
```

4. The program will output:

   * Top 5 predicted best-selling products.
   * Products selling quickly that may need restocking.
   * A business insight summary.
   * A CSV file with detailed recommendations: `smart_stock_recommendations_<season>.csv`.

---

## Data Format

Your `retail_store_inventory.csv` should include at least the following columns:

* `date` – The date of the sales record.
* `product_id` – Unique product identifier.
* `units_sold` – Number of units sold on that date.
* `inventory_level` – Current inventory level of the product.
* `demand_forecast` – (Optional) Precomputed forecast values.

Column names are normalized automatically, so minor variations are handled.

---

## How It Works

1. **Data Preparation:** Cleans and formats the dataset, extracts features like month, day of week, and season.
2. **Season Filtering:** Focuses analysis on the chosen season.
3. **Model Training:** Uses a Decision Tree Regressor to predict future sales based on historical trends.
4. **Restock Logic:** Calculates days of coverage and recommended reorder quantities to maintain optimal stock levels.
5. **Reporting:** Displays predictions in plain language and exports detailed CSV reports.

---

## Example Output

```
Top 5 Bestselling Products (Predicted Highest Demand):
• P123: Expected to sell about 50 units soon, 20 in stock → recommend ordering 80 more.
• P456: Expected to sell about 45 units soon, 15 in stock → recommend ordering 75 more.

Products Selling Fast and May Need Restocking Soon:
• P789: Selling quickly (30 units/day) with only 5 in stock → suggest ordering 55 more.

Business Insight Summary:
This analysis focuses on summer demand patterns.
It helps ensure you have enough stock for seasonal demand peaks, keeping about two weeks of stock ready.
```

## Author & Course

**Author:** Esteban Villacorta, Jonas Ambaya, Adrian Lafjell Ed (Group 11) 
**Course:** AI3000R - Artificial Intelligence for Business Applications
