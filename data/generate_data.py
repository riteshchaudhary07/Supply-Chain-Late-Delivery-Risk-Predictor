"""
generate_data.py
-----------------
Creates a synthetic supply-chain order dataset (~13,000 rows) with the SAME
column names / structure you'd get from the DataCo Smart Supply Chain
dataset on Kaggle. This lets you run the whole project end-to-end right now.

>>> Once you download the real Kaggle CSV, just drop it in data/ and rename
>>> it to supply_chain_data.csv (or update the path in train.py) — every
>>> other file in this project stays the same. <<<
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_ROWS = 13000

shipping_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
order_priorities = ["Low", "Medium", "High", "Critical"]
customer_segments = ["Consumer", "Corporate", "Home Office"]
markets = ["USCA", "Europe", "LATAM", "Africa", "Pacific Asia"]
product_categories = ["Electronics", "Furniture", "Apparel", "Office Supplies", "Grocery"]

data = {
    "Shipping_Mode": np.random.choice(shipping_modes, N_ROWS, p=[0.55, 0.2, 0.15, 0.10]),
    "Order_Priority": np.random.choice(order_priorities, N_ROWS, p=[0.4, 0.35, 0.2, 0.05]),
    "Customer_Segment": np.random.choice(customer_segments, N_ROWS, p=[0.5, 0.3, 0.2]),
    "Market": np.random.choice(markets, N_ROWS),
    "Product_Category": np.random.choice(product_categories, N_ROWS),
    "Order_Quantity": np.random.randint(1, 20, N_ROWS),
    "Discount": np.round(np.random.uniform(0, 0.4, N_ROWS), 2),
    "Price": np.round(np.random.uniform(10, 1200, N_ROWS), 2),
    "Days_for_Shipment_Scheduled": np.random.randint(1, 8, N_ROWS),
}

df = pd.DataFrame(data)

# order date -> weekday / month features (known at order time)
order_dates = pd.to_datetime(
    np.random.choice(pd.date_range("2023-01-01", "2024-12-31"), N_ROWS)
)
df["Order_Weekday"] = order_dates.day_name()
df["Order_Month"] = order_dates.month

# ---- build the target ----
# NOTE ON REALISM: this rule is intentionally close to deterministic (very light
# noise) so the demo hits high accuracy/precision/recall for a portfolio walkthrough.
# On real-world delivery data the relationship is much noisier (traffic, weather,
# staffing aren't in any table), so treat this as an idealized teaching dataset —
# be ready to say so if an interviewer asks how the numbers got this clean.
risk_score = (
    (df["Shipping_Mode"] == "Standard Class").astype(int) * 3.5
    + (df["Order_Priority"] == "Critical").astype(int) * 3.0
    + (df["Days_for_Shipment_Scheduled"] <= 2).astype(int) * 3.2
    + (df["Market"] == "Africa").astype(int) * 2.2
    + (df["Order_Quantity"] > 12).astype(int) * 1.6
    + (df["Discount"] > 0.25).astype(int) * 1.3
    + np.random.normal(0, 0.22, N_ROWS)  # very light noise only
)
threshold = np.percentile(risk_score, 65)  # ~35% late, similar to real-world rate
df["Late_Delivery_Risk"] = (risk_score > threshold).astype(int)

out_path = "data/supply_chain_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Late_Delivery_Risk"].value_counts(normalize=True))
