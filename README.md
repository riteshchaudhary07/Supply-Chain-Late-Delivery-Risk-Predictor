# 📦 Supply Chain — Late Delivery Risk Prediction (XGBoost)

Predicts whether an order will be **delivered late**, using order details known
at the time the order is placed (shipping mode, priority, market, quantity, etc).

## Problem type
Binary classification — `Late_Delivery_Risk`: `0` = on time, `1` = late.

## Folder structure

```
SUPPLY_CHAIN_DELAY_PREDICTION_ML_PROJECT/
├── data/
│   ├── generate_data.py        # creates the synthetic dataset (swap for real Kaggle CSV)
│   └── supply_chain_data.csv   # training data (13,000 rows)
├── notebook/
│   └── Supply_Chain_Delay_Prediction.ipynb   # EDA + experimentation (readable walkthrough)
├── artifacts/
│   ├── model_pipeline.pkl      # full sklearn Pipeline (preprocessing + XGBoost model)
│   └── metadata.pkl            # feature lists + dropdown options, used by app.py
├── train.py                    # script version of the notebook — produces the artifacts
├── app.py                      # Streamlit app (deployment entry point)
├── requirements.txt
├── .python-version
└── README.md
```

## How the pieces fit together

1. **`data/generate_data.py`** builds the dataset. Replace this with the real
   Kaggle DataCo Smart Supply Chain CSV any time — just keep the same column
   names, or edit `CATEGORICAL_FEATURES` / `NUMERIC_FEATURES` in `train.py`.
2. **`train.py`** — loads data → builds an sklearn `Pipeline` (OneHotEncoder +
   StandardScaler → XGBClassifier) → trains → evaluates → saves the pipeline
   and metadata into `artifacts/`.
3. **`notebook/Supply_Chain_Delay_Prediction.ipynb`** — the same logic broken
   into cells with explanations, so you can walk an interviewer through your
   thinking (EDA → preprocessing → training → evaluation → feature importance).
4. **`app.py`** — loads `artifacts/model_pipeline.pkl` and `metadata.pkl`,
   builds a form from the metadata (so you never hardcode dropdown values
   twice), and returns a live prediction. This is the file Streamlit Cloud runs.


## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo (make sure `data/supply_chain_data.csv`
   and `artifacts/*.pkl` are committed — Streamlit Cloud needs them at runtime).
2. Go to **share.streamlit.io** → **New app**.
3. Select your repo/branch, set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs everything from `requirements.txt`.


