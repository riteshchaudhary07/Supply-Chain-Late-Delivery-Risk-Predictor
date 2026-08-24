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

## Run locally

```bash
python -m venv mlven
source mlven/bin/activate        # Windows: mlven\Scripts\activate
pip install -r requirements.txt

python data/generate_data.py     # build the dataset
python train.py                  # train model, save artifacts
streamlit run app.py             # launch the app
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo (make sure `data/supply_chain_data.csv`
   and `artifacts/*.pkl` are committed — Streamlit Cloud needs them at runtime).
2. Go to **share.streamlit.io** → **New app**.
3. Select your repo/branch, set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs everything from `requirements.txt`.

## Troubleshooting

**`AttributeError` when running `streamlit run app.py` (e.g. `has no attribute '_RemainderColsList'`)**
This means the `.pkl` files were saved by a *different* scikit-learn version than the one
currently loading them — almost always because your terminal isn't using the venv. Fix:
```bash
source mlven/bin/activate      # make sure your prompt shows (mlven)
which streamlit                 # should point inside mlven/, not /Library/Frameworks/...
streamlit run app.py
```
If it still fails, delete `artifacts/*.pkl` and run `python train.py` again **inside the
same activated venv** you use to run Streamlit, so both use the same scikit-learn build.

## Talking points for an interview

- **Why a Pipeline object?** Preprocessing (OneHotEncoder + StandardScaler) and
  the model are bundled together, so `app.py` never has to remember how to
  transform raw input — `pipeline.predict()` does it all, and there's no
  train/serve skew.
- **Why `scale_pos_weight`?** The late/on-time classes are imbalanced (~35/65),
  so this reweights the loss instead of letting the model just predict the
  majority class.
- **Why these metrics?** Accuracy alone is misleading on imbalanced data —
  the notebook reports precision/recall/F1/ROC-AUC and a full classification
  report so recall on the "late" class (the one that matters for the business)
  is visible.
- **What would you do next?** Hyperparameter tuning (Optuna/RandomizedSearchCV),
  SHAP for feature-level explanations, and swapping the synthetic data for the
  real Kaggle dataset.
- **Why not push accuracy above 95%?** On a *real* delivery dataset, a late-delivery
  outcome depends on factors (traffic, warehouse staffing, weather) that aren't in
  any table, so some irreducible error is expected — genuine real-world models for
  this kind of problem usually land around 75-90% accuracy. This project's current
  numbers (~98% accuracy, ~98% precision, ~97% recall) come from a synthetic dataset
  with a deliberately clean, close-to-deterministic rule (`data/generate_data.py`) —
  built this way on request for a polished portfolio demo. Be ready to say so plainly
  if an interviewer asks how the numbers got this clean; swapping in the real Kaggle
  CSV (same column names) will show more realistic, noisier performance.
- **Why no PCA in the pipeline?** XGBoost is tree-based, so it doesn't need PCA to
  handle correlated/high-dimensional features the way linear models do — and PCA would
  destroy the clean feature-importance chart, which is one of the best interview talking
  points this project has. PCA is still shown on the dashboard's Insights tab as a
  visualization only (not fed into the model).
