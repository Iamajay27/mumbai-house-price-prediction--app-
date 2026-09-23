# 🏠 Mumbai House Price Predictor

A Streamlit machine-learning application that estimates Mumbai residential property prices using **LightGBM**.

## Dataset
- Records: **71,938**
- Features: property area, locality, property type, BHK, bathrooms, balconies, furnishing, age, total floors and geographic coordinates.
- `price_per_sqft` is intentionally excluded from model inputs because it is derived from the target price and would cause target leakage.

## Model
The target is trained as `log1p(price)` and converted back with `expm1()` for prediction.

Holdout results from the current training run:
- Log RMSE: **0.1837**
- R²: **0.8890**
- MAE: **₹2,817,345**

These metrics are from one 80/20 holdout split and should not be presented as a guaranteed real-world valuation accuracy.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Project structure

```text
mumbai_house_price_predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── mumbai_house_price_data_cleaned.csv
└── model/
    ├── mumbai_house_price_lgbm.joblib
    ├── metadata.json
    └── locality_geo.json
```

## Important
The prediction is an ML estimate, not a certified property valuation. Exact market price can vary because of building condition, floor, view, amenities, road access, legal status and current market conditions.
