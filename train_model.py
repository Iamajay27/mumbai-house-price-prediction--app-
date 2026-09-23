
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "mumbai_house_price_data_cleaned.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = [
    "area", "locality", "property_type", "bedroom_num", "bathroom_num",
    "balcony_num", "furnished", "age", "total_floors", "latitude", "longitude"
]
target = "price"
df = df.dropna(subset=features + [target]).copy()
df = df[df[target] > 0].copy()

categorical = ["locality", "property_type", "furnished"]
categories = {}
for c in categorical:
    cats = sorted(df[c].astype(str).unique().tolist())
    categories[c] = cats
    df[c] = pd.Categorical(df[c].astype(str), categories=cats)

X = df[features]
y = np.log1p(df[target].astype(float))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LGBMRegressor(
    objective="regression", n_estimators=2500, learning_rate=0.03,
    num_leaves=63, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=0.2, random_state=42, n_jobs=-1
)
model.fit(
    X_train, y_train, categorical_feature=categorical,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
)

pred_log = model.predict(X_test)
pred = np.maximum(0, np.expm1(pred_log))
actual = np.expm1(y_test)

metadata = {
    "features": features,
    "categorical_features": categorical,
    "categories": categories,
    "metrics": {
        "cv_or_holdout_rmse_log": float(mean_squared_error(y_test, pred_log) ** 0.5),
        "holdout_mae_rupees": float(mean_absolute_error(actual, pred)),
        "holdout_r2": float(r2_score(actual, pred)),
    },
    "dataset_rows": len(df),
    "localities": categories["locality"],
    "property_types": categories["property_type"],
    "furnished_options": categories["furnished"],
    "ranges": {
        "area": [float(df.area.min()), float(df.area.max())],
        "bedroom_num": [int(df.bedroom_num.min()), int(df.bedroom_num.max())],
        "bathroom_num": [int(df.bathroom_num.min()), int(df.bathroom_num.max())],
        "balcony_num": [int(df.balcony_num.min()), int(df.balcony_num.max())],
        "age": [int(df.age.min()), int(df.age.max())],
        "total_floors": [int(df.total_floors.min()), int(df.total_floors.max())],
    },
}

joblib.dump(model, MODEL_DIR / "mumbai_house_price_lgbm.joblib")
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

geo = df.groupby("locality")[["latitude", "longitude"]].median().dropna().to_dict(orient="index")
(MODEL_DIR / "locality_geo.json").write_text(json.dumps(geo, indent=2), encoding="utf-8")

print("Model trained and saved successfully.")
print(f"Holdout log RMSE: {metadata['metrics']['cv_or_holdout_rmse_log']:.4f}")
print(f"Holdout R2: {metadata['metrics']['holdout_r2']:.4f}")
