"""
Train Random‑Forest & XGBoost on pre‑processed macro‑features
-------------------------------------------------------------
Assumes that 01_preprocess.py (або train_prepare_and_fit.py) уже
згенерував:
    dataset_ready/X_train.csv
    dataset_ready/X_test.csv
    dataset_ready/y_train.csv
    dataset_ready/y_test.csv
    dataset_ready/prep.pkl
Outputs into models/ :
    rf_model.pkl,  xgb_model.pkl                –  full Pipeline(prep, model)
    rf_feat_imp.csv, xgb_feat_imp.csv           –  feature importance
    cv_mae.json                                 –  5‑fold TS‑MAE
"""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ----------------------------------------------------- PATHS
ready_dir = Path("dataset_ready")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

X_train = pd.read_csv(ready_dir / "X_train.csv", index_col=0, parse_dates=True)
X_test  = pd.read_csv(ready_dir / "X_test.csv",  index_col=0, parse_dates=True)

# ←– ОНОВЛЕНО –→
y_train = pd.read_csv(ready_dir / "y_train.csv", index_col=0).squeeze("columns")
y_test  = pd.read_csv(ready_dir / "y_test.csv",  index_col=0).squeeze("columns")

prep = joblib.load(ready_dir / "prep.pkl")

# NB: if y_train is a DataFrame read with squeeze=False
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]
    y_test  = y_test.iloc[:, 0]

# ----------------------------------------------------- MODELS
models = {
    "RF": RandomForestRegressor(
            n_estimators=600, max_depth=None,
            max_features="sqrt", min_samples_leaf=10,
            n_jobs=-1, random_state=42),
    "XGB": XGBRegressor(
            n_estimators=900, learning_rate=0.03,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.5, reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1, random_state=42)
}

tscv = TimeSeriesSplit(n_splits=5)
cv_mae = {}

for name, est in models.items():
    pipe = Pipeline([("prep", prep), ("model", est)])

    # ------------- CV on train
    fold_mae = []
    for k, (tr, val) in enumerate(tscv.split(X_train)):
        pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = pipe.predict(X_train.iloc[val])
        mae  = mean_absolute_error(y_train.iloc[val], pred)
        fold_mae.append(mae)
        print(f"{name}  fold {k+1}:  MAE = {mae: .4f}")

    cv_mae[name] = float(np.mean(fold_mae))
    print(f"{name}  mean CV‑MAE: {cv_mae[name]: .4f}")

    # ------------- Fit on full train & save
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, model_dir / f"{name.lower()}_model.pkl")

    # ------------- Feature importance
    if name == "RF":
        imp = pipe.named_steps["model"].feature_importances_
    else:
        booster  = pipe.named_steps["model"].get_booster()
        fmap     = pipe.named_steps["prep"].get_feature_names_out()
        imp_dict = booster.get_score(importance_type="gain")
        imp      = np.array([imp_dict.get(f, 0.0) for f in fmap])

    pd.Series(imp,
              index=pipe.named_steps["prep"].get_feature_names_out()
             ).sort_values(ascending=False)\
              .to_csv(model_dir / f"{name.lower()}_feat_imp.csv")

# ----------------------------------------------------- METRICS OUT
with open(model_dir / "cv_mae.json", "w") as fp:
    json.dump(cv_mae, fp, indent=2)

print("\n✓  Training finished.  Models & metrics saved to /models/")
