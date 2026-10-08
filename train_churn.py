"""Train the churn model exactly as described in the report and save artefacts.

Usage:
    python train_churn.py [path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv]

Outputs:
    models/churn_model.joblib   (pipeline + chosen decision threshold)
    models/churn_metrics.json   (held-out test metrics shown in the app)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_features import CAT_COLS, NUM_COLS, RAW_COLS, add_features

SEED = 42
ROOT = Path(__file__).resolve().parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

# ---- 1. Load & audit -------------------------------------------------------
df = pd.read_csv(CSV)
df = df.drop(columns=["customerID"], errors="ignore").drop_duplicates()
y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
X = df[RAW_COLS]
print(f"Rows: {len(df)} | churn rate: {y.mean():.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---- 2. Leak-free pipeline: features -> preprocess -> SMOTE -> RF ----------
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(
    [("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)], sparse_threshold=0)

model = ImbPipeline(steps=[
    ("features", FunctionTransformer(add_features)),
    ("prep", preprocessor),
    ("smote", SMOTE(k_neighbors=5, random_state=SEED)),   # only runs on training folds
    ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
])

# ---- 3. Hyper-parameter search (F1 on churn class, stratified 5-fold) ------
param_dist = {
    "clf__n_estimators": [200, 300, 400],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
    "clf__class_weight": [None, "balanced"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(model, param_dist, n_iter=12, scoring="f1", cv=cv,
                            random_state=SEED, n_jobs=1, verbose=1)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_)

# ---- 4. Threshold tuning on out-of-fold TRAIN predictions (favour recall) ---
oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
grid = np.arange(0.20, 0.80, 0.02)
threshold = float(max(grid, key=lambda t: fbeta_score(y_train, oof >= t, beta=2)))
print(f"Chosen threshold (max F2 on OOF): {threshold:.2f}")

# ---- 5. One-shot evaluation on held-out test set ---------------------------
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
names = best.named_steps["prep"].get_feature_names_out()
imps = best.named_steps["clf"].feature_importances_
top = sorted(zip(names, imps), key=lambda p: -p[1])[:12]

metrics = {
    "precision": float(precision_score(y_test, pred)),
    "recall": float(recall_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "threshold": threshold,
    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "churn_rate": float(y.mean()),
    "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
    "top_features": [[n.split("__", 1)[-1], float(v)] for n, v in top],
}
print(json.dumps(metrics, indent=2, default=str))

joblib.dump({"model": best, "threshold": threshold}, OUT / "churn_model.joblib", compress=3)
(OUT / "churn_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
print("Saved to", OUT)
