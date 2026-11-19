"""
Train the BASE-PRICE model on the real Kaggle "Laptop Price" dataset (laptop specs -> launch price, EUR).

    python -m src.train_base_model            (run from the repository root)

Steps: load -> inspect -> feature extraction -> split -> compare 8 models by 5-fold CV -> pick best by CV ->
conformal price-range margins -> evaluation figures -> save models/base_price_model.joblib + results_real.json
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from src.features import build_features, standardise, CAT, NUM

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=.95)
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figs"; FIG.mkdir(exist_ok=True)
SEED = 42; C1, C2 = "#2F5D8C", "#E8772E"
R = {}

def save(name): plt.tight_layout(); plt.savefig(FIG / f"{name}.png", dpi=170); plt.close()

# ---- 1. load & inspect
raw = standardise(pd.read_csv(ROOT / "data" / "laptop_price_dataset.csv"))
R["shape"] = list(raw.shape); R["missing_cells"] = int(raw.isna().sum().sum()); R["duplicate_rows"] = int(raw.duplicated().sum())
raw = raw.drop_duplicates().reset_index(drop=True)
X_all = build_features(raw); y_all = np.log(raw["price_eur"].values)
R["rows_used"] = len(raw); R["cpu_gen_missing"] = int(X_all["cpu_gen"].isna().sum())
R["price_summary"] = raw["price_eur"].describe().round(1).to_dict()

# ---- 2. EDA figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(raw["price_eur"], bins=40, color=C1, ax=ax[0]); ax[0].set_title("Launch price (EUR)")
order = raw.groupby("Company")["price_eur"].median().sort_values(ascending=False).index
sns.boxplot(data=raw[raw["Company"].isin(raw["Company"].value_counts().head(8).index)], x="Company", y="price_eur", color="#9db7d1", ax=ax[1], fliersize=2,
            order=[c for c in order if c in raw["Company"].value_counts().head(8).index])
ax[1].set_title("Price by brand (8 most common)"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=30); save("real_price_dist")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.boxplot(data=raw, x="TypeName", y="price_eur", color="#9db7d1", ax=ax[0], fliersize=2); ax[0].set_title("Price by laptop type")
ax[0].tick_params(axis="x", rotation=30); ax[0].set_xlabel("")
sns.boxplot(data=raw, x="ram_gb", y="price_eur", color="#9db7d1", ax=ax[1], fliersize=2); ax[1].set_title("Price by RAM (GB)"); ax[1].set_xlabel("RAM (GB)"); save("real_type_ram")

num_df = X_all[NUM].assign(price_eur=raw["price_eur"].values)
plt.figure(figsize=(7.2, 5.6)); sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 6}, cbar_kws={"shrink": .8})
plt.title("Correlation of numeric features with price"); save("real_corr")
R["top_corr"] = num_df.corr()["price_eur"].drop("price_eur").sort_values(ascending=False).round(3).head(6).to_dict()

# ---- 3. split + rare GPU bucketing (fitted on training data only)
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=.2, random_state=SEED)
gcount = Xtr["gpu_model"].value_counts(); rare_gpu = set(gcount[gcount < 5].index)
for d in (Xtr, Xte): d["gpu_model"] = d["gpu_model"].where(~d["gpu_model"].isin(rare_gpu), "Other")
R["train_n"], R["test_n"], R["rare_gpu_models"] = len(Xtr), len(Xte), len(rare_gpu)

def make_pre():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    return ColumnTransformer([("num", num, NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

def metrics(model, X, y):
    p, t = np.exp(model.predict(X)), np.exp(y)
    return dict(R2_price=r2_score(t, p), R2_log=r2_score(y, model.predict(X)), MAE=mean_absolute_error(t, p),
                RMSE=mean_squared_error(t, p) ** .5, MAPE=float(np.mean(np.abs(t - p) / t) * 100))

# ---- 4. compare models
cands = {
    "Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=3.0), "Lasso": Lasso(alpha=.001),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=3, random_state=SEED),
    "Random Forest": RandomForestRegressor(400, min_samples_leaf=1, n_jobs=-1, random_state=SEED),
    "Extra Trees": ExtraTreesRegressor(400, n_jobs=-1, random_state=SEED),
    "Gradient Boosting (default)": GradientBoostingRegressor(n_estimators=300, learning_rate=.05, max_depth=3, random_state=SEED),
}
space = {"m__n_estimators": [300, 500, 800], "m__learning_rate": [.03, .05, .08], "m__max_depth": [2, 3, 4, 5],
         "m__subsample": [.7, .85, 1.0], "m__min_samples_leaf": [1, 3, 5]}
search = RandomizedSearchCV(Pipeline([("pre", make_pre()), ("m", GradientBoostingRegressor(random_state=SEED))]), space, n_iter=14, cv=3,
                            scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
best = {k.replace("m__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
R["gb_best_params"] = best
cands["Gradient Boosting (tuned)"] = GradientBoostingRegressor(random_state=SEED, **best)

kf = KFold(5, shuffle=True, random_state=SEED); rows = []; fitted = {}
for name, est in cands.items():
    pipe = Pipeline([("pre", make_pre()), ("m", est)])
    cv = cross_val_score(pipe, Xtr, ytr, cv=kf, scoring="r2"); pipe.fit(Xtr, ytr); fitted[name] = pipe
    rows.append(dict(Model=name, CV_R2_mean=cv.mean(), CV_R2_std=cv.std(), **metrics(pipe, Xte, yte)))
    print(f"{name:30s} CV R2={cv.mean():.4f}  test R2={rows[-1]['R2_price']:.4f}  MAE={rows[-1]['MAE']:.0f}")
comp = pd.DataFrame(rows).sort_values("CV_R2_mean", ascending=False).reset_index(drop=True)
R["model_table"] = comp.round(4).to_dict("records")
final_name = comp.loc[0, "Model"]; model = fitted[final_name]            # chosen on CV only
R["final_name"], R["final_metrics"] = final_name, metrics(model, Xte, yte)
R["final_cv_r2"] = [float(comp.loc[0, "CV_R2_mean"]), float(comp.loc[0, "CV_R2_std"])]
print("FINAL:", final_name, R["final_metrics"])

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
sns.barplot(data=comp, y="Model", x="CV_R2_mean", color=C1, ax=ax[0]); ax[0].set_xlim(.6, 1); ax[0].set_title("5-fold CV R\u00b2 (log price)"); ax[0].set_ylabel("")
sns.barplot(data=comp, y="Model", x="MAE", color=C2, ax=ax[1]); ax[1].set_title("Test MAE (EUR)"); ax[1].set_ylabel(""); ax[1].set_yticklabels([]); save("real_models")

# ---- 5. evaluation
pred, act = np.exp(model.predict(Xte)), np.exp(yte); res = (pred - act) / act * 100
plt.figure(figsize=(5.2, 5)); plt.scatter(act, pred, s=10, alpha=.45, color=C1); lim = [0, max(act.max(), pred.max()) * 1.02]
plt.plot(lim, lim, color=C2); plt.xlabel("Actual price (EUR)"); plt.ylabel("Predicted price (EUR)"); plt.title("Actual vs predicted (test set)"); save("real_actual_pred")
plt.figure(figsize=(6.2, 3.6)); sns.histplot(res, bins=40, color=C1); plt.axvline(0, color=C2); plt.xlabel("Percentage error (%)"); plt.title("Residual distribution (test set)"); save("real_residuals")
R["pct_err_within"] = {str(t): float(np.mean(np.abs(res) <= t) * 100) for t in (10, 15, 20)}; R["median_ape"] = float(np.median(np.abs(res)))
pi = permutation_importance(model, Xte, yte, n_repeats=8, random_state=SEED, scoring="r2", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False); R["importance"] = imp.round(4).to_dict()
plt.figure(figsize=(6.4, 4.4)); imp.head(12).iloc[::-1].plot(kind="barh", color=C1); plt.xlabel("Drop in R\u00b2 when shuffled"); plt.title("Permutation importance (top 12)"); save("real_importance")

# ---- 6. price-range margins (split-conformal, one margin per predicted-price band)
oof = cross_val_predict(model, Xtr, ytr, cv=kf); abs_res = np.abs(ytr - oof)
edges = [float(np.quantile(oof, .33)), float(np.quantile(oof, .66))]
band = lambda p: np.digitize(p, edges)
def margin_for(a):  # 80% quantile with finite-sample correction
    n = len(a); return float(np.quantile(a, min(1, np.ceil((n + 1) * .8) / n)))
q_band = [margin_for(abs_res[band(oof) == b]) for b in range(3)]
pt = model.predict(Xte); qv = np.array(q_band)[band(pt)]
R["margins_band"], R["band_edges_log"] = q_band, edges
R["coverage"] = float(np.mean((yte >= pt - qv) & (yte <= pt + qv)) * 100)
R["median_rel_width"] = float(np.median((np.exp(pt + qv) - np.exp(pt - qv)) / np.exp(pt)) * 100)
o = np.argsort(act)[::2]; xs = np.arange(len(o)); plt.figure(figsize=(8, 3.8))
plt.fill_between(xs, np.exp(pt - qv)[o], np.exp(pt + qv)[o], alpha=.25, color=C1, label="80% range"); plt.plot(xs, pred[o], color=C1, lw=1.2, label="Predicted")
plt.scatter(xs, act[o], s=6, color=C2, label="Actual", zorder=3); plt.xlabel("Test laptops sorted by actual price"); plt.ylabel("EUR"); plt.title("Predicted price range vs actual"); plt.legend(); save("real_interval")
print("coverage", R["coverage"], "width", R["median_rel_width"])

# ---- 7. save
bundle = dict(model=model, rare_gpu=rare_gpu, band_edges=edges, margins=q_band, final_name=final_name,
              ref=raw[["Company", "TypeName", "ram_gb"]].copy())
joblib.dump(bundle, ROOT / "models" / "base_price_model.joblib", compress=3)
json.dump(R, open(ROOT / "models" / "results_real.json", "w"), indent=1, default=float)
print("saved")
