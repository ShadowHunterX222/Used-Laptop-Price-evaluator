"""
Two-stage used-laptop price evaluator.

  Stage 1 (learned from REAL data): specs -> launch price in EUR   (Kaggle "Laptop Price" dataset, ~2016-2018 laptops)
  Stage 2 (documented ASSUMPTIONS):  launch price -> used price      (age, condition, battery, accessories, platform)

The price range combines both uncertainties; see README for the formulas.
"""
import math, runpy
from pathlib import Path
import numpy as np, pandas as pd, joblib
from src.features import build_features
from src import depreciation as dep

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "base_price_model.joblib"
DATA_PATH = ROOT / "data" / "laptop_price_dataset.csv"


def load_bundle():
    """Load the saved model; if it is missing or was saved with an incompatible scikit-learn, retrain it (takes ~1 minute)."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        runpy.run_module("src.train_base_model", run_name="__retrain__")
        return joblib.load(MODEL_PATH)


class Evaluator:
    def __init__(self, bundle=None):
        self.b = bundle or load_bundle()
        self.model, self.ref = self.b["model"], self.b["ref"]

    # ---------- stage 1
    def launch_price_eur(self, spec: dict):
        """spec uses the Kaggle column names (Company, TypeName, Inches, ScreenResolution, CPU_Company, CPU_Type,
        CPU_Frequency (GHz), RAM (GB), Memory, GPU_Company, GPU_Type, OpSys, Weight (kg)). Returns (price, margin_log)."""
        X = build_features(pd.DataFrame([spec]), rare_gpu=self.b["rare_gpu"])
        p = float(self.model.predict(X)[0])
        band = int(np.digitize(p, self.b["band_edges"]))
        return math.exp(p), float(self.b["margins"][band])

    def comparables(self, spec):
        r = self.ref
        return int(((r["Company"] == spec["Company"]) & (r["TypeName"] == spec["TypeName"])).sum())

    # ---------- full evaluation
    def evaluate(self, spec, used, eur_to_inr=105.0, launch_override_inr=None):
        """used: dict(age_years, condition, battery_pct, has_box, has_charger, warranty_months, platform)."""
        if launch_override_inr:
            launch_inr, q1 = float(launch_override_inr), 0.0
        else:
            eur, q1 = self.launch_price_eur(spec); launch_inr = eur * eur_to_inr
        hdd_only = "SSD" not in str(spec["Memory"]).upper() and "FLASH" not in str(spec["Memory"]).upper()
        ram = float(spec["RAM (GB)"])

        def price(u, plat=None):
            return dep.used_price(launch_inr, u["age_years"], spec["Company"], spec["TypeName"], u["condition"], u["battery_pct"],
                                  u["has_box"], u["has_charger"], u["warranty_months"], plat or u["platform"], hdd_only, ram)

        median = price(used)
        q = math.sqrt(q1 ** 2 + dep.PLATFORM_MARGIN[used["platform"]] ** 2)      # combine the two sources of error
        lo, hi = median * math.exp(-q), median * math.exp(q)
        n = self.comparables(spec)
        conf = round(100 * (.5 * min(1.0, n / 25) + .5 * min(1.0, max(0.0, 1 - (hi - lo) / median))), 1)
        traj = {}
        for d in (0, 1, 2, 3):
            u = dict(used, age_years=used["age_years"] + d, battery_pct=max(40, used["battery_pct"] - 5.5 * d),
                     warranty_months=max(0, used["warranty_months"] - 12 * d))
            traj[d] = price(u)
        plats = {p: price(used, p) for p in dep.PLATFORMS}
        return dict(launch_inr=launch_inr, retained=dep.retained_fraction(used["age_years"], spec["Company"], spec["TypeName"]),
                    low=lo, median=median, high=hi, margin_stage1=q1, margin_total=q, confidence=conf,
                    confidence_label="High" if conf >= 75 else "Medium" if conf >= 55 else "Low",
                    comparables=n, supply="High" if n >= 100 else "Medium" if n >= 30 else "Low",
                    trajectory=traj, platforms=plats, hdd_only=hdd_only)
