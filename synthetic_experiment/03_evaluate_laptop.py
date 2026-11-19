"""
03_evaluate_laptop.py
---------------------
Loads laptop_price_evaluator.joblib and prices ONE used laptop.
Edit the SPEC dictionary below (17 inputs) and run:  python 03_evaluate_laptop.py
This is the function a Flask/Django endpoint would call in the web app described in the synopsis.
"""
import joblib, numpy as np, pandas as pd

B = joblib.load("laptop_price_evaluator.joblib")
CAT, NUM, ref, q_by_plat, point = B["CAT"], B["NUM"], B["ref"], B["q_by_plat"], B["point"]

SPEC = dict(platform="OLX", brand="Dell", series="Latitude", category="Business", cpu="Core i5",
            storage_type="NVMe SSD", gpu_class="Integrated", resolution="FHD", condition="Good",
            age_years=3.0, ram_gb=16, storage_gb=512, screen_inches=14.0, battery_health_pct=80,
            warranty_months_left=0, has_box=0, has_charger=1)

def _price(spec):
    return float(np.exp(point.predict(pd.DataFrame([spec])[CAT + NUM]))[0])

def n_comparables(brand, series, age):
    return int(((ref["brand"] == brand) & (ref["series"] == series) & ((ref["age_years"] - age).abs() <= 1.0)).sum())

def confidence(n_comp, rel_width):
    return round(100 * (.5 * min(1.0, n_comp / 25) + .5 * float(np.clip(1 - rel_width, 0, 1))), 1)

def evaluate(spec):
    m = float(point.predict(pd.DataFrame([spec])[CAT + NUM])[0]); q = q_by_plat[spec["platform"]]
    med, lo, hi = np.exp(m), np.exp(m - q), np.exp(m + q)
    n = n_comparables(spec["brand"], spec["series"], spec["age_years"]); c = confidence(n, (hi - lo) / med)
    traj = {}
    for d in (1, 2, 3):
        s = dict(spec); s["age_years"] += d
        s["battery_health_pct"] = max(40, s["battery_health_pct"] - 5.5 * d)
        s["warranty_months_left"] = max(0, s["warranty_months_left"] - 12 * d)
        traj[d] = _price(s)
    return dict(minimum=lo, median=med, maximum=hi, confidence=c, comparable_listings=n,
                market_supply="High" if n >= 40 else "Medium" if n >= 15 else "Low", trajectory=traj,
                by_platform={p: _price({**spec, "platform": p}) for p in B["platforms"]})

if __name__ == "__main__":
    r = evaluate(SPEC)
    print(f"Fair price range : Rs {r['minimum']:,.0f}  -  Rs {r['maximum']:,.0f}   (median Rs {r['median']:,.0f})")
    print(f"Confidence score : {r['confidence']}/100  ({r['comparable_listings']} comparable listings, supply: {r['market_supply']})")
    print("Value in +1/+2/+3 yrs:", {k: f"Rs {v:,.0f}" for k, v in r["trajectory"].items()})
    print("By platform      :", {k: f"Rs {v:,.0f}" for k, v in sorted(r["by_platform"].items(), key=lambda kv: kv[1])})
