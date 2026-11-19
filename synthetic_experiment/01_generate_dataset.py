"""
01_generate_dataset.py
----------------------
Builds a SYNTHETIC but market-shaped dataset of used-laptop listings for the Indian
second-hand market (prices in INR). It exists because no public dataset combines
laptop specs + age + condition + resale price + platform (see report, Section 4).

Output: used_laptops_raw.csv  (deliberately "dirty", like freshly scraped data)

The generating rules (launch price -> depreciation -> condition/battery/accessory/
platform adjustments -> noise) are ASSUMPTIONS, documented in the report. Replace this
file's output with real scraped listings in the same column layout to retrain on real data.
"""
import numpy as np, pandas as pd

SEED = 2026
N = 3200
rng = np.random.default_rng(SEED)

START, END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-31")

# brand -> series -> (category, base launch price in INR for a reference i5/16GB/512GB config)
SERIES = {
    "Dell":   {"Inspiron": ("Everyday", 50000), "Vostro": ("Business", 50000), "Latitude": ("Business", 80000),
               "XPS": ("Premium", 110000), "G15": ("Gaming", 70000)},
    "HP":     {"Pavilion": ("Everyday", 55000), "15s": ("Everyday", 45000), "ProBook": ("Business", 65000),
               "EliteBook": ("Business", 95000), "Envy": ("Premium", 85000), "Victus": ("Gaming", 60000)},
    "Lenovo": {"IdeaPad": ("Everyday", 42000), "ThinkPad E": ("Business", 62000), "ThinkPad T": ("Business", 95000),
               "Yoga": ("Premium", 90000), "Legion": ("Gaming", 80000)},
    "Apple":  {"MacBook Air": ("Premium", 85000), "MacBook Pro": ("Premium", 130000)},
    "Asus":   {"VivoBook": ("Everyday", 44000), "ZenBook": ("Premium", 82000), "TUF Gaming": ("Gaming", 65000),
               "ROG Strix": ("Gaming", 100000)},
    "Acer":   {"Aspire": ("Everyday", 40000), "Swift": ("Premium", 68000), "Nitro": ("Gaming", 62000)},
    "MSI":    {"Modern": ("Business", 58000), "Thin": ("Gaming", 58000), "Katana": ("Gaming", 72000)},
}
BRANDS = list(SERIES)
BRAND_P = np.array([.22, .24, .22, .10, .10, .08, .04]); BRAND_P /= BRAND_P.sum()
CAT_W = {"Everyday": 2.0, "Business": 1.5, "Premium": 1.0, "Gaming": 1.0}

CPU_MULT = {"Celeron/Pentium": .75, "Core i3": .88, "Core i5": 1.0, "Core i7": 1.22, "Core i9": 1.5,
            "Ryzen 3": .88, "Ryzen 5": 1.0, "Ryzen 7": 1.2, "Ryzen 9": 1.45,
            "Apple M1": 1.0, "Apple M2": 1.15, "Apple M3": 1.3, "Apple M4": 1.45}
INTEL = ["Celeron/Pentium", "Core i3", "Core i5", "Core i7", "Core i9"]
INTEL_W = {"Everyday": [.12, .30, .38, .20, 0], "Business": [.03, .15, .50, .32, 0],
           "Premium": [0, .10, .50, .40, 0], "Gaming": [0, 0, .45, .45, .10]}
RYZEN = ["Ryzen 3", "Ryzen 5", "Ryzen 7", "Ryzen 9"]
RYZEN_W = [.20, .45, .30, .05]

PLATFORMS = ["OLX", "Quikr", "Facebook Marketplace", "Cashify", "Amazon Renewed", "Flipkart 2GUD"]
PLAT_P = [.26, .12, .10, .16, .18, .18]
PLAT_FACTOR = {"OLX": .93, "Quikr": .92, "Facebook Marketplace": .95, "Cashify": 1.04,
               "Amazon Renewed": 1.09, "Flipkart 2GUD": 1.07}
PLAT_SIGMA = {"OLX": .12, "Quikr": .13, "Facebook Marketplace": .11, "Cashify": .06,
              "Amazon Renewed": .06, "Flipkart 2GUD": .06}
CERTIFIED = {"Cashify", "Amazon Renewed", "Flipkart 2GUD"}

K = {"Everyday": .30, "Business": .24, "Premium": .24, "Gaming": .30}   # depreciation rate / year
COND = ["Mint", "Good", "Fair", "Poor"]
COND_MULT = {"Mint": 1.00, "Good": .93, "Fair": .82, "Poor": .66}
COND_BASE = np.array([.15, .45, .28, .12]); COND_SLOPE = np.array([-.15, 0, .08, .20])
GPU_ADD = {"Integrated": 0, "Entry": 14000, "Mid": 30000, "High": 58000}
RES_ADD = {"HD": -4000, "FHD": 0, "QHD+": 9000}


def pick_cpu(brand, year, cat):
    if brand == "Apple":
        if year <= 2019: return rng.choice(["Core i5", "Core i7"], p=[.65, .35])
        if year == 2020: return rng.choice(["Core i5", "Core i7", "Apple M1"], p=[.2, .1, .7])
        if year == 2021: return "Apple M1"
        if year == 2022: return rng.choice(["Apple M1", "Apple M2"], p=[.4, .6])
        if year == 2023: return rng.choice(["Apple M2", "Apple M3"], p=[.5, .5])
        return rng.choice(["Apple M3", "Apple M4"], p=[.6, .4])
    ryzen_share = 0 if year < 2019 else (.15 if year == 2019 else (.25 if year == 2020 else .30))
    if rng.random() < ryzen_share:
        return rng.choice(RYZEN, p=RYZEN_W)
    w = np.array(INTEL_W[cat], float)
    if year < 2019: w[4] = 0
    return rng.choice(INTEL, p=w / w.sum())


def make_row(i):
    date = START + pd.Timedelta(days=int(rng.integers(0, (END - START).days + 1)))
    age = float(np.clip(rng.gamma(2.6, 1.3), 0.3, 8.8))
    release_year = (date - pd.Timedelta(days=age * 365.25)).year
    brand = rng.choice(BRANDS, p=BRAND_P)
    names = list(SERIES[brand]); w = np.array([CAT_W[SERIES[brand][s][0]] for s in names]); w /= w.sum()
    series = rng.choice(names, p=w)
    cat, base = SERIES[brand][series]
    cpu = pick_cpu(brand, release_year, cat)
    apple = brand == "Apple"

    # ---- specs ----
    y = release_year
    ram_w = [.20, .50, .28, .02] if y <= 2018 else ([.07, .50, .40, .03] if y <= 2021 else [.01, .30, .55, .14])
    ram_w = np.array(ram_w, float)
    if cat == "Gaming" or apple: ram_w[0] = 0
    ram = int(rng.choice([4, 8, 16, 32], p=ram_w / ram_w.sum()))

    if apple: stype = "NVMe SSD"
    else:
        sp = [.5, .35, .15] if y <= 2018 else ([.3, .3, .4] if y <= 2020 else [.04, .10, .86])
        stype = rng.choice(["HDD", "SATA SSD", "NVMe SSD"], p=sp)
    if stype == "HDD": gb = int(rng.choice([500, 1000]))
    else:
        sw = [.2, .45, .3, .05] if y <= 2019 else [.02, .35, .5, .13]
        if cat == "Gaming": sw = [0, .1, .5, .4]
        gb = int(rng.choice([128, 256, 512, 1024], p=sw))

    if cat == "Gaming":
        gpu = rng.choice(["Entry", "Mid", "High"], p=[.45, .40, .15] if series not in ("ROG Strix", "Legion", "Katana") else [.2, .45, .35])
    elif apple: gpu = "Integrated"
    else: gpu = rng.choice(["Integrated", "Entry"], p=[.88, .12])

    if apple: size = 13.3 if series == "MacBook Air" else float(rng.choice([13.3, 14, 16]))
    elif cat == "Gaming": size = float(rng.choice([15.6, 16, 17.3], p=[.7, .2, .1]))
    elif cat == "Premium": size = float(rng.choice([13.3, 14, 15.6, 16], p=[.3, .4, .2, .1]))
    else: size = float(rng.choice([14, 15.6], p=[.3, .7]))
    if apple: res = "QHD+"
    else:
        rp = [.5, .5, 0] if (y <= 2020 and cat == "Everyday") else ([.15, .8, .05] if y <= 2021 else [.05, .75, .20])
        if cat == "Gaming": rp = [0, .8, .2]
        res = rng.choice(["HD", "FHD", "QHD+"], p=rp)

    # ---- launch (original) price ----
    infl = 1 + .025 * (release_year - 2020)
    ram_add = (ram - 16) * (2200 if apple else 900)
    if stype == "HDD": st_add = -9000 + (gb - 500) * 3
    elif stype == "SATA SSD": st_add = (gb - 512) * 22
    else: st_add = (gb - 512) * (50 if apple else 28)
    size_add = 1500 * (size - 15.6)
    launch = (base * CPU_MULT[cpu] * infl + ram_add + st_add + GPU_ADD[gpu] + RES_ADD[res] + size_add)
    launch = max(22000, launch * rng.lognormal(0, .04))
    launch = round(launch / 100) * 100

    # ---- condition, battery, accessories, warranty ----
    platform = rng.choice(PLATFORMS, p=PLAT_P)
    cw = COND_BASE * np.exp(COND_SLOPE * age)
    if platform in CERTIFIED: cw[3] = 0
    cond = rng.choice(COND, p=cw / cw.sum())
    battery = int(np.clip(round(100 - 5.5 * age + rng.normal(0, 6) - (8 if cond == "Poor" else 0)), 40, 100))
    has_box = int(rng.random() < (.35 if age < 2 else .20))
    has_charger = int(rng.random() < .93)
    warranty = 0
    if age < 3 and rng.random() < .45: warranty = int(rng.integers(1, 19))
    elif age < 4 and rng.random() < .08: warranty = int(rng.integers(1, 7))

    # ---- depreciation -> listed price ----
    k = (.19 if apple else K[cat]) * rng.lognormal(0, .10)
    floor = .18 if apple else .10
    retained = floor + (1 - floor) * np.exp(-k * age)
    m = COND_MULT[cond]
    m *= .85 + .15 * (battery - 50) / 50
    m *= 1.025 if has_box else 1.0
    m *= 1.0 if has_charger else .94
    m *= 1 + min(.08, .006 * warranty)
    m *= .93 if stype == "HDD" else 1.0
    m *= .92 if ram == 4 else 1.0
    m *= PLAT_FACTOR[platform]
    month_idx = (date.year - 2025) * 12 + date.month - 11
    m *= 1 - .004 * month_idx                                  # slow market softening
    price = launch * retained * m * rng.lognormal(0, PLAT_SIGMA[platform])
    price = max(5000, round(price / 500) * 500)

    return dict(listing_id=f"LP-{i+1:05d}", listing_date=date.strftime("%Y-%m-%d"), platform=platform,
                brand=brand, series=series, category=cat, release_year=release_year, cpu=cpu,
                ram=f"{ram} GB", storage=f"{gb} GB {stype}", gpu_class=gpu, screen_inches=size,
                resolution=res, age_years=round(age, 1), battery_health_pct=battery, condition=cond,
                has_box="Yes" if has_box else "No", has_charger="Yes" if has_charger else "No",
                warranty_months_left=warranty, original_price_inr=int(launch), listed_price_inr=int(price))


df = pd.DataFrame([make_row(i) for i in range(N)])

# ------------------------------------------------------------------ inject scraping "dirt"
def sample_idx(frac): return rng.choice(df.index, size=int(len(df) * frac), replace=False)

# 1. inconsistent text (case / stray spaces)
for col, frac in [("brand", .04), ("platform", .04), ("condition", .04)]:
    idx = sample_idx(frac)
    df.loc[idx, col] = [rng.choice([str(v).upper(), str(v).lower(), f" {v} ", f"{v} "]) for v in df.loc[idx, col]]
# 2. RAM written two ways
idx = sample_idx(.05); df.loc[idx, "ram"] = df.loc[idx, "ram"].str.replace(" GB", "GB")
# 3. price scraped as text with currency symbol and commas
df["listed_price_inr"] = df["listed_price_inr"].map(lambda v: f"\u20b9{v:,}")
# 4. typo outliers (extra / missing zero) - applied to the numeric price before text formatting would be
#    unrealistic, so we edit the text directly
idx = sample_idx(.006)
df.loc[idx, "listed_price_inr"] = [f"\u20b9{int(v.replace('\u20b9','').replace(',',''))*10:,}" if rng.random() < .5
                                   else f"\u20b9{max(500,int(v.replace('\u20b9','').replace(',',''))//10):,}"
                                   for v in df.loc[idx, "listed_price_inr"]]
# 5. missing values
for col, frac in [("battery_health_pct", .06), ("screen_inches", .02), ("condition", .015),
                  ("has_charger", .03), ("resolution", .01)]:
    df.loc[sample_idx(frac), col] = np.nan
# 6. structural blank: "no warranty" is scraped as an empty cell
df.loc[df["warranty_months_left"] == 0, "warranty_months_left"] = np.nan
# 7. duplicate postings (cross-posted / re-scraped)
df = pd.concat([df, df.sample(45, random_state=SEED)], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

df.to_csv("used_laptops_raw.csv", index=False, encoding="utf-8-sig")
print("raw rows:", len(df)); print(df.head(3).T)
