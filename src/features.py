"""
Feature extraction for the Kaggle "Laptop Price" dataset (laptop specs -> price in EUR).
The same functions are used for training and inside the Streamlit app, so the app accepts the same kind of
values as the dataset (a CPU_Type string, a Memory string, a ScreenResolution string ...).
"""
import re
import numpy as np
import pandas as pd

RENAME = {"CPU_Frequency (GHz)": "cpu_ghz", "RAM (GB)": "ram_gb", "Weight (kg)": "weight_kg", "Price (Euro)": "price_eur"}
CAT = ["Company", "TypeName", "CPU_Company", "cpu_family", "GPU_Company", "gpu_model", "os_group", "panel"]
NUM = ["Inches", "cpu_ghz", "cpu_gen", "ram_gb", "ssd_gb", "hdd_gb", "flash_gb", "hybrid_gb",
       "res_w", "res_h", "ppi", "touch", "ips", "weight_kg", "gpu_dedicated"]

CPU_FAMILIES = ["Core i3", "Core i5", "Core i7", "Core M", "Celeron", "Pentium", "Atom", "Xeon", "Ryzen",
                "A4", "A6", "A8", "A9", "A10", "A12", "E-Series", "FX", "Cortex", "Exynos"]


def parse_resolution(s):
    s = str(s); low = s.lower()
    m = re.search(r"(\d{3,4})\s*x\s*(\d{3,4})", s)
    w, h = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
    panel = "Retina" if "retina" in low else ("IPS" if "ips" in low else "Standard")
    return dict(res_w=w, res_h=h, touch=int("touch" in low), ips=int("ips" in low), panel=panel)


def parse_memory(s):
    """'128GB SSD +  1TB HDD' -> ssd_gb=128, hdd_gb=1000 (TB counted as 1000 GB, as in the source)."""
    out = dict(ssd_gb=0.0, hdd_gb=0.0, flash_gb=0.0, hybrid_gb=0.0)
    for part in str(s).split("+"):
        m = re.search(r"([\d.]+)\s*(TB|GB)\s*(.*)", part.strip(), flags=re.I)
        if not m:
            continue
        gb = float(m.group(1)) * (1000 if m.group(2).upper() == "TB" else 1); kind = m.group(3).lower()
        key = "hybrid_gb" if "hybrid" in kind else "flash_gb" if "flash" in kind else "ssd_gb" if "ssd" in kind else "hdd_gb"
        out[key] += gb
    return out


def parse_cpu(cpu_type, cpu_company):
    """CPU family ('Core i5', 'Celeron' ...) and Intel generation from the model number (7200U -> 7, 10210U -> 10)."""
    t = str(cpu_type)
    fam = next((p for p in CPU_FAMILIES if p in t), "Other")
    gen = np.nan
    if str(cpu_company) == "Intel" and fam.startswith("Core i"):
        m = re.search(r"\b(\d{4,5})[A-Z]*\b", t)
        if m:
            d = m.group(1); gen = float(d[:2]) if len(d) == 5 else float(d[0])
    return fam, gen


def gpu_is_dedicated(company, gpu_type):
    c, t = str(company), str(gpu_type)
    if c == "Nvidia":
        return 1
    if c == "AMD":
        return int(bool(re.search(r"\bPro\b|\bRX\b|FirePro|R[5-9] M|Radeon 5\d\d|Vega M|\bM\d{3}", t)))
    return 0


def os_group(s):
    low = str(s).lower()
    return "macOS" if "mac" in low else "Windows" if "windows" in low else str(s)


def standardise(df):
    return df.rename(columns=RENAME)


def build_features(df, rare_gpu=None):
    """Raw Kaggle columns (original or renamed) -> model feature table. rare_gpu: GPU models bucketed as 'Other'."""
    df = standardise(df)
    rows = []
    for r in df.to_dict("records"):
        d = dict(Company=r["Company"], TypeName=r["TypeName"], CPU_Company=r["CPU_Company"], GPU_Company=r["GPU_Company"],
                 Inches=float(r["Inches"]), cpu_ghz=float(r["cpu_ghz"]), ram_gb=float(r["ram_gb"]), weight_kg=float(r["weight_kg"]))
        d["cpu_family"], d["cpu_gen"] = parse_cpu(r["CPU_Type"], r["CPU_Company"])
        d.update(parse_resolution(r["ScreenResolution"])); d.update(parse_memory(r["Memory"]))
        d["gpu_model"] = str(r["GPU_Type"]); d["gpu_dedicated"] = gpu_is_dedicated(r["GPU_Company"], r["GPU_Type"])
        d["os_group"] = os_group(r["OpSys"])
        d["ppi"] = float(np.hypot(d["res_w"], d["res_h"]) / d["Inches"])
        rows.append(d)
    X = pd.DataFrame(rows)
    if rare_gpu:
        X["gpu_model"] = X["gpu_model"].where(~X["gpu_model"].isin(rare_gpu), "Other")
    return X[CAT + NUM]
