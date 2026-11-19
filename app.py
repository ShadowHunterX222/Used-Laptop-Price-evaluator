"""
Used Laptop Price Evaluator - Streamlit app
Run:  streamlit run app.py
Authors: Kousthub S Kumar (NN25ISE114), Neeraj Kumar (NN25ISE133) - 3rd semester, Dept. of Information Science and Engineering
"""
import pandas as pd
import streamlit as st

from src.evaluator import Evaluator, DATA_PATH
from src import depreciation as dep

st.set_page_config(page_title="Used Laptop Price Evaluator", page_icon="\U0001F4BB", layout="wide")


# ---------------------------------------------------------------- helpers
def inr(x):
    """Indian digit grouping: 168498 -> \u20b91,68,498"""
    s = str(int(round(x)))
    if len(s) > 3:
        head, tail, parts = s[:-3], s[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head: parts.insert(0, head)
        s = ",".join(parts + [tail])
    return "\u20b9" + s


@st.cache_resource(show_spinner="Loading model (first run may retrain it, ~1 minute)...")
def get_evaluator():
    return Evaluator()


@st.cache_data
def get_options():
    df = pd.read_csv(DATA_PATH)
    by_count = lambda col: df[col].value_counts().index.tolist()
    cpu = df.groupby("CPU_Type").agg(company=("CPU_Company", "first"), ghz=("CPU_Frequency (GHz)", "median"), n=("CPU_Type", "size")).sort_values("n", ascending=False)
    gpu = {c: g["GPU_Type"].value_counts().index.tolist() for c, g in df.groupby("GPU_Company")}
    weight = df.groupby("TypeName")["Weight (kg)"].median().to_dict()
    return dict(company=by_count("Company"), type=by_count("TypeName"), resolution=by_count("ScreenResolution"), os=by_count("OpSys"),
                cpu=cpu, gpu=gpu, weight=weight, rams=sorted(df["RAM (GB)"].unique().tolist()))


ev, opt = get_evaluator(), get_options()

# ---------------------------------------------------------------- header + sidebar
st.title("Used Laptop Price Evaluator")
st.caption("Kousthub S Kumar (NN25ISE114) \u00b7 Neeraj Kumar (NN25ISE133) \u00b7 3rd semester, Department of Information Science and Engineering")

with st.sidebar:
    st.header("Settings")
    rate = st.number_input("EUR \u2192 INR rate", min_value=50.0, max_value=200.0, value=105.0, step=0.5,
                           help="The training data is priced in euros. Update this to the current rate (it was about 105 in Nov 2026).")
    st.divider()
    st.subheader("How to read this")
    st.markdown(
        "**Stage 1 (real data):** a model trained on the Kaggle *Laptop Price* dataset (1,275 laptops, mostly 2016\u20132018) "
        "estimates the laptop's **launch price** from its specs.\n\n"
        "**Stage 2 (assumptions):** age, condition, battery, accessories and platform turn that into a **used price**. "
        "These depreciation rules are documented assumptions, **not** fitted to real resale data yet.")
    st.warning("Treat results as an estimate for learning and demonstration, not a guaranteed market price.")

# ---------------------------------------------------------------- inputs
st.subheader("1. Laptop specifications")
c1, c2, c3 = st.columns(3)
with c1:
    company = st.selectbox("Brand", opt["company"])
    type_name = st.selectbox("Type", opt["type"])
    inches = st.number_input("Screen size (inches)", 10.0, 18.4, 15.6, 0.1)
    resolution = st.selectbox("Screen resolution / panel", opt["resolution"])
    os_name = st.selectbox("Operating system", opt["os"])
with c2:
    cpu_type = st.selectbox("Processor", opt["cpu"].index.tolist())
    cpu_company = opt["cpu"].loc[cpu_type, "company"]
    ghz = st.number_input("CPU speed (GHz)", 0.9, 3.6, float(opt["cpu"].loc[cpu_type, "ghz"]), 0.1, key=f"ghz_{cpu_type}")
    ram = st.selectbox("RAM (GB)", opt["rams"], index=opt["rams"].index(8) if 8 in opt["rams"] else 0)
    gpu_company = st.selectbox("Graphics brand", list(opt["gpu"]), index=list(opt["gpu"]).index("Intel") if "Intel" in opt["gpu"] else 0)
    gpu_type = st.selectbox("Graphics model", opt["gpu"][gpu_company])
with c3:
    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000, 2000], index=2)
    hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000], index=0)
    flash = st.selectbox("Flash storage (GB)", [0, 16, 32, 64, 128], index=0)
    weight = st.number_input("Weight (kg)", 0.6, 5.0, float(opt["weight"].get(type_name, 2.0)), 0.05, key=f"w_{type_name}")

parts = [f"{ssd}GB SSD" if ssd else "", f"{hdd}GB HDD" if hdd else "", f"{flash}GB Flash Storage" if flash else ""]
memory = " + ".join(p for p in parts if p)

st.subheader("2. Condition and listing")
d1, d2, d3 = st.columns(3)
with d1:
    age = st.slider("Age (years since purchase / release)", 0.5, 9.0, 3.0, 0.5)
    condition = st.radio("Condition", dep.CONDITIONS, index=1, horizontal=True, help="Mint = like new \u00b7 Good = light wear \u00b7 Fair = visible wear \u00b7 Poor = heavy wear")
with d2:
    battery = st.slider("Battery health (%)", 40, 100, 80)
    warranty = st.slider("Warranty left (months)", 0, 24, 0)
with d3:
    platform = st.selectbox("Where will it be sold / bought?", dep.PLATFORMS)
    has_box = st.checkbox("Original box included")
    has_charger = st.checkbox("Original charger included", value=True)

with st.expander("Optional: I already know the launch price"):
    override = st.number_input("Original launch price (\u20b9). Leave at 0 to estimate it from specs.", 0, 1000000, 0, 1000,
                               help="Useful for laptops newer than the training data (2016\u20132018 hardware).")

if not memory:
    st.error("Choose at least one storage type (SSD, HDD or flash).")
    st.stop()

spec = {"Company": company, "TypeName": type_name, "Inches": inches, "ScreenResolution": resolution, "CPU_Company": cpu_company,
        "CPU_Type": cpu_type, "CPU_Frequency (GHz)": ghz, "RAM (GB)": ram, "Memory": memory, "GPU_Company": gpu_company,
        "GPU_Type": gpu_type, "OpSys": os_name, "Weight (kg)": weight}
used = dict(age_years=age, condition=condition, battery_pct=battery, has_box=has_box, has_charger=has_charger,
            warranty_months=warranty, platform=platform)

res = ev.evaluate(spec, used, eur_to_inr=rate, launch_override_inr=override or None)

# ---------------------------------------------------------------- results
st.divider()
st.subheader("Estimated fair price")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Minimum", inr(res["low"]))
m2.metric("Fair price (median)", inr(res["median"]))
m3.metric("Maximum", inr(res["high"]))
m4.metric("Confidence", f"{res['confidence_label']} ({res['confidence']:.0f}/100)")
st.caption(f"The range is an approximate 80% interval. Based on {res['comparables']} training laptops of the same brand and type "
           f"(supply: {res['supply']}).")

k1, k2, k3 = st.columns(3)
k1.metric("Estimated launch price" + (" (you entered)" if override else ""), inr(res["launch_inr"]))
k2.metric("Value retained at this age", f"{res['retained'] * 100:.0f}%")
k3.metric("Range width (\u00b1 on price)", f"\u00b1{(res['high'] - res['low']) / 2 / res['median'] * 100:.0f}%")

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Depreciation trajectory** (same laptop, getting older)")
    traj = pd.DataFrame({"Predicted price (\u20b9)": list(res["trajectory"].values())},
                        index=["Today", "+1 year", "+2 years", "+3 years"])
    st.line_chart(traj)
with g2:
    st.markdown("**Same laptop, price by platform**")
    plat = pd.Series(res["platforms"], name="Predicted price (\u20b9)").sort_values()
    st.bar_chart(plat)

tbl = pd.DataFrame({"Platform": list(res["platforms"]), "Predicted price": [inr(v) for v in res["platforms"].values()]})
with st.expander("Price by platform (table)"):
    st.dataframe(tbl, hide_index=True)

with st.expander("How this estimate was computed"):
    st.markdown(f"""
1. **Launch price** \u2192 {'you entered it' if override else 'a gradient-boosting model trained on the Kaggle Laptop Price dataset predicts it from the specs, in euros, then converts at the rate in the sidebar'}.
2. **Used price** = launch price \u00d7 depreciation \u00d7 adjustments. Depreciation = `floor + (1 - floor) \u00d7 exp(-k \u00d7 age)`; adjustments cover condition,
   battery health, box, charger, warranty, HDD-only storage, 4 GB RAM and platform.
3. **Range**: the two error sources (spec model error \u00b1{res['margin_stage1']:.2f} and platform price scatter) are combined in quadrature on log price,
   giving \u00b1{res['margin_total']:.2f}.
4. **Confidence** = 50% sample density (brand+type laptops in training data, capped at 25) + 50% range tightness.

**Limits:** the training laptops are mostly 2016\u20132018 models priced in euros; newer CPUs/GPUs are outside the training range, so enter the launch
price yourself for those. Stage-2 rates are assumptions until real used-laptop listings are collected.
""")
