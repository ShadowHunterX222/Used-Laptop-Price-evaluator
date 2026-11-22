# Used Laptop Price Evaluator

Estimates a fair second-hand price range for a laptop from its specifications, age, condition and the platform it is sold on,
and serves the result in a Streamlit web app.

**Authors:** Kousthub S Kumar (NN25ISE114) and Neeraj Kumar (NN25ISE133)  
3rd semester, Department of Information Science and Engineering

## What it does

Enter a laptop's brand, type, screen, processor, RAM, graphics, storage and operating system, then its age, condition, battery health,
warranty, box/charger and the platform (OLX, Quikr, Facebook Marketplace, Cashify, Amazon Renewed, Flipkart 2GUD). The app shows:

- a fair price range in rupees (minimum, median, maximum) with a confidence label and score
- the estimated launch price and the share of value retained at that age
- a three-year depreciation trajectory and the predicted price on each platform

## How it works (two stages)

| Stage | What | Data | Status |
|---|---|---|---|
| 1 | specifications -> **launch price** (EUR, converted to INR) | **Real**: Kaggle *Laptop Price* dataset, 1,275 laptops, mostly 2016-2018 | Learned and evaluated |
| 2 | launch price -> **used price** (age, condition, battery, accessories, platform) | Documented **assumptions** (`src/depreciation.py`) | Not yet validated on real resale data |

Stage 1 model: tuned gradient boosting (selected by 5-fold cross-validation among 8 models).
Held-out test results: R2 = 0.888, mean absolute error = EUR 162, median absolute error = 10.8%,
80% range coverage = 80.8%.

The price range combines the stage-1 error (split-conformal, from real data) with platform price scatter (from the synthetic experiment) in
quadrature. The confidence score is 50% sample density + 50% range tightness; it is a heuristic, not a calibrated probability.

## Quick start

```bash
git clone <your-repo-url>
cd used-laptop-price-evaluator
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The saved model (`models/base_price_model.joblib`) is loaded on start. If it is missing or was saved with a different scikit-learn
version, the app retrains it automatically from the CSV (about a minute).

To retrain and regenerate the figures and metrics yourself:

```bash
python -m src.train_base_model        # run from the repository root
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io, choose **New app**, select the repository, branch `main` and main file `app.py`.
3. Deploy. The first start may take a minute while the model loads or retrains.

## Repository layout

```
app.py                          Streamlit app
requirements.txt
data/laptop_price_dataset.csv   real Kaggle data (prices in EUR)
data/synthetic/                 synthetic used-laptop listings (Part 1 of the report)
models/                         trained stage-1 model + metrics (results_real.json)
src/features.py                 text -> numeric features (resolution, memory, CPU, GPU, OS)
src/train_base_model.py         training, model comparison, evaluation, figures
src/depreciation.py             stage-2 depreciation assumptions
src/evaluator.py                two-stage evaluator used by the app
synthetic_experiment/           scripts that generate and model the synthetic dataset
figs/                           charts used in the report
docs/                           full project report (PDF)
```

## Limitations (please read)

- **Stage 1 knows 2016-2018 hardware** and euro prices of that period. Newer processors and graphics cards are outside the training range.
  For those laptops, enter the known launch price in the app's override box.
- **Stage 2 is assumption-based.** Depreciation rates, condition and platform factors have not been fitted to real used-laptop listings.
  Collect a few hundred real listings (OLX, Quikr, Cashify, Amazon Renewed, ...) and compare them with the app's output before trusting it.
  Check each site's terms of use and robots.txt before scraping.
- The dataset has no age or condition information, so depreciation cannot be learned from it.
- The EUR to INR rate is a user setting (default 105); it changes daily.
- Results are estimates for learning and demonstration, not guaranteed market prices.

## Testing status

The evaluator and feature code were run end to end, and `app.py` was exercised with a stand-in for the Streamlit API. The app has not yet been
opened in a real Streamlit session by the authors of this repository. If a widget behaves differently in your Streamlit version, check
`app.py` first (it uses only common widgets: selectbox, number_input, slider, radio, checkbox, metric, line_chart, bar_chart, dataframe).

## Data and credits

- Report and methodology: see `docs/Used_Laptop_Price_Evaluator_Report.pdf`.

## Licence

Code: MIT (see `LICENSE`). The dataset keeps its own licence.
