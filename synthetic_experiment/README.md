# Synthetic experiment (Part 1 of the report)

Generates a synthetic used-laptop dataset and runs the full cleaning / modelling pipeline on it.
Run from this folder, in order:

    python 01_generate_dataset.py      # writes used_laptops_raw.csv
    python 02_laptop_price_model.py    # cleans, trains, evaluates; writes figs/, results.json, model
    python 03_evaluate_laptop.py       # prices one example laptop

The generating rules are assumptions, not measured market data (see report, Section 4).
