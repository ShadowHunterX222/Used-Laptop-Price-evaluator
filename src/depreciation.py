"""
Stage 2 of the evaluator: turns the estimated LAUNCH price into a USED price.

!! These numbers are ASSUMPTIONS, not fitted to measured resale data. !!
They are the same rules used to generate the synthetic dataset in synthetic_experiment/ and are documented in
the project report (Section 4.3). Replace them with values learned from real scraped used-laptop listings
(OLX, Quikr, Cashify, Amazon Renewed, Flipkart 2GUD) when you have them.
"""
import math

# depreciation rate k per year (price retained = floor + (1-floor) * exp(-k * age))
K_BY_TYPE = {"Gaming": .30, "Notebook": .30, "Netbook": .30, "Ultrabook": .24, "2 in 1 Convertible": .24, "Workstation": .24}
K_APPLE, FLOOR, FLOOR_APPLE = .19, .10, .18

CONDITIONS = ["Mint", "Good", "Fair", "Poor"]
COND_MULT = {"Mint": 1.00, "Good": .93, "Fair": .82, "Poor": .66}

PLATFORMS = ["OLX", "Quikr", "Facebook Marketplace", "Cashify", "Amazon Renewed", "Flipkart 2GUD"]
PLATFORM_FACTOR = {"OLX": .93, "Quikr": .92, "Facebook Marketplace": .95, "Cashify": 1.04,
                   "Amazon Renewed": 1.09, "Flipkart 2GUD": 1.07}
# 80% margin (on log price) for how far used asking prices scatter around the model: peer-to-peer is noisier
PLATFORM_MARGIN = {"OLX": .212, "Quikr": .214, "Facebook Marketplace": .219, "Cashify": .166,
                   "Amazon Renewed": .161, "Flipkart 2GUD": .173}


def retained_fraction(age_years, company, type_name):
    apple = str(company) == "Apple"
    k = K_APPLE if apple else K_BY_TYPE.get(type_name, .28)
    floor = FLOOR_APPLE if apple else FLOOR
    return floor + (1 - floor) * math.exp(-k * max(0.0, age_years))


def adjustment_factor(condition, battery_pct, has_box, has_charger, warranty_months, platform, hdd_only, ram_gb):
    m = COND_MULT[condition]
    m *= .85 + .15 * (min(100, max(40, battery_pct)) - 50) / 50          # battery 100% -> 1.00, 50% -> 0.85
    m *= 1.025 if has_box else 1.0
    m *= 1.0 if has_charger else .94
    m *= 1 + min(.08, .006 * warranty_months)
    m *= .93 if hdd_only else 1.0
    m *= .92 if ram_gb <= 4 else 1.0
    m *= PLATFORM_FACTOR[platform]
    return m


def used_price(launch_price, age_years, company, type_name, condition, battery_pct, has_box, has_charger,
               warranty_months, platform, hdd_only, ram_gb):
    return (launch_price * retained_fraction(age_years, company, type_name) *
            adjustment_factor(condition, battery_pct, has_box, has_charger, warranty_months, platform, hdd_only, ram_gb))
