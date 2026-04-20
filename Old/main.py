"""
EXPLORATORY DATA ANALYSIS (EDA)

Transit Anomaly Detection — Airport & Border Control Data 
Datasets:  
            ALLARMI.csv  (Alarms) 
            TIPOLOGIA_VIAGGIATORE.csv  (Travelers)  
Authors: Giulio D'Amico · Alexis Mitracos

TABLE OF CONTENTS
─────────────────
0   Setup & Data Loading
1   Data Quality Audit
  1.1  Raw Dataset Inspection (shape, dtypes, cardinality, first missing-value snapshot)
  1.2  Categorical Cardinality Inspection
  1.3  Redundant-Column Resolution (Travelers)
  1.4  Text Standardisation & Unknown-Value Unification (Travelers)
  1.5  Numeric & Domain-Specific Standardisation (Travelers)
  1.6  Date Parsing & Temporal Alignment (both datasets)
  1.7  Alarms Dataset — Full Cleaning Pipeline
  1.8  Post-Cleaning Missing-Value Audit (before/after comparison)
  1.9  Redundant-Column Drop & Final Schema Verification

2   Univariate Analysis
  2.1  Numerical Distributions (histograms + descriptive stats)
  2.2  Categorical Frequency Analysis (bar charts)
  2.3  Temporal Distribution (departure_date by month)

3   Bivariate Analysis
  3.1  Entries vs Alarms (scatter + correlation)
  3.2  Alarm Reason × Zone (heatmap)
  3.3  Top Routes by Volume (departure → arrival)

4   Multivariate Analysis
  4.1  Correlation Matrix — Numerical Features
  4.2  Pair Plot — Key Numerical Features (Travelers)

5   Outlier Analysis
  5.1  IQR-Based Outlier Detection (numerical columns)
  5.2  Box Plots — Key Metrics

6   Cross-Dataset Coverage
  6.1  Temporal Overlap (Alarms vs Travelers)
  6.2  Airport & Route Coverage
  6.3  Final Summary & Observations
"""
# ═══════════════════════════════════════════════════════════════════
# 0  SETUP & DATA LOADING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("§0 · SETUP & DATA LOADING")
print("=" * 70)

import warnings
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import pycountry
try:
    from IPython.display import display
except ImportError:
    display = print
import missingno as msno

from Old.config import (
    RAW_FILES, SNAPSHOT_DATE,
    NAVY, STEEL, CORAL, GREEN, PALETTE,
    manual_mapping, it_to_en, city_mapping, iata_mapping
)
# Plot defaults
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05, palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:.2f}".format)

SAVE_FIGS = False # Set to True to save figures to io/ folder

def savefig(name, tight=True):
    """Save current figure to io/{name}.png and display it."""
    if tight:
        plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"io/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# Load raw CSVs and apply the Italian→English column mapping
df_alarms   = pd.read_csv(RAW_FILES["Alarms"], sep=",")
df_travelers = pd.read_csv(RAW_FILES["Traveler_type"], sep=",")
df_alarms   = df_alarms.rename(columns=manual_mapping)
df_travelers = df_travelers.rename(columns=manual_mapping)

print("Setup complete — libraries loaded, data imported, columns renamed.")

# ═══════════════════════════════════════════════════════════════════
# 1  DATA QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1 · DATA QUALITY AUDIT")
print("=" * 70)
# ───────────────────────────────────────────────────────────────────
# 1.1  Raw Dataset Inspection
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.1 · Raw Dataset Inspection")
print("-" * 70)

def inspect_dataset(df, dataset_name):
    """Print shape, dtype summary, missing-value counts and cardinality."""
    print(f"\n{'=' * 60}")
    print(f"  {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    overview = pd.DataFrame({
        "Dtype":       df.dtypes,
        "Missing":     df.isnull().sum(),
        "Missing %":   (df.isnull().sum() / len(df) * 100).round(2),
        "Cardinality": df.nunique(),
    })
    print("Column-level overview:")
    display(overview)

inspect_dataset(df_alarms,   "ALARMS (ALLARMI)")
inspect_dataset(df_travelers, "TRAVELERS (TIPOLOGIA_VIAGGIATORE)")

# Missingness bar-charts (visual snapshot BEFORE any cleaning)
print("\n1.1 — Missingness bar chart (BEFORE cleaning)")

print("  ▸ Alarms dataset")
msno.bar(df_alarms, figsize=(10, 4), color=STEEL)
savefig("missing_bar_alarms_raw")

print("  ▸ Travelers dataset")
msno.bar(df_travelers, figsize=(10, 4), color=STEEL)
savefig("missing_bar_travelers_raw")

# ── SNAPSHOT: store raw missing counts for the before/after comparison ──
raw_missing_alarms   = df_alarms.isnull().sum().copy()
raw_missing_travelers = df_travelers.isnull().sum().copy()
raw_shape_alarms     = df_alarms.shape
raw_shape_travelers  = df_travelers.shape
raw_cols_alarms      = list(df_alarms.columns)
raw_cols_travelers   = list(df_travelers.columns)

print("Raw missing-value snapshots saved for later comparison.")

# ───────────────────────────────────────────────────────────────────
# 1.2  Categorical Cardinality Inspection
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.2 · Categorical Cardinality Inspection")
print("-" * 70)

def inspect_categorical(df, dataset_name):
    """Bar chart of cardinality + printout of unique values per column."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) == 0:
        print(f"  No categorical columns in {dataset_name}.")
        return

    card = df[cat_cols].nunique().sort_values(ascending=True)
    card_low  = card[card <= 400]
    card_high = card[card > 400]

    # Plot
    if not card_low.empty:
        plt.figure(figsize=(10, max(4, len(card_low) * 0.35)))
        colors = [CORAL if v > 15 else STEEL for v in card_low]
        card_low.plot(kind="barh", color=colors, edgecolor="white")
        plt.axvline(15, color=CORAL, ls="--", lw=1.5,
                    label="High-cardinality threshold (>15)")
        plt.title(f"Categorical Cardinality (≤400) — {dataset_name}",
                  fontweight="bold", color=NAVY)
        plt.xlabel("Unique Values")
        plt.legend()
        sns.despine()
        savefig(f"cardinality_{dataset_name.lower().replace(' ', '_')}")

    # High-cardinality warning
    if not card_high.empty:
        print("HIGH-CARDINALITY features (excluded from chart):")
        for col, val in card_high.items():
            print(f"    ➤ '{col}': {val} unique — needs deep cleaning")
        print()

    # Unique-value review
    print(f"  Reviewing unique values ({dataset_name}):")
    for col in card_low.index:
        uv = sorted(df[col].dropna().astype(str).unique())
        print(f"  ➤ '{col}' ({len(uv)} values): {uv}\n")

inspect_categorical(df_alarms,   "Alarms")
inspect_categorical(df_travelers, "Travelers")

# ───────────────────────────────────────────────────────────────────
# 1.3  Redundant-Column Resolution (Travelers)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.3 · Redundant-Column Resolution (Travelers)")
print("-" * 70)

# Pairs: (column_to_keep, column_to_drop)
# Decision based on Pipeline.md notes:
#   keep Tipo Documento > document_type
#   keep airline%       > airline
#   keep flight number  > flight_number
#   keep nationality_3  > nationality
#   keep FASCIA ETA     > age_group
redundant_pairs = [
    ("document_type",  "document_type2"),   # document_type2 = "Tipo_Documento" after rename
    ("airline",        "airline%"),
    ("flight_number",  "flight number"),
    ("nationality",    "nationality_3"),
    ("age_group",      "age_group2"),       # age_group2 = "FASCIA_ETA2" after rename
]

def _norm(series):
    return series.astype(str).str.strip().str.upper()

for keep_col, drop_col in redundant_pairs:
    if keep_col not in df_travelers.columns or drop_col not in df_travelers.columns:
        print(f"  ⚠ Column pair missing: '{keep_col}' / '{drop_col}' — skipped")
        continue

    s_keep = df_travelers[keep_col]
    s_drop = df_travelers[drop_col]

    # Conflicts: rows where both non-null but different
    mask_conflict = s_keep.notna() & s_drop.notna() & (_norm(s_keep) != _norm(s_drop))
    n_conflicts = mask_conflict.sum()

    # Fill missing in keep_col from drop_col  [ADDED — was not done before]
    n_filled = s_keep.isna().sum() - s_keep.fillna(s_drop).isna().sum()
    df_travelers[keep_col] = s_keep.fillna(s_drop)

    print(f"  ➤ '{keep_col}' ← merged from '{drop_col}'  "
          f"| {n_filled} NaNs filled | {n_conflicts} conflicts")
    if n_conflicts > 0:
        display(df_travelers.loc[mask_conflict, [keep_col, drop_col]].head(3))

# Drop the redundant columns  [ADDED — previously only inspected, never dropped]
cols_to_drop = [c for _, c in redundant_pairs if c in df_travelers.columns]
df_travelers.drop(columns=cols_to_drop, inplace=True)
print(f"\n  ✔ Dropped redundant columns: {cols_to_drop}")
print(f"  ✔ Travelers now has {df_travelers.shape[1]} columns (was {raw_shape_travelers[1]}).")

# ───────────────────────────────────────────────────────────────────
# 1.4  Text Standardisation & Unknown-Value Unification (Travelers)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.4 · Text Standardisation & Unknown-Value Unification (Travelers)")
print("-" * 70)

text_cols = df_travelers.select_dtypes(include=["object"]).columns

# a) Strip + UPPERCASE
for col in text_cols:
    df_travelers[col] = df_travelers[col].astype(str).str.strip().str.upper()

# b) Map unknown variants to the canonical 'N.D.'
unknown_variants = ["ND", "UNKNOWN", "UNKN", "N/A", "N/C"]
df_travelers[text_cols] = df_travelers[text_cols].replace(unknown_variants, "N.D.")

# c) Convert junk tokens to proper NaN
junk_values = ["-", "//", "?", "???", "XX", "ZZ", "NAN", ""]
df_travelers[text_cols] = df_travelers[text_cols].replace(junk_values, np.nan)

print("  ✔ Text → UPPERCASE, unknown variants → 'N.D.', junk → NaN.")

# Quick check on selected columns
for col in ["departure_airport_iata", "arrival_airport_iata", "gender"]:
    if col in df_travelers.columns:
        print(f"    '{col}' unique: {sorted(df_travelers[col].dropna().unique())}")

# ───────────────────────────────────────────────────────────────────
# 1.5  Numeric & Domain-Specific Standardisation (Travelers)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.5 · Numeric & Domain-Specific Standardisation (Travelers)")
print("-" * 70)

# ── 1.5a  Logical Mapping (country, year, month) ──
print("\n  1.5a — Logical Mapping")

if "arrival_country_code" in df_travelers.columns:
    before = df_travelers["arrival_country_code"].dropna().unique()
    df_travelers["arrival_country_code"] = df_travelers["arrival_country_code"].replace({"IT": "ITA"})
    after  = df_travelers["arrival_country_code"].dropna().unique()
    print(f"    arrival_country_code  BEFORE: {sorted(before)}")
    print(f"    arrival_country_code  AFTER:  {sorted(after)}")

if "departure_year" in df_travelers.columns:
    before = df_travelers["departure_year"].dropna().unique()
    year_map = {"24": "2024", "ANNO 2024": "2024", "anno 2024": "2024"}
    df_travelers["departure_year"] = df_travelers["departure_year"].replace(year_map)
    after  = df_travelers["departure_year"].dropna().unique()
    print(f"    departure_year  BEFORE: {sorted(before)}")
    print(f"    departure_year  AFTER:  {sorted(after)}")

if "departure_month" in df_travelers.columns:
    before = df_travelers["departure_month"].dropna().unique()
    month_map = {"GEN": "01", "FEB": "02"}
    df_travelers["departure_month"] = df_travelers["departure_month"].replace(month_map)
    after  = df_travelers["departure_month"].dropna().unique()
    print(f"    departure_month BEFORE: {sorted(before)}")
    print(f"    departure_month AFTER:  {sorted(after)}")

# ── 1.5b  Numeric columns: entries, investigated, alarms ──
print("\n  1.5b — Numeric Cleaning (entries, investigated, alarms)")

quant_cols = ["entries", "investigated", "alarms"]
for col in quant_cols:
    if col not in df_travelers.columns:
        continue
    before_dtype = df_travelers[col].dtype
    before_nulls = df_travelers[col].isna().sum()

    # Italian comma → dot
    df_travelers[col] = df_travelers[col].astype(str).str.replace(",", ".")
    # Strip non-numeric characters (removes 'PAX', '~', 'N.D.' etc.)
    df_travelers[col] = df_travelers[col].str.replace(r"[^\d.\-]", "", regex=True)
    # Convert to numeric
    df_travelers[col] = pd.to_numeric(df_travelers[col], errors="coerce")
    # Business-logic bounds: 0 ≤ value ≤ 1000
    df_travelers.loc[(df_travelers[col] < 0) | (df_travelers[col] > 1000), col] = np.nan
    # Nullable integer (can't have 1.5 passengers)
    df_travelers[col] = df_travelers[col].astype("Int64")

    after_dtype  = df_travelers[col].dtype
    after_nulls  = df_travelers[col].isna().sum()
    print(f"    {col}:  dtype {before_dtype}→{after_dtype}  |  NaN {before_nulls}→{after_nulls}")

# ── 1.5c  Zone column ──
print("\n  1.5c — Zone Cleaning")
if "zone" in df_travelers.columns:
    before_nulls = df_travelers["zone"].isna().sum()
    df_travelers["zone"] = df_travelers["zone"].astype(str).str.strip()
    df_travelers["zone"] = df_travelers["zone"].replace(["??", "?", "-"], np.nan)
    df_travelers["zone"] = pd.to_numeric(df_travelers["zone"], errors="coerce")
    df_travelers.loc[(df_travelers["zone"] < 0) | (df_travelers["zone"] > 10), "zone"] = np.nan
    df_travelers["zone"] = df_travelers["zone"].astype("Int64")
    after_nulls = df_travelers["zone"].isna().sum()
    print(f"    zone: NaN {before_nulls}→{after_nulls}  |  restricted to [0, 10]")

# ── 1.5d  Gender standardisation ──
print("\n  1.5d — Gender Standardisation")
if "gender" in df_travelers.columns:
    before_vals = sorted(df_travelers["gender"].dropna().unique())
    gender_map = {
        "1": "M", "MALE": "M", "MASCHIO": "M", "M": "M",
        "2": "F", "FEMALE": "F", "FEMMINA": "F", "F": "F",
        "N/B": "X", "X": "X",
    }
    df_travelers["gender"] = df_travelers["gender"].replace(gender_map)
    after_vals = sorted(df_travelers["gender"].dropna().unique())
    print(f"    BEFORE: {before_vals}")
    print(f"    AFTER:  {after_vals}")

# ───────────────────────────────────────────────────────────────────
# 1.6  Date Parsing & Temporal Alignment
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.6 · Date Parsing & Temporal Alignment")
print("-" * 70)

def extract_pattern(val):
    """Map digits→N, letters→A to reveal format heterogeneity."""
    s = str(val)
    s = re.sub(r"\d", "N", s)
    s = re.sub(r"[a-zA-Z]", "A", s)
    return s

# ── BEFORE: raw date patterns ──
print("\n  Date-format patterns BEFORE cleaning:")

travelers_patterns_before = (
    df_travelers["departure_date"].dropna().map(extract_pattern).value_counts()
)
alarms_patterns_before = (
    df_alarms["departure_date"].dropna().map(extract_pattern).value_counts()
)
print("    Travelers:")
print(travelers_patterns_before.to_string().replace("\n", "\n    "))
print("    Alarms:")
print(alarms_patterns_before.to_string().replace("\n", "\n    "))

# ── Regex normalisation for Alarms (multiple dirty formats) ──
# dd/mm/yyyy HH:MM:SS → yyyy-mm-dd HH:MM:SS
df_alarms["departure_date"] = (
    df_alarms["departure_date"].astype(str)
    .str.replace(
        r"(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})",
        r"\3-\2-\1 \4:\5:\6", regex=True)
)
# yyyy-mm-ddHH:MM:SS (missing space) → yyyy-mm-dd HH:MM:SS
df_alarms["departure_date"] = (
    df_alarms["departure_date"].astype(str)
    .str.replace(
        r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})",
        r"\1 \2", regex=True)
)
# yyyy/mm/dd → yyyy-mm-dd 00:00:00
df_alarms["departure_date"] = (
    df_alarms["departure_date"].astype(str)
    .str.replace(
        r"(\d{4})/(\d{2})/(\d{2})",
        r"\1-\2-\3 00:00:00", regex=True)
)
# dd.mm.yyyy → yyyy-mm-dd 00:00:00
df_alarms["departure_date"] = (
    df_alarms["departure_date"].astype(str)
    .str.replace(
        r"(\d{2})\.(\d{2})\.(\d{4})",
        r"\3-\2-\1 00:00:00", regex=True)
)
# MMM dd yyyy (e.g., "GEN 11 2024") → yyyy-mm-dd 00:00:00
df_alarms["departure_date"] = pd.to_datetime(
    df_alarms["departure_date"].astype(str)
    .str.extract(r"([A-Za-z]{3} \d{2} \d{4})")[0],
    format="%b %d %Y", errors="coerce"
).dt.strftime("%Y-%m-%d 00:00:00").fillna(df_alarms["departure_date"])

# dd-mm-yy → 20yy-mm-dd 00:00:00
df_alarms["departure_date"] = (
    df_alarms["departure_date"].astype(str)
    .str.replace(
        r"^(\d{2})-(\d{2})-(\d{2})$",
        lambda m: f"20{m.group(3)}-{m.group(2)}-{m.group(1)} 00:00:00",
        regex=True)
)
# ── Convert to datetime ──
df_alarms["departure_date"]   = pd.to_datetime(df_alarms["departure_date"],   errors="coerce")
df_travelers["departure_date"] = pd.to_datetime(df_travelers["departure_date"], errors="coerce")

# ── AFTER: verify patterns ──
alarms_patterns_after = (
    df_alarms["departure_date"].dropna()
    .astype(str).map(extract_pattern).value_counts()
)
travelers_patterns_after = (
    df_travelers["departure_date"].dropna()
    .astype(str).map(extract_pattern).value_counts()
)
print("\n  Date-format patterns AFTER cleaning:")
print("    Travelers:")
print(travelers_patterns_after.to_string().replace("\n", "\n    "))
print("    Alarms:")
print(alarms_patterns_after.to_string().replace("\n", "\n    "))

# ── Parse-failure report ──
n_nat_alarms   = df_alarms["departure_date"].isna().sum()
n_nat_travelers = df_travelers["departure_date"].isna().sum()
print(f"\n  Parse failures (NaT):  Alarms={n_nat_alarms}  |  Travelers={n_nat_travelers}")

# ── Force numeric types for quantitative columns ──
print("\n  Forcing remaining quantitative columns to numeric…")
quant_alarms = ["total", "zone_3", "total_flights", "zone"]
for col in quant_alarms:
    if col in df_alarms.columns:
        df_alarms[col] = pd.to_numeric(df_alarms[col], errors="coerce")

quant_travelers_extra = ["entries", "investigated", "alarms"]
for col in quant_travelers_extra:
    if col in df_travelers.columns:
        df_travelers[col] = pd.to_numeric(df_travelers[col], errors="coerce")

print(f"  ✔ Alarms  — {len(df_alarms.select_dtypes(include='number').columns)} numeric cols")
print(f"  ✔ Travelers — {len(df_travelers.select_dtypes(include='number').columns)} numeric cols")

# ───────────────────────────────────────────────────────────────────
# 1.7  Alarms Dataset — Full Cleaning Pipeline
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.7 · Alarms Dataset — Full Cleaning Pipeline")
print("-" * 70)

# ── 1.7a  Uppercase column names for consistency ──
df_alarms.columns = df_alarms.columns.str.upper()
missing_tokens = [
    "", "-", "//", "?", "n.d.", "nd", "n/a", "null",
    "unknown", "N.D.", "ND", "UNKNOWN", "XX", "ZZ", "00",
]
# ── 1.7b  Strip + case-normalise key text columns ──
upper_cols = [
    "ARRIVAL_AIRPORT_IATA", "DEPARTURE_COUNTRY_CODE",
    "DEPARTURE_AIRPORT_IATA", "DEPARTURE_COUNTRY",
    "ARRIVAL_COUNTRY_CODE", "ARRIVAL_COUNTRY_CODE_FULL",
]
for col in upper_cols:
    if col in df_alarms.columns:
        df_alarms[col] = df_alarms[col].astype(str).str.strip().str.upper()

title_cols = [
    "ARRIVAL_AIRPORT_DESCRIPTION", "DEPARTURE_AIRPORT_DESCRIPTION",
    "ARRIVAL_CITY", "DEPARTURE_CITY", "DEPARTURE_COUNTRY",
    "ARRIVAL_COUNTRY", "ARRIVAL_COUNTRY_PERCENTAGE",
    "DEPARTURE_COUNTRY_FULL",
]
for col in title_cols:
    if col in df_alarms.columns:
        df_alarms[col] = df_alarms[col].astype(str).str.strip().str.title()

# ── 1.7c  Clean month/year then DROP (unreliable) ──
if "DEPARTURE_MONTH" in df_alarms.columns:
    df_alarms["DEPARTURE_MONTH"] = df_alarms["DEPARTURE_MONTH"].replace({
        "GEN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAG": "05", "GIU": "06", "LUG": "07", "AGO": "08",
        "SET": "09", "OTT": "10", "NOV": "11", "DIC": "12",
    })
if "DEPARTURE_YEAR" in df_alarms.columns:
    df_alarms["DEPARTURE_YEAR"] = df_alarms["DEPARTURE_YEAR"].replace({
        "2024.": "2024", "24": "2024", "anno 2024": "2024",
    })
# After verifying that departure_date is now reliable (1.6), we drop the
# redundant year/month columns which contained inconsistent values.
for col in ["DEPARTURE_YEAR", "DEPARTURE_MONTH"]:
    if col in df_alarms.columns:
        df_alarms.drop(columns=[col], inplace=True)
        print(f"  ✔ Dropped '{col}' (redundant — extracted from DEPARTURE_DATE instead)")

# ── 1.7d  Country enrichment ──
# Fix arrival country code where country is Italia
if "ARRIVAL_COUNTRY" in df_alarms.columns:
    mask_it = df_alarms["ARRIVAL_COUNTRY"].str.lower().str.strip() == "italia"
    df_alarms.loc[mask_it, "ARRIVAL_COUNTRY_CODE"] = "ITA"
    print(f"  ✔ Fixed ARRIVAL_COUNTRY_CODE for {mask_it.sum()} rows where country = Italia")

# Merge DEPARTURE_COUNTRY_FULL into DEPARTURE_COUNTRY
if "DEPARTURE_COUNTRY_FULL" in df_alarms.columns and "DEPARTURE_COUNTRY" in df_alarms.columns:
    before_nulls = df_alarms["DEPARTURE_COUNTRY"].isna().sum()
    df_alarms["DEPARTURE_COUNTRY"] = (
        df_alarms["DEPARTURE_COUNTRY_FULL"]
        .combine_first(df_alarms["DEPARTURE_COUNTRY"])
    )
    after_nulls = df_alarms["DEPARTURE_COUNTRY"].isna().sum()
    print(f"  ✔ DEPARTURE_COUNTRY enriched from _FULL column: NaN {before_nulls}→{after_nulls}")

# ── 1.7e  Risk flag standardisation ──
if "RISK_FLAG" in df_alarms.columns:
    before_vals = df_alarms["RISK_FLAG"].dropna().unique()
    df_alarms["RISK_FLAG"] = df_alarms["RISK_FLAG"].fillna("LOW")
    df_alarms["RISK_FLAG"] = (
        df_alarms["RISK_FLAG"].str.upper()
        .replace({"ALTO": "HIGH", "MEDIO": "MEDIUM"})
    )
    after_vals = df_alarms["RISK_FLAG"].dropna().unique()
    print(f"  ✔ RISK_FLAG: {sorted(before_vals)} → {sorted(after_vals)}")

# ── 1.7f  Fill placeholder text for notes / alarm reason ──
for col in ["OPERATOR_NOTES", "ALARM_REASON"]:
    if col in df_alarms.columns:
        n_fill = df_alarms[col].isna().sum()
        df_alarms[col] = df_alarms[col].fillna("No information provided")
        print(f"  ✔ {col}: filled {n_fill} NaNs with placeholder text")

# ── 1.7g  Departure country → ISO alpha-3 via pycountry ──
def get_alpha3(country_it):
    if pd.isna(country_it):
        return np.nan
    country_it = str(country_it).strip()
    country_en = it_to_en.get(country_it)
    if not country_en:
        return np.nan
    if country_it == "Cossovo":
        return "RKS"
    try:
        return pycountry.countries.lookup(country_en).alpha_3
    except LookupError:
        return np.nan

if "DEPARTURE_COUNTRY" in df_alarms.columns:
    before_nulls = df_alarms.get("DEPARTURE_COUNTRY_CODE", pd.Series(dtype="object")).isna().sum()
    df_alarms["DEPARTURE_COUNTRY_CODE"] = df_alarms["DEPARTURE_COUNTRY"].apply(get_alpha3)
    after_nulls = df_alarms["DEPARTURE_COUNTRY_CODE"].isna().sum()
    print(f"  ✔ DEPARTURE_COUNTRY_CODE rebuilt via pycountry: NaN {before_nulls}→{after_nulls}")

# ── 1.7h  City & airport description cleanup + IATA back-fill ──
if "DEPARTURE_CITY" in df_alarms.columns:
    df_alarms["DEPARTURE_CITY"] = df_alarms["DEPARTURE_CITY"].replace(city_mapping)
    for col in ["DEPARTURE_CITY", "DEPARTURE_AIRPORT_DESCRIPTION"]:
        if col in df_alarms.columns:
            df_alarms[col] = df_alarms[col].replace(missing_tokens, np.nan)

    def clean_text(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        x = " ".join(x.split())
        return x.title()

    df_alarms["DEPARTURE_CITY"] = df_alarms["DEPARTURE_CITY"].apply(clean_text)

if "DEPARTURE_AIRPORT_DESCRIPTION" in df_alarms.columns:
    df_alarms["DEPARTURE_AIRPORT_DESCRIPTION"] = (
        df_alarms["DEPARTURE_AIRPORT_DESCRIPTION"].apply(
            lambda x: clean_text(x) if not pd.isna(x) else np.nan
        ))
# Back-fill from IATA mapping
if "DEPARTURE_AIRPORT_IATA" in df_alarms.columns:
    mapped_city    = df_alarms["DEPARTURE_AIRPORT_IATA"].map(
        lambda x: iata_mapping.get(x, {}).get("city", np.nan) if pd.notna(x) else np.nan
    )
    mapped_airport = df_alarms["DEPARTURE_AIRPORT_IATA"].map(
        lambda x: iata_mapping.get(x, {}).get("airport", np.nan) if pd.notna(x) else np.nan
    )
    if "DEPARTURE_CITY" in df_alarms.columns:
        n_fill = df_alarms["DEPARTURE_CITY"].isna().sum()
        df_alarms["DEPARTURE_CITY"] = df_alarms["DEPARTURE_CITY"].fillna(mapped_city)
        print(f"  ✔ DEPARTURE_CITY: back-filled {n_fill - df_alarms['DEPARTURE_CITY'].isna().sum()} from IATA mapping")

    if "DEPARTURE_AIRPORT_DESCRIPTION" in df_alarms.columns:
        n_fill = df_alarms["DEPARTURE_AIRPORT_DESCRIPTION"].isna().sum()
        df_alarms["DEPARTURE_AIRPORT_DESCRIPTION"] = (
            df_alarms["DEPARTURE_AIRPORT_DESCRIPTION"].fillna(mapped_airport)
        )
        print(f"  ✔ DEPARTURE_AIRPORT_DESCRIPTION: back-filled "
              f"{n_fill - df_alarms['DEPARTURE_AIRPORT_DESCRIPTION'].isna().sum()} from IATA mapping")

# ── 1.7i  Redundant columns in Alarms ──
print("\n  1.7i — Redundant-column check (Alarms)")

# TOTAL vs TOTAL_FLIGHTS
if "TOTAL" in df_alarms.columns and "TOTAL_FLIGHTS" in df_alarms.columns:
    diff_mask = df_alarms["TOTAL_FLIGHTS"] != df_alarms["TOTAL"]
    n_diff = diff_mask.sum()
    print(f"    TOTAL vs TOTAL_FLIGHTS: {n_diff} rows differ")
    if n_diff > 0:
        display(df_alarms.loc[diff_mask, ["TOTAL_FLIGHTS", "TOTAL"]].drop_duplicates().head(5))
    # Keep TOTAL_FLIGHTS (more complete per project notes), fill NaN from TOTAL, drop TOTAL
    df_alarms["TOTAL_FLIGHTS"] = df_alarms["TOTAL_FLIGHTS"].fillna(df_alarms["TOTAL"])
    df_alarms.drop(columns=["TOTAL"], inplace=True)
    print("    ✔ Merged TOTAL → TOTAL_FLIGHTS, dropped TOTAL")

# ZONE vs ZONE_3
if "ZONE" in df_alarms.columns and "ZONE_3" in df_alarms.columns:
    diff_mask = df_alarms["ZONE"] != df_alarms["ZONE_3"]
    n_diff = diff_mask.sum()
    print(f"    ZONE vs ZONE_3: {n_diff} rows differ")
    # Keep ZONE_3 (more complete), drop ZONE
    df_alarms["ZONE_3"] = df_alarms["ZONE_3"].fillna(df_alarms["ZONE"])
    df_alarms.drop(columns=["ZONE"], inplace=True)
    print("    ✔ Merged ZONE → ZONE_3, dropped ZONE")

# ───────────────────────────────────────────────────────────────────
# 1.8  Post-Cleaning Missing-Value Audit (BEFORE vs AFTER)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.8 · Post-Cleaning Missing-Value Audit (BEFORE vs AFTER)")
print("-" * 70)

def missing_comparison(df_now, raw_missing, dataset_name, raw_shape):
    """Side-by-side before/after missing-value table."""
    now_missing = df_now.isnull().sum()
    # Align on columns present in both
    all_cols = sorted(set(raw_missing.index) | set(now_missing.index))
    rows = []
    for c in all_cols:
        bef = raw_missing.get(c, "—")
        aft = now_missing.get(c, "dropped")
        rows.append({"Column": c, "Missing BEFORE": bef, "Missing AFTER": aft})
    report = pd.DataFrame(rows).set_index("Column")
    print(f"\n  {dataset_name}  (shape: {raw_shape} → {df_now.shape})")
    display(report[report["Missing BEFORE"] != report["Missing AFTER"]])

missing_comparison(df_alarms,   raw_missing_alarms,   "ALARMS",   raw_shape_alarms)
missing_comparison(df_travelers, raw_missing_travelers, "TRAVELERS", raw_shape_travelers)

# Visual confirmation: missingness bar charts AFTER cleaning
print("\n  Missingness bar chart (AFTER cleaning)")

print("  ▸ Alarms")
msno.bar(df_alarms, figsize=(10, 4), color=GREEN)
savefig("missing_bar_alarms_clean")

print("  ▸ Travelers")
msno.bar(df_travelers, figsize=(10, 4), color=GREEN)
savefig("missing_bar_travelers_clean")

# ───────────────────────────────────────────────────────────────────
# 1.9  Final Schema Verification
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("1.9 · Final Schema Verification")
print("-" * 70)

print(f"\n  ALARMS   — {df_alarms.shape[0]:,} rows × {df_alarms.shape[1]} cols")
print(f"  Columns: {list(df_alarms.columns)}")
print(f"  Dtypes:\n{df_alarms.dtypes.value_counts().to_string()}\n")

print(f"  TRAVELERS — {df_travelers.shape[0]:,} rows × {df_travelers.shape[1]} cols")
print(f"  Columns: {list(df_travelers.columns)}")
print(f"  Dtypes:\n{df_travelers.dtypes.value_counts().to_string()}")

# ═══════════════════════════════════════════════════════════════════
# 2  UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2 · UNIVARIATE ANALYSIS")
print("=" * 70)
# ───────────────────────────────────────────────────────────────────
# 2.1  Numerical Distributions
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("2.1 · Numerical Distributions")
print("-" * 70)

def plot_numeric_distributions(df, dataset_name):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        print(f"  No numeric columns in {dataset_name}.")
        return
    print(f"\n  Descriptive statistics — {dataset_name}:")
    display(df[num_cols].describe().T)

    n = len(num_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(num_cols):
        ax = axes[i]
        df[col].dropna().hist(bins=30, ax=ax, color=STEEL, edgecolor="white")
        ax.set_title(col, fontsize=10, color=NAVY, fontweight="bold")
        ax.set_ylabel("Count")
        sns.despine(ax=ax)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"Numerical Distributions — {dataset_name}",
                 fontweight="bold", color=NAVY, y=1.02)
    savefig(f"num_dist_{dataset_name.lower()}")

plot_numeric_distributions(df_alarms,   "Alarms")
plot_numeric_distributions(df_travelers, "Travelers")

# ───────────────────────────────────────────────────────────────────
# 2.2  Categorical Frequency Analysis
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("2.2 · Categorical Frequency Analysis")
print("-" * 70)

def plot_categorical_freq(df, cols, dataset_name, top_n=10):
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return
    n = len(cols_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols_present):
        counts = df[col].value_counts().head(top_n)
        counts.plot(kind="barh", ax=ax, color=STEEL, edgecolor="white")
        ax.set_title(f"{col} (top {top_n})", fontsize=10, fontweight="bold", color=NAVY)
        ax.set_xlabel("Count")
        sns.despine(ax=ax)
    plt.suptitle(f"Categorical Frequencies — {dataset_name}",
                 fontweight="bold", color=NAVY, y=1.02)
    savefig(f"cat_freq_{dataset_name.lower()}")

# Alarms: alarm reason, risk flag
alarm_cat_cols = [c for c in ["ALARM_REASON", "RISK_FLAG", "ARRIVAL_COUNTRY_CODE"]
                  if c in df_alarms.columns]
if alarm_cat_cols:
    plot_categorical_freq(df_alarms, alarm_cat_cols, "Alarms")

# Travelers: gender, document_type, transit_flag, age_group, control_outcome
trav_cat_cols = [c for c in ["gender", "document_type", "transit_flag",
                             "age_group", "control_outcome"]
                 if c in df_travelers.columns]
if trav_cat_cols:
    plot_categorical_freq(df_travelers, trav_cat_cols, "Travelers")

# ───────────────────────────────────────────────────────────────────
# 2.3  Temporal Distribution (departure_date by month)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("2.3 · Temporal Distribution")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (df_tmp, name) in zip(axes, [(df_alarms, "Alarms"), (df_travelers, "Travelers")]):
    date_col = "DEPARTURE_DATE" if "DEPARTURE_DATE" in df_tmp.columns else "departure_date"
    if date_col not in df_tmp.columns:
        continue
    monthly = df_tmp[date_col].dropna().dt.to_period("M").value_counts().sort_index()
    monthly.plot(kind="bar", ax=ax, color=STEEL, edgecolor="white")
    ax.set_title(f"Records per Month — {name}", fontweight="bold", color=NAVY)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)

savefig("temporal_distribution")

# ═══════════════════════════════════════════════════════════════════
# 3  BIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3 · BIVARIATE ANALYSIS")
print("=" * 70)

# ───────────────────────────────────────────────────────────────────
# 3.1  Entries vs Alarms (Travelers)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("§3.1 · Entries vs Alarms (Travelers)")
print("-" * 70)

if {"entries", "alarms"}.issubset(df_travelers.columns):
    corr = df_travelers[["entries", "alarms"]].corr().iloc[0, 1]
    print(f"  Pearson r(entries, alarms) = {corr:.3f}")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_travelers, x="entries", y="alarms",
                    alpha=0.4, color=STEEL, edgecolor="none")
    plt.title(f"Entries vs Alarms  (r = {corr:.3f})", fontweight="bold", color=NAVY)
    plt.xlabel("Entries")
    plt.ylabel("Alarms")
    sns.despine()
    savefig("scatter_entries_alarms")
else:
    print("'entries' or 'alarms' column not found — skipping.")

# ───────────────────────────────────────────────────────────────────
# 3.2  Alarm Reason × Zone (Alarms heatmap)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("3.2 · Alarm Reason × Zone Heatmap")
print("-" * 70)

zone_col  = "ZONE_3" if "ZONE_3" in df_alarms.columns else "ZONE"
reason_col = "ALARM_REASON" if "ALARM_REASON" in df_alarms.columns else None

if reason_col and zone_col in df_alarms.columns:
    ct = pd.crosstab(df_alarms[reason_col], df_alarms[zone_col])
    plt.figure(figsize=(10, max(4, len(ct) * 0.5)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5)
    plt.title("Alarm Reason × Zone", fontweight="bold", color=NAVY)
    plt.ylabel("Alarm Reason")
    plt.xlabel("Zone")
    savefig("heatmap_reason_zone")
else:
    print("Required columns not available — skipping.")

# ───────────────────────────────────────────────────────────────────
# 3.3  Top Routes by Volume
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("3.3 · Top Routes by Volume")
print("-" * 70)

dep_col = "DEPARTURE_AIRPORT_IATA" if "DEPARTURE_AIRPORT_IATA" in df_alarms.columns else None
arr_col = "ARRIVAL_AIRPORT_IATA" if "ARRIVAL_AIRPORT_IATA" in df_alarms.columns else None

if dep_col and arr_col:
    routes = (
        df_alarms.groupby([dep_col, arr_col]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    routes["route"] = routes[dep_col] + " → " + routes[arr_col]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=routes, y="route", x="count", palette="Blues_d")
    plt.title("Top 15 Routes by Alarm Count", fontweight="bold", color=NAVY)
    plt.xlabel("Count")
    plt.ylabel("Route")
    sns.despine()
    savefig("top_routes")
    display(routes[["route", "count"]])
else:
    print("Airport IATA columns not available — skipping.")

# ═══════════════════════════════════════════════════════════════════
# 4  MULTIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4 · MULTIVARIATE ANALYSIS")
print("=" * 70)
# ───────────────────────────────────────────────────────────────────
# 4.1  Correlation Matrix — Numerical Features
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("4.1 · Correlation Matrix")
print("-" * 70)

def plot_corr_matrix(df, dataset_name):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        print(f"  Not enough numeric columns in {dataset_name}.")
        return
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(max(6, len(num_cols)), max(5, len(num_cols) * 0.8)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlation Matrix — {dataset_name}", fontweight="bold", color=NAVY)
    savefig(f"corr_matrix_{dataset_name.lower()}")

plot_corr_matrix(df_alarms,   "Alarms")
plot_corr_matrix(df_travelers, "Travelers")

# ───────────────────────────────────────────────────────────────────
# 4.2  Pair Plot — Key Numerical Features (Travelers)
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("4.2 · Pair Plot — Travelers")
print("-" * 70)

pair_cols = [c for c in ["entries", "investigated", "alarms", "zone"]
             if c in df_travelers.columns]
if len(pair_cols) >= 2:
    sample = df_travelers[pair_cols].dropna()
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)
    g = sns.pairplot(sample, diag_kind="kde",
                     plot_kws={"alpha": 0.3, "color": STEEL},
                     diag_kws={"color": NAVY})
    g.figure.suptitle("Pair Plot — Travelers", y=1.02,
                      fontweight="bold", color=NAVY)
    savefig("pairplot_travelers")
else:
    print("Not enough numeric columns for pair plot.")

# ═══════════════════════════════════════════════════════════════════
# 5  OUTLIER ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5 · OUTLIER ANALYSIS")
print("=" * 70)
# ───────────────────────────────────────────────────────────────────
# 5.1  IQR-Based Outlier Detection
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("5.1 · IQR-Based Outlier Detection")
print("-" * 70)

def iqr_outlier_report(df, dataset_name):
    """Report outlier counts per numeric column using 1.5×IQR rule."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    rows = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((s < lower) | (s > upper)).sum()
        rows.append({
            "Column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Lower": lower, "Upper": upper,
            "Outliers": n_out, "Outlier %": round(n_out / len(s) * 100, 2),
        })
    report = pd.DataFrame(rows).set_index("Column")
    print(f"\n  {dataset_name} — IQR Outlier Report:")
    display(report)
    return report

iqr_alarms   = iqr_outlier_report(df_alarms,   "Alarms")
iqr_travelers = iqr_outlier_report(df_travelers, "Travelers")

# ───────────────────────────────────────────────────────────────────
# 5.2  Box Plots — Key Metrics
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("5.2 · Box Plots")
print("-" * 70)

def plot_boxplots(df, cols, dataset_name):
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return
    n = len(cols_present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols_present):
        sns.boxplot(y=df[col].dropna(), ax=ax, color=STEEL,
                    flierprops={"marker": "o", "markerfacecolor": CORAL, "markersize": 4})
        ax.set_title(col, fontsize=10, fontweight="bold", color=NAVY)
        sns.despine(ax=ax)
    plt.suptitle(f"Box Plots — {dataset_name}", fontweight="bold", color=NAVY, y=1.02)
    savefig(f"boxplots_{dataset_name.lower()}")

plot_boxplots(df_alarms,   ["TOTAL_FLIGHTS", "ZONE_3"], "Alarms")
plot_boxplots(df_travelers, ["entries", "investigated", "alarms", "zone"], "Travelers")

# ═══════════════════════════════════════════════════════════════════
# 6  CROSS-DATASET COVERAGE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6 · CROSS-DATASET COVERAGE")
print("=" * 70)
# ───────────────────────────────────────────────────────────────────
# 6.1  Temporal Overlap
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("6.1 · Temporal Overlap")
print("-" * 70)

date_al = "DEPARTURE_DATE" if "DEPARTURE_DATE" in df_alarms.columns else "departure_date"
date_tr = "departure_date"

if date_al in df_alarms.columns and date_tr in df_travelers.columns:
    al_min, al_max = df_alarms[date_al].min(), df_alarms[date_al].max()
    tr_min, tr_max = df_travelers[date_tr].min(), df_travelers[date_tr].max()
    print(f"  Alarms    : {al_min} → {al_max}")
    print(f"  Travelers : {tr_min} → {tr_max}")

    overlap_start = max(al_min, tr_min)
    overlap_end   = min(al_max, tr_max)
    if overlap_start <= overlap_end:
        print(f"  ✔ Overlap : {overlap_start} → {overlap_end}")
    else:
        print("No temporal overlap between datasets!")

# ───────────────────────────────────────────────────────────────────
# 6.2  Airport & Route Coverage
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("6.2 · Airport & Route Coverage")
print("-" * 70)

def coverage_report(col_alarms, col_travelers, label):
    al_col = col_alarms.upper() if col_alarms.upper() in df_alarms.columns else col_alarms
    tr_col = col_travelers

    if al_col not in df_alarms.columns or tr_col not in df_travelers.columns:
        print(f"  ⚠ '{label}' columns not found — skipping.")
        return

    set_al = set(df_alarms[al_col].dropna().astype(str).str.upper().unique())
    set_tr = set(df_travelers[tr_col].dropna().astype(str).str.upper().unique())
    intersection = set_al & set_tr
    union = set_al | set_tr
    jaccard = len(intersection) / len(union) if union else 0

    print(f"  {label}:")
    print(f"    Alarms only   : {len(set_al - set_tr)}")
    print(f"    Travelers only: {len(set_tr - set_al)}")
    print(f"    Shared        : {len(intersection)}")
    print(f"    Jaccard index : {jaccard:.2%}")

coverage_report("DEPARTURE_AIRPORT_IATA", "departure_airport_iata", "Departure Airport (IATA)")
coverage_report("ARRIVAL_AIRPORT_IATA",   "arrival_airport_iata",   "Arrival Airport (IATA)")

# ───────────────────────────────────────────────────────────────────
# 6.3  Final Summary & Observations
# ───────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("6.3 · Final Summary & Observations")
print("-" * 70)

print(f"""
  ALARMS dataset  : {df_alarms.shape[0]:,} rows × {df_alarms.shape[1]} columns
  TRAVELERS dataset: {df_travelers.shape[0]:,} rows × {df_travelers.shape[1]} columns

  Key observations:
  1. Data Quality: multiple date formats normalised; redundant columns merged
     and dropped; missing-token variants unified.
  2. Missingness: residual NaNs concentrated in optional fields (operator notes,
     risk codes, airline details) — acceptable for downstream modelling.
  3. Cardinality: high-cardinality features (flight_number, airline) will need
     grouping or encoding in feature-engineering phase.
  4. Distributions: entries/alarms are right-skewed → log-transform may help
     normalise for distance-based anomaly detectors.
  5. Outliers: IQR analysis flags extreme values in 'entries' and 'total_flights'
     that should be investigated before modelling.
  6. Cross-dataset: temporal overlap confirmed; airport coverage partially shared —
     route-level joins will be used in the feature-engineering step.

     EDA complete — data is ready for build_features.py
""")