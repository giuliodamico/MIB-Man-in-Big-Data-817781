"""
Exploratory Data Analysis
===================================================================
"""
# SETUP AND DATA LOADING
print("\n" + "="*60)
print("SECTION 1.1 · SETUP AND DATA LOADING")
print("="*60)

# Import
import warnings
from IPython.display import display
import missingno as msno
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
from config import (
    RAW_FILES, SNAPSHOT_DATE,
    NAVY, STEEL, CORAL, GREEN, PALETTE,
    manual_mapping, it_to_en, city_mapping, iata_mapping
)
# Plot settings
warnings.filterwarnings("ignore")
sns.set_theme(style='whitegrid', font_scale=1.05, palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 5)
pd.set_option('display.max_columns', 40)
pd.set_option('display.float_format', '{:.2f}'.format)

SAVE_FIGS = False  # set to True to save figures to io/ as PNG files
def savefig(name, tight=True):
    """Save current figure to io/{name}.png and show it."""
    if tight:
        plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"io/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

print('Setup complete')

# Load data and apply manual mapping
df_alarms = pd.read_csv(RAW_FILES["Alarms"], sep=',')
df_travelers = pd.read_csv(RAW_FILES["Traveler_type"], sep=',')
df_alarms = df_alarms.rename(columns=manual_mapping)
df_travelers = df_travelers.rename(columns=manual_mapping)

# Dataset inspection
def inspect_dataset(df, dataset_name):
    print(f"\n{'='*60}")
    print(f"--- {dataset_name} DATASET ---")
    print(f"{'='*60}")
    
    # 1. Shape
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # 2. Global Overview (Types, Missing, Cardinality)
    # We build a custom summary dataframe for a much cleaner output than .info()
    overview = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values (Cardinality)': df.nunique()
    })
    
    print("Data Overview:")
    display(overview)

inspect_dataset(df_alarms, "ALARMS (ALLARMI)")
inspect_dataset(df_travelers, "TRAVELERS (TIPOLOGIA_VIAGGIATORE)")

# Visualize missing values for the Alarms dataset
print("Missing Values - Alarms Dataset")
msno.bar(df_alarms, figsize=(10, 4))
plt.show()

# Visualize missing values for the Travelers dataset
print("Missing Values - Travelers Dataset")
msno.bar(df_travelers, figsize=(10, 4))
plt.show()

def inspect_categorical_clean(df, dataset_name):
    print(f"\n{'='*60}")
    print(f"--- CATEGORICAL INSPECTION: {dataset_name} ---")
    print(f"{'='*60}")
    
    # Select only categorical (text) columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) == 0:
        return
    
    # Calculate cardinality
    card = df[cat_cols].nunique().sort_values(ascending=True)
    
    # Keep only columns with <= 400 unique values for the chart
    card_to_plot = card[card <= 400]
    high_card_cols = card[card > 400]
    
    if not card_to_plot.empty:
        plt.figure(figsize=(10, 5))
        colors = ['#E8735A' if v > 15 else '#4682B4' for v in card_to_plot]
        card_to_plot.plot(kind='barh', color=colors, edgecolor='white')
        plt.axvline(15, color='#E8735A', ls='--', lw=1.5, label='High cardinality threshold (>15)')
        plt.title(f'Categorical Cardinality (<400 limits) - {dataset_name}', fontweight='bold', color='#1A3764')
        plt.xlabel('Number of Unique Values')
        plt.legend()
        sns.despine()
        plt.tight_layout()
        plt.show()
    
    print(f"\n--- Unique Values Breakdown ({dataset_name}) ---")
    
    # Display the problematic columns we excluded from the chart
    if not high_card_cols.empty:
        print("HIGH CARDINALITY FEATURES (Excluded from plot):")
        for col, val in high_card_cols.items():
            print(f"   ➤ '{col}': {val} unique values (Action needed in Step 2: Deep analysis and cleaning)")
        print("-" * 50)
        
    print("\n REVIEWING CATEGORIES (To spot typos/standardization issues):")
    for col in card_to_plot.index:
        # Sort values alphabetically
        unique_vals = sorted(df[col].dropna().astype(str).unique())
        print(f"➤ '{col}' ({len(unique_vals)} values):")
        print(f"   {unique_vals}\n")

inspect_categorical_clean(df_alarms, "ALARMS")
inspect_categorical_clean(df_travelers, "TRAVELERS")

# Inspecting redundant columns in Travelers dataset
print(f"\n{'=' * 60}")
print("--- STEP 1: INSPECTING REDUNDANT COLUMNS (TRAVELERS) ---")
print(f"{'=' * 60}")

# Column pairs to compare: (main_column, duplicate_column)
pairs_to_check = [
    ("document_type", "Tipo Documento"),
    ("airline", "airline%"),
    ("flight_number", "flight number"),
    ("nationality", "nationality_3"),
    ("age_group", "FASCIA ETA"),
]

def normalize(series):
    """Standardize values to make comparisons cleaner."""
    return series.astype(str).str.strip().str.upper()

for col1, col2 in pairs_to_check:
    if col1 not in df_travelers.columns or col2 not in df_travelers.columns:
        print(f"⚠️ Missing column: '{col1}' or '{col2}'")
        print("-" * 50)
        continue

    s1 = df_travelers[col1]
    s2 = df_travelers[col2]

    # Compare only rows where both columns are not null
    mask_conflict = s1.notna() & s2.notna() & (normalize(s1) != normalize(s2))
    n_conflicts = mask_conflict.sum()

    print(f"➤ {col1} vs {col2}")
    print(f"  Conflicts found: {n_conflicts}")

    if n_conflicts > 0:
        print("  Example conflicts:")
        display(df_travelers.loc[mask_conflict, [col1, col2]].head(3))

    print("-" * 50)

print(f"\n{'='*60}")
print("--- STEP 2: TEXT STANDARDIZATION & UNIFYING 'N.D.' ---")
print(f"{'='*60}")

# 1. Select all text (categorical) columns
text_cols = df_travelers.select_dtypes(include=['object']).columns

# 2. Global Standardization: Strip spaces and convert to UPPERCASE
# This automatically fixes issues like " nd ", "Unknown", or "N.D. "
for col in text_cols:
    df_travelers[col] = df_travelers[col].astype(str).str.strip().str.upper()

# 3. Standardize all "Unknown" variations into the official "N.D." format
# Since we already uppercased everything, we only need to look for uppercase variants
unknown_variants = ['ND', 'UNKNOWN', 'UNKN', 'N/A', 'N/C']
df_travelers[text_cols] = df_travelers[text_cols].replace(unknown_variants, 'N.D.')

# 4. Handle genuine junk values by converting them to actual pandas NaNs (Nulls)
# The string 'NAN' is generated when pandas converts an actual NaN to string in step 2
junk_values = ['-', '//', '?', '???', 'XX', 'ZZ', 'NAN', '']
df_travelers[text_cols] = df_travelers[text_cols].replace(junk_values, np.nan)

print("Standardization complete!")
print(" - Text converted to UPPERCASE and stripped of hidden spaces.")
print(" - All unknown variants (ND, UNKNOWN, N/C) unified to 'N.D.'.")
print(" - Junk symbols (-, ?, //) properly converted to NaN.")
print("-" * 60)

# Quick verification on a few key columns
cols_to_check = ['departure_airport_iata', 'arrival_airport_iata', 'gender']
for col in cols_to_check:
    if col in df_travelers.columns:
        print(f"Unique values in '{col}': {df_travelers[col].dropna().unique()}")

print(f"\n{'='*60}")
print("--- STEP 4: NUMERIC & SPECIFIC COLUMN STANDARDIZATION ---")
print(f"{'='*60}")

# ---------------------------------------------------------
# ACTION 1: Logical Mapping (Country, Year, Month)
# ---------------------------------------------------------

# 1. Arrival Country Code
if 'arrival_country_code' in df_travelers.columns:
    df_travelers['arrival_country_code'] = df_travelers['arrival_country_code'].replace({'IT': 'ITA'})

# 2. Departure Year (Catching both standard and uppercased formats)
if 'departure_year' in df_travelers.columns:
    year_map = {'24': '2024', 'ANNO 2024': '2024', 'anno 2024': '2024'}
    df_travelers['departure_year'] = df_travelers['departure_year'].replace(year_map)

# 3. Departure Month
if 'departure_month' in df_travelers.columns:
    month_map = {'GEN': '01', 'FEB': '02'}
    df_travelers['departure_month'] = df_travelers['departure_month'].replace(month_map)

print("1. Country, Year, and Month columns logically mapped.")

# ---------------------------------------------------------
# ACTION 2: Numeric Standardization (entries, investigated, alarms)
# ---------------------------------------------------------

quant_cols = ['entries', 'investigated', 'alarms']

for col in quant_cols:
    if col in df_travelers.columns:
        # Step A: Replace Italian comma with decimal dot (e.g., '1,0' -> '1.0')
        df_travelers[col] = df_travelers[col].astype(str).str.replace(',', '.')
        
        # Step B: Regex to remove all characters except digits, minus sign, and decimal dot
        # This automatically destroys ' PAX', '~', and 'N.D.' turning them into clean numbers or empty strings
        df_travelers[col] = df_travelers[col].str.replace(r'[^\d\.\-]', '', regex=True)
        
        # Step C: Convert to float (errors='coerce' turns empty strings into NaN)
        df_travelers[col] = pd.to_numeric(df_travelers[col], errors='coerce')
        
        # Step D: Business Logic Filter
        # Passengers cannot be negative. Also, a single cohort > 1000 on a flight is likely a typo (like 9999).
        # We replace these unrealistic values with NaN so they don't break the ML model.
        df_travelers.loc[(df_travelers[col] < 0) | (df_travelers[col] > 1000), col] = np.nan
        
        # Step E: Convert to Nullable Integer type ('Int64') since we can't have 1.5 people
        df_travelers[col] = df_travelers[col].astype('Int64')

print("2. 'entries', 'investigated' and 'alarms' cleaned and converted to nullable integers (Int64).")

# ---------------------------------------------------------
# ACTION 3: Cleaning the 'zone' column
# ---------------------------------------------------------

if 'zone' in df_travelers.columns:
    # Remove junk like '??'
    df_travelers['zone'] = df_travelers['zone'].astype(str).str.strip()
    df_travelers['zone'] = df_travelers['zone'].replace(['??', '?', '-'], np.nan)
    
    # Convert to numeric
    df_travelers['zone'] = pd.to_numeric(df_travelers['zone'], errors='coerce')
    
    # Business Logic Filter: Zones should be between 0 and 10.
    # We replace error codes like '-1' or '99' with NaN
    df_travelers.loc[(df_travelers['zone'] < 0) | (df_travelers['zone'] > 10), 'zone'] = np.nan
    
    # Convert to Nullable Integer ('Int64')
    df_travelers['zone'] = df_travelers['zone'].astype('Int64')

print("3. 'zone' successfully cleaned (values restricted to 0-10).")
print("-" * 60)

# ---------------------------------------------------------
# ACTION 4: Gender Standardization
# ---------------------------------------------------------
if 'gender' in df_travelers.columns:
    gender_map = {
        '1': 'M', 'MALE': 'M', 'MASCHIO': 'M', 'M': 'M',
        '2': 'F', 'FEMALE': 'F', 'FEMMINA': 'F', 'F': 'F',
        'N/B': 'X', 'X': 'X'  # 'X' is the international passport standard for Unspecified/Non-Binary
        # 'N.D.' will remain untouched as intended
    }
    df_travelers['gender'] = df_travelers['gender'].replace(gender_map)
    
    print("➤ Gender column standardized.")
    print(f"  Unique valid values now: {sorted(df_travelers['gender'].dropna().unique())}")
    print("-" * 60)

# ---------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------
print("➤ Quick Verification:")
if 'arrival_country_code' in df_travelers.columns:
    print(f"  - arrival_country_code: {df_travelers['arrival_country_code'].dropna().unique()}")
if 'departure_year' in df_travelers.columns:
    print(f"  - departure_year: {df_travelers['departure_year'].dropna().unique()}")
if 'departure_month' in df_travelers.columns:
    print(f"  - departure_month: {df_travelers['departure_month'].dropna().unique()}")
if 'entries' in df_travelers.columns:
    print(f"  - entries type: {df_travelers['entries'].dtype}")
    print(f"  - entries min/max: {df_travelers['entries'].min()} to {df_travelers['entries'].max()}")
if 'zone' in df_travelers.columns:
    print(f"  - zone type: {df_travelers['zone'].dtype}")
    print(f"  - zone unique valid values: {sorted(df_travelers['zone'].dropna().unique())}")

# ---------------------------------------------------------
# ACTION 2: The "Moment of Truth" - Missing Values Report
# ---------------------------------------------------------
print("➤ MISSING VALUES SUMMARY (True NaNs):")

# Count missing values per column
missing_counts = df_travelers.isna().sum()

# Filter only columns that actually have missing values and sort them descending
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

if missing_counts.empty:
    print("  Wow! 0 missing values in the entire dataset!")
else:
    # Calculate the percentage for business context
    missing_percent = (missing_counts / len(df_travelers)) * 100
    
    # Create a nice DataFrame to display the results cleanly
    missing_report = pd.DataFrame({
        'Missing Values': missing_counts, 
        'Percentage (%)': missing_percent.round(2)
    })
    display(missing_report)

print("-" * 60)

def extract_pattern(val):
    s = str(val)
    s = re.sub(r'\d', 'N', s)
    s = re.sub(r'[a-zA-Z]', 'A', s)  
    return s

travelers_pattern = (
    df_travelers['departure_date']
    .dropna()
    .map(extract_pattern)
    .value_counts()
)    
alarms_pattern = (
    df_alarms['departure_date']
    .dropna()
    .map(extract_pattern)
    .value_counts()
)
print(travelers_pattern)
print(alarms_pattern)

# fix AA/AA/AAAA AA:AA:AA
df_alarms['departure_date'] = df_alarms['departure_date'].astype(str).str.replace(r'(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})', r'\3-\2-\1 \4:\5:\6', regex=True)
# from AAAA-AA-AAAAA:AA:AA to AAAA-AA-AA AA:AA:AA
df_alarms['departure_date'] = df_alarms['departure_date'].astype(str).str.replace(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})', r'\1 \2', regex=True)
# from AAAA/AA/AA to AAAA-AA-AA AA:AA:AA where time will be missing
df_alarms['departure_date'] = df_alarms['departure_date'].astype(str).str.replace(r'(\d{4})/(\d{2})/(\d{2})', r'\1-\2-\3 00:00:00', regex=True)
# from AA.AA.AAAA to AAAA-AA-AA AA:AA:AA where time will be missing
df_alarms['departure_date'] = df_alarms['departure_date'].astype(str).str.replace(r'(\d{2})\.(\d{2})\.(\d{4})', r'\3-\2-\1 00:00:00', regex=True)
# from AAA AA AAAA to AAAA-AA-AA AA:AA:AA where time will be missing
df_alarms['departure_date'] = pd.to_datetime(
    df_alarms['departure_date'].astype(str).str.extract(r'([A-Za-z]{3} \d{2} \d{4})')[0],
    format='%b %d %Y',
    errors='coerce'
).dt.strftime('%Y-%m-%d 00:00:00').fillna(df_alarms['departure_date'])
# from AA-AA-AA to AAAA-AA-AA AA:AA:AA where time will be missing
df_alarms['departure_date'] = df_alarms['departure_date'].astype(str).str.replace(
    r'^(\d{2})-(\d{2})-(\d{2})$', 
    lambda m: f'20{m.group(3)}-{m.group(2)}-{m.group(1)} 00:00:00', 
    regex=True)

print(f'Travelers Pattern after cleaning:\n{travelers_pattern}')
print(f'Alarms Pattern after cleaning:\n{alarms_pattern}')

# --- 1. Dates and Temporal Categories ---
# We convert the full date to datetime, but leave ANNO (Year) and MESE (Month) as objects (categorical)
df_alarms['departure_date'] = pd.to_datetime(df_alarms['departure_date'], errors='coerce')
df_travelers['departure_date'] = pd.to_datetime(df_travelers['departure_date'], errors='coerce')

# --- 2. Forcing Quantitative Variables to Numeric ---
# 'coerce' will turn any hidden text (like "N/A" or " ") into a clean NaN (Missing Value)
quant_alarms = ['total', 'zone_3', 'total_flights', 'zone']
df_alarms[quant_alarms] = df_alarms[quant_alarms].apply(pd.to_numeric, errors='coerce')

quant_travelers = ['entries', 'investigated', 'alarms']
df_travelers[quant_travelers] = df_travelers[quant_travelers].apply(pd.to_numeric, errors='coerce')

# --- 3. Verification ---
print("Conversion Complete.")
print(f"Alarms dataset now has {len(df_alarms.select_dtypes(include=['float64', 'int64']).columns)} numerical columns.")
print(f"Travelers dataset now has {len(df_travelers.select_dtypes(include=['float64', 'int64']).columns)} numerical columns.")

# Data cleaning on ALARMS dataset
df_alarms.columns = df_alarms.columns.str.upper()
missing_tokens = ['', '-', '//', '?', 'n.d.', 'nd', 'n/a', 'null', 'unknown', 'N.D.', 'ND', 'UNKNOWN', 'XX', 'ZZ', '00']

df_alarms[['ARRIVAL_AIRPORT_IATA', 'DEPARTURE_COUNTRY_CODE', 'DEPARTURE_AIRPORT_IATA', 'DEPARTURE_COUNTRY', 'ARRIVAL_COUNTRY_CODE', 'ARRIVAL_COUNTRY_CODE_FULL']] = df_alarms[[
    'ARRIVAL_AIRPORT_IATA', 'DEPARTURE_COUNTRY_CODE', 'DEPARTURE_AIRPORT_IATA', 'DEPARTURE_COUNTRY', 'ARRIVAL_COUNTRY_CODE', 'ARRIVAL_COUNTRY_CODE_FULL']].apply(lambda x: x.str.strip().str.upper())

df_alarms[['ARRIVAL_AIRPORT_DESCRIPTION', 'DEPARTURE_AIRPORT_DESCRIPTION', 'ARRIVAL_CITY', 'DEPARTURE_CITY', 'DEPARTURE_COUNTRY', 'ARRIVAL_COUNTRY', 'ARRIVAL_COUNTRY_PERCENTAGE', 'DEPARTURE_COUNTRY_FULL']] = df_alarms[[
    'ARRIVAL_AIRPORT_DESCRIPTION', 'DEPARTURE_AIRPORT_DESCRIPTION', 'ARRIVAL_CITY', 'DEPARTURE_CITY', 'DEPARTURE_COUNTRY', 'ARRIVAL_COUNTRY', 'ARRIVAL_COUNTRY_PERCENTAGE', 'DEPARTURE_COUNTRY_FULL']].apply(lambda x: x.str.strip().str.capitalize())

# We clean the month and year columns to analyse it
df_alarms['DEPARTURE_MONTH'] = df_alarms['DEPARTURE_MONTH'].replace({
    'GEN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAG': '05', 'GIU': '06',
    'LUG': '07', 'AGO': '08', 'SET': '09', 'OTT': '10', 'NOV': '11', 'DIC': '12'})
df_alarms['DEPARTURE_YEAR'] = df_alarms['DEPARTURE_YEAR'].replace({
    '2024.': '2024', '24': '2024', 'anno 2024': '2024'})

# After distribution analysis of departure date, we decide to drop departure_year and departure_month as they are uncorrect.
df_alarms.drop(columns=['DEPARTURE_YEAR', 'DEPARTURE_MONTH'], inplace=True)

# TODO add comment
df_alarms.loc[df_alarms['ARRIVAL_COUNTRY'] == 'Italia', 'ARRIVAL_COUNTRY_CODE'] = 'ITA'

# TODO add comment
df_alarms['DEPARTURE_COUNTRY'] = df_alarms['DEPARTURE_COUNTRY_FULL'].combine_first(df_alarms['DEPARTURE_COUNTRY'])

# We standardize the RISK_FLAG values to be consistent and fill missing with 'LOW'
df_alarms['RISK_FLAG'] = df_alarms['RISK_FLAG'].fillna('LOW')
df_alarms['RISK_FLAG'] = df_alarms['RISK_FLAG'].str.upper().replace({'ALTO': 'HIGH', 'MEDIO': 'MEDIUM'})

# We fill missing values in OPERATOR_NOTES and ALARM_REASON with a placeholder text to avoid issues in analysis and to acknowledge that the information is missing.
df_alarms['OPERATOR_NOTES'] = df_alarms['OPERATOR_NOTES'].fillna('No notes provided')
df_alarms['ALARM_REASON'] = df_alarms['ALARM_REASON'].fillna('No reason provided')

#Cleaning Departure Country and Country Code
def get_alpha3_from_italian_country(country_it):
    if pd.isna(country_it):
        return np.nan
    
    country_it = str(country_it).strip()
    country_en = it_to_en.get(country_it)
    
    if not country_en:
        return np.nan
    
    # Kosovo special case
    if country_it == 'Cossovo':
        return 'RKS'
    
    try:
        return pycountry.countries.lookup(country_en).alpha_3
    except LookupError:
        return np.nan

df_alarms['DEPARTURE_COUNTRY_CODE'] = df_alarms['DEPARTURE_COUNTRY'].apply(get_alpha3_from_italian_country)

# cleaning departure city
df_alarms['DEPARTURE_CITY'] = df_alarms['DEPARTURE_CITY'].replace(city_mapping)
for col in ['DEPARTURE_CITY', 'DEPARTURE_AIRPORT_DESCRIPTION']:
    df_alarms[col] = df_alarms[col].replace(missing_tokens, np.nan)

def clean_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = ' '.join(x.split())
    return x.title()

df_alarms['DEPARTURE_CITY'] = df_alarms['DEPARTURE_CITY'].apply(clean_text)
df_alarms['DEPARTURE_AIRPORT_DESCRIPTION'] = df_alarms['DEPARTURE_AIRPORT_DESCRIPTION'].apply(clean_text)

def fill_city_from_iata(iata):
    if pd.isna(iata):
        return np.nan
    return iata_mapping.get(iata, {}).get('city', np.nan)

def fill_airport_from_iata(iata):
    if pd.isna(iata):
        return np.nan
    return iata_mapping.get(iata, {}).get('airport', np.nan)

mapped_city = df_alarms['DEPARTURE_AIRPORT_IATA'].map(fill_city_from_iata)
mapped_airport = df_alarms['DEPARTURE_AIRPORT_IATA'].map(fill_airport_from_iata)

df_alarms['DEPARTURE_CITY'] = df_alarms['DEPARTURE_CITY'].fillna(mapped_city)
df_alarms['DEPARTURE_AIRPORT_DESCRIPTION'] = df_alarms['DEPARTURE_AIRPORT_DESCRIPTION'].fillna(mapped_airport)

# TODO remove after inspection
# Count na for each column in df_alarms
na_counts = df_alarms.isna().sum()
na_percentages = (na_counts / len(df_alarms) * 100).round(2)
na_summary = pd.DataFrame({
    'Missing Values': na_counts,
    'Missing Percentage': na_percentages
}).sort_values(by='Missing Values', ascending=False)
print("\nMissing Values Summary for ALARMS Dataset:")
display(na_summary)

# TODO remove after inspection
# distribution of dates 
plt.figure(figsize=(10, 5))
sns.histplot(df_alarms['DEPARTURE_DATE'], bins=30, kde=True, color='#4682B4')
plt.title('Distribution of Departure Dates', fontweight='bold', color='#1A3764')
plt.xlabel('Departure Date')
plt.ylabel('Frequency')
sns.despine()
plt.tight_layout()
plt.show()

# TODO remove after inspection
# GRAPHIC DISTRIBUTION OF TOTAL FLIGHT 
plt.figure(figsize=(10, 5))
sns.histplot(df_alarms['TOTAL_FLIGHTS'], bins=30, kde=True, color='#4682B4')
plt.title('Distribution of Total Flights', fontweight='bold', color='#1A3764')
plt.xlabel('Total Flights')
plt.ylabel('Frequency')
sns.despine()
plt.tight_layout()
plt.show()

# TODO remove after inspection
# SEE IF TOTAL FLIGHTS HAS THESAME VALUE OF TOTAL
df_alarms['TOTAL_FLIGHTS'].equals(df_alarms['TOTAL'])
#SEE WHERE THEY DIFFER
df_alarms[df_alarms['TOTAL_FLIGHTS'] != df_alarms['TOTAL']][['TOTAL_FLIGHTS', 'TOTAL']]
# see all the different values
df_alarms[df_alarms['TOTAL_FLIGHTS'] != df_alarms['TOTAL']][['TOTAL_FLIGHTS', 'TOTAL']].drop_duplicates()
# For the moment (30/03) we replace missing_values in total_flights with mode and we drop total column

# TODO remove after inspection
# Zone is equal to zone_3?'
df_alarms['ZONE'].equals(df_alarms['ZONE_3'])
#SEE WHERE THEY DIFFER
df_alarms[df_alarms['ZONE'] != df_alarms['ZONE_3']][['ZONE', 'ZONE_3']]
# If the ZOne and ZOne3 columns represent the same information, we can drop one of them. We will keep ZONE_3 as it is more complete.