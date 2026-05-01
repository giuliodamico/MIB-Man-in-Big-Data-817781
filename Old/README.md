
# Transit Anomaly Detection — Whitehall Reply Project 2

**Whitehall Reply — Project 2 | Academic Year 2025/26**
**Team: Giulio D'Amico · Alexis Mitracos**

## Project Overview

Border control authorities and airport operators process thousands of passenger transits daily, each associated with rich metadata: timestamps, routes, nationalities, document types, control outcomes, and security alerts. Today, anomaly detection on such data is largely reactive — incidents are identified after the fact rather than flagged as they emerge. A proactive analytical system capable of identifying  uspicious patterns in real time would meaningfully improve both operational efficiency and security monitoring.

This project addresses that gap by implementing the same anomaly detection system twice, under two distinct paradigms, and producing a rigorous comparative analysis of the results.

The **classical pipeline** follows a structured, deterministic workflow:
raw data is cleaned and merged, features are engineered from historical baselines, three unsupervised detectors (Isolation Forest, Local Outlier Factor, Z-score) are applied in ensemble, and a rule-based post-processing layer confirms final anomalies. Every step is reproducible and auditable.

The **multi-agent architecture** distributes the same logic across five specialised LangGraph agents — Data Agent, Baseline Agent, Outlier Detection Agent, Risk Profiling Agent, and Report Agent — each  esponsible for a discrete analytical task. Three of the five agents incorporate a local LLM (Gemma via LM Studio) to interpret intermediate results, prioritise findings, and generate a structured natural-language report. A shared set of global safety rules is prepended to every LLM prompt to enforce factual accuracy and operational neutrality across all agents.

The objective is not simply to detect anomalies, but to understand under which operational conditions each paradigm is preferable — comparing the two approaches across dimensions of modularity,  interpretability, flexibility, scalability, and practical deployability. The Jaccard similarity between the anomaly sets produced by both pipelines serves as the primary quantitative validation metric.

## Dataset

## 1. Dataset Description

### 1.1 Overview

The project uses two datasets provided by the client, both derived from Italian airport border-control operations. They describe two complementary observation units and share a subset of join keys.

| Dataset               | Rows  | Columns | Observation unit                               |
| --------------------- | ----- | ------- | ---------------------------------------------- |
| TIPOLOGIA_VIAGGIATORE | 5,095 | 33      | Traveler-category daily aggregates per airport |
| ALLARMI               | 5,080 | 25      | Alarm events per flight and departure airport  |

Both files were delivered in Italian with inconsistent casing, typographic errors (e.g. `AREOPORTO`), and a mixture of Italianised and English spellings for countries, cities, and airports.

---

### 1.2 Reference Mappings

Before loading the raw files, four curated dictionaries were prepared to support the cleaning pipeline:

- `manual_mapping` — Italian to English column-name mapping. Disambiguates
  duplicated columns (e.g. `Tipo_Documento` and `TIPO_DOCUMENTO` become
  `document_type` and `document_type2`) that are reconciled in the cleaning
  section.
- `it_to_en` — Italian to English country names, consumed by `pycountry`
  to emit the ISO alpha-3 code of every departure country. Kosovo, which
  has no official ISO code, is hand-coded as `RKS`.
- `city_mapping` — Italianised city names mapped to their canonical English
  spelling.
- `iata_mapping` — a curated `IATA → {city, airport}` lookup used to
  back-fill missing city and airport descriptions when the IATA code is
  populated but the free-text fields are not.
- `occurrences_map` — maps the categorical labels in the ALLARMI
  `OCCURRENCES` column (e.g. `"Voli con Allarmi"`, `"Viaggiatori entrati nel Sistema"`) to a clean controlled English vocabulary.

All dictionaries are embedded directly in the notebook to keep it fully
self-contained, as required by the submission guidelines.

---

### 1.3 Raw Data Loading

Both CSVs are loaded and the Italian-to-English column rename is applied
in a single step. The two DataFrames (`df_alarms`, `df_travelers`) are
kept separate throughout the cleaning phase because they describe different
observation units and share only a subset of columns. They are merged into
a single master dataset only after cleaning is complete.

---

### 1.4 Missing Value Analysis

The audit reveals that both datasets are largely complete, with missing
values concentrated in a small number of columns.

In **TIPOLOGIA_VIAGGIATORE**, the columns with missing values are:
NATIONALITY (2.28%), DOCUMENT_TYPE (1.22%), GENDER (0.88%), AIRLINE (1.71%),
FLIGHT_NUMBER (1.37%), CONTROL_OUTCOME (25.3%), OPERATOR_NOTES (98.8%),and RISK_CODE (99.2%).

In **ALLARMI**, the columns with missing values are: DEPARTURE_AIRPORT_
DESCRIPTION (2.15%), DEPARTURE_CITY (1.99%), DEPARTURE_COUNTRY (1.46%),
DEPARTURE_COUNTRY_CODE (1.06%), ALARM_REASON (22.8%), OPERATOR_NOTES
(98.5%), and RISK_FLAG (99.0%).

> 📊 *Missing value bar charts for both datasets are available in
> sections 2.1 of the notebook.*

The handling strategy for each column is addressed in the cleaning
section that follows.

---

### 1.5 Cardinality and Data Quality Audit

To go beyond the binary present/missing view, a categorical cardinality
inspection was performed on all columns in both datasets. Columns were
split into two buckets: low-cardinality (≤ 400 unique values, amenable
to visual inspection and rule-based cleanup) and high-cardinality (> 400,
requiring grouping or encoding). Cardinality above 15 was flagged as a
soft threshold for downstream one-hot encoding caution.

This inspection revealed data quality issues well beyond what the missing
value analysis alone captured. Columns officially showing 0% missing are
in fact dirty: numeric fields like `ENTRIES`, `ALARMS`, and `TOTAL`
contain strings such as `"1 pax"`, `"~5"`, `"N.D."`, `"-500"`, and
`"0,0"`, making them unusable as-is. Categorical fields like `GENDER`,
`NATIONALITY`, `DEPARTURE_AIRPORT_IATA`, and `AIRLINE` suffer from severe
inconsistency — mixed case (`"TIA"`, `"Tia"`, `"tia"`), trailing spaces,
and multiple representations of unknown values (`"-"`, `"//"`, `"?"`,
`"ND"`, `"N.D."`, `"unknown"`).

These findings motivated the targeted multi-step cleaning strategy
described in the following section.

## 2. Data Cleaning

### 2.1 TIPOLOGIA_VIAGGIATORE — Cleaning Steps

#### 2.1.1 Redundant Column Resolution

The Travelers dataset ships with duplicated columns encoding the same logical
field under different formats — `document_type` vs `document_type2`,
`flight_number` vs `flight number`, `nationality` vs `nationality_3`. For
each pair, conflicting rows (both non-null but different after normalisation)
were quantified before merging. The more complete or normalised column was
retained as the canonical source, with the legacy column used only to fill
remaining gaps before being dropped.

#### 2.1.2 Text Standardisation and Unknown-Value Unification

Three passes were applied to every text column. First, `strip + UPPERCASE`
to collapse casing and whitespace variants. Second, all unknown tokens
(`ND`, `UNKNOWN`, `N/A`, `n.d.`) were mapped to the canonical `N.D.` to
preserve them as informative categoricals rather than converting them to
NaN. Third, junk tokens (`-`, `//`, `?`, `XX`) were converted to proper
NaN so downstream imputation logic can handle them correctly.

#### 2.1.3 Numeric and Domain-Specific Standardisation

Numeric columns arrived mixed with unit suffixes (`PAX`), Italian decimal
commas, and illegal sentinel values. Four sub-steps were applied.

- **Year and month recoding** — Italian 2-digit years and month
  abbreviations were normalised. Country codes were aligned (`IT` → `ITA`).
- **Entry counts** — `entries`, `investigated`, `alarms`: decimal commas
  replaced with dots, non-numeric characters stripped, values clipped to
  `[0, 1000]`, and cast to pandas nullable `Int64`.
- **Zone** — cleaned to integer in `[0, 10]`, out-of-range values
  nullified.
- **Gender** — numeric codes (`1`/`2`), Italian variants (`MASCHIO`,
  `FEMMINA`), and English variants collapsed to canonical `M / F / X`.

---

### 2.2 ALLARMI — Cleaning Steps

#### 2.2.1 Redundant Column Resolution

Two pairs of columns encoded the same logical field. `TOTAL` and
`TOTAL_FLIGHTS` both counted total flights — `TOTAL_FLIGHTS` was retained.
`ZONE` and `ZONE_3` both encoded the risk zone — values were merged via
`combine_first` before dropping `ZONE_3`. Conflicting rows were quantified
before each merge. All column names were uppercased to align with the
Travelers convention.

#### 2.2.2 Derived Columns Removed

`DEPARTURE_MONTH` and `DEPARTURE_YEAR` were dropped as both are derivable
from `DEPARTURE_DATE`, which becomes the single authoritative temporal
reference.

#### 2.2.3 Text Standardisation and Unknown-Value Unification

The same three-pass strategy was applied. UPPERCASE for codes and
identifiers, Title Case for city and country descriptors. Unknown tokens
replaced with NaN. Free-text columns (`ALARM_REASON`, `OPERATOR_NOTES`)
received explicit placeholders (`NO REASON PROVIDED`, `NO MANUAL NOTES`)
rather than NaN to preserve row count and categorical usability.

#### 2.2.4 Domain-Specific Standardisation

- **IATA codes** — uppercased and stripped for consistent merge key matching.
- **Redundant arrival columns** — `ARRIVAL_COUNTRY_CODE` and
  `ARRIVAL_COUNTRY_CODE_PERCENTAGE` dropped since all flights arrive in Italy.
- **ZONE** — sentinel values (`-1`, `99`, `??`) nullified, cast to integer,
  gaps filled from `ZONE_3`.
- **RISK_FLAG** — Italian labels translated (`ALTO` → `HIGH RISK`,
  `MEDIO` → `MEDIUM RISK`), NaN filled with `LOW RISK`.
- **Country enrichment** — `DEPARTURE_COUNTRY_FULL` (0 nulls) used to
  derive missing `DEPARTURE_COUNTRY_CODE` via `pycountry` ISO alpha-3
  mapping, reducing nulls from 102 to 3.
- **City and airport** — gaps filled from IATA mapping dictionary,
  reducing city nulls to 4 and airport nulls to 1.
- **TOTAL_FLIGHTS** — suffixes (`"1 voli"`), approximation markers (`"~5"`)
  stripped, values cast to integers capped at 1,000. `TOTAL` dropped.
- **Date parsing** — six mixed formats detected and normalised to ISO 8601
  using a pattern fingerprint technique. Italian month abbreviations
  (`GEN`, `FEB`) translated before final parsing.

#### 2.2.5 Cleaning Results Summary

| Step                          | Outcome                                              |
| ----------------------------- | ---------------------------------------------------- |
| Redundant columns             | `TOTAL` and `ZONE_3` dropped after merge         |
| OCCURRENCES                   | 15 clean categories, 3 NaN remaining                 |
| ZONE                          | 9 valid zones retained, sentinel values nullified    |
| ALARM_REASON                  | 6 categories,`NO REASON PROVIDED` placeholder used |
| OPERATOR_NOTES                | 5 categories,`NO MANUAL NOTES` placeholder used    |
| RISK_FLAG                     | LOW RISK 5,029 — HIGH RISK 27 — MEDIUM RISK 24     |
| DEPARTURE_COUNTRY_CODE        | NaN reduced 102 → 3 via ISO mapping                 |
| DEPARTURE_CITY                | 4 nulls remaining after IATA imputation              |
| DEPARTURE_AIRPORT_DESCRIPTION | 1 null remaining after IATA imputation               |
| TOTAL_FLIGHTS                 | Clean integer, max 544, mean 40                      |
| Date formats                  | All rows normalised to ISO 8601                      |

---

### 2.3 Post-Cleaning Missing Value Review

After standardisation, a second audit was run on both datasets to measure
the net effect of cleaning on missing values.

**ALLARMI** — cleaning resolved the majority of gaps. `ALARM_REASON` was
fully imputed (1,160 → 0) via placeholder. `OPERATOR_NOTES` and `RISK_FLAG`were similarly resolved to 0 remaining nulls. `DEPARTURE_COUNTRY_CODE`
was reduced from 54 to 3 via ISO mapping, and `DEPARTURE_CITY` from 101 to 4. `DEPARTURE_AIRPORT_DESCRIPTION` retains 1 null. `OCCURRENCES` introduced 3 new NaN from junk-token conversion. Five columns (`ARRIVAL_COUNTRY_CODE`, `DEPARTURE_COUNTRY`, `DEPARTURE_MONTH`,`DEPARTURE_YEAR`, `TOTAL`, `ZONE_3`) were dropped as planned.

**TIPOLOGIA_VIAGGIATORE** NEED TO BE CHANGE
 — several columns saw an increase in missing
values as a direct consequence of standardisation converting junk tokens to NaN. `NATIONALITY` increased from 116 to 232, `AIRLINE` from 87 to 192, `FLIGHT_NUMBER` from 70 to 141, `DOCUMENT_TYPE` from 62 to 140, `GENDER` from 45 to 110, `ALARMS` and `INVESTIGATED` from 0 to 33.`CONTROL_OUTCOME` (1,289) and `OPERATOR_NOTES` (5,034) and `RISK_CODE`(5,054) remain unchanged — these were already identified as high-missing or near-empty columns and are handled in the imputation step.`ENTRIES` introduced 40 new NaN from numeric cleaning.

The increase in missing counts post-cleaning is expected and correct — it reflects junk values that were previously masking true gaps. These
remaining nulls are addressed in the imputation step that follows.

## 3. Exploratory Data Analysis

### 3.1 EDA Summary and Key Takeaways

Before moving to feature engineering, six observations from the exploratory
analysis drive the design choices in the pipeline.

1. **Data quality** — dates normalised across six mixed formats; redundant
   columns merged and dropped; missing-token variants unified to a single
   canonical representation.
2. **Residual missingness** — remaining NaN values concentrate on truly
   optional fields (operator notes, risk codes, airline details) that carry
   low analytical value.
3. **Cardinality** — `FLIGHT_NUMBER` and `AIRLINE` are high-cardinality
   and require grouping or target encoding before being used as features.
4. **Distributions** — `ENTRIES` and `ALARMS` are right-skewed count
   variables. Ratio-based features (alarm rate, investigation rate) will
   perform better than raw counts for distance-based detectors.
5. **Outliers** — IQR analysis flags extreme values in `ENTRIES` and
   `TOTAL_FLIGHTS`. These are intentionally retained — they represent
   the signal the anomaly detectors are designed to find.
6. **Cross-dataset coverage** — temporal overlap between the two datasets
   is confirmed. Partial airport overlap implies a left join on Travelers
   as the base table to avoid discarding traveler records with zero alarms.

---

### 3.2 Univariate Analysis — ALLARMI Categorical Distributions

> 📊 *Figure — Categorical Frequencies (Alarms)*

The two most informative categorical columns after cleaning are
`ALARM_REASON` and `RISK_FLAG`.

`ALARM_REASON` distributes relatively evenly across five operational
categories. **SDI** (Sistema d'Indagine — the Italian national
investigation database) and **NSIS** (National Schengen Information
System) are the two primary institutional sources, each accounting for
roughly 800 records. **INTERPOL** and **TSC** (Terrorist Screening
Center) follow closely. **MANUALE** indicates alarms raised manually
by border control officers rather than by an automated system.
`NO REASON PROVIDED` (1,160 records) represents the previously missing
values, now preserved as an explicit categorical.

`RISK_FLAG` is heavily imbalanced: 5,029 out of 5,080 records carry
`LOW RISK`, with only 27 `HIGH RISK` and 24 `MEDIUM RISK` cases.
This imbalance is operationally meaningful — elevated risk flags are
rare events by design — but it reinforces the need for rate-based
anomaly detection rather than raw count thresholds.

---

### 3.3 Bivariate Analysis — Entries vs Alarms

> 📊 *Figure — Entries vs Alarms (r = 0.580)*

The scatter plot reveals a moderate positive correlation (r = 0.580)
between the number of passengers processed (`ENTRIES`) and the number
of alarms triggered. The relationship is not linear — at higher entry
volumes the alarm count disperses significantly, indicating that flight
volume alone does not explain alarm activity. A substantial cluster of
high-entry, low-alarm observations confirms that busy flights are not
systematically more suspicious. This motivates the use of `alarm_rate`
(alarms per entry) as the primary detection feature rather than raw
alarm counts.

---

### 3.4 Bivariate Analysis — Alarm Reason × Zone Heatmap

> 📊 *Figure — Alarm Reason × Zone*

The heatmap crosses `ALARM_REASON` against `ZONE` to reveal which alarm
types concentrate in which geographic risk zones. Zones 2, 4, and 5
dominate across all alarm categories — these three zones account for
the large majority of all records regardless of alarm type, suggesting
they represent the highest-volume operational areas. No alarm category
shows a strong exclusive affinity for a single zone, meaning alarm type
and zone provide partially independent information. This supports
including both as features or priors in the anomaly detector rather than
treating one as redundant.

---

### 3.5 Top Routes by Alarm Count

> 📊 *Figure — Top 15 Routes by Alarm Count*

Two corridors dominate alarm activity. UK-to-Italy routes lead the
ranking — LHR → LIN (102), STN → BGY (101), LHR → FCO (97),
LGW → MXP (86), STN → CIA (84) — with London airports (Heathrow,
Stansted, Gatwick) as the primary departure hubs. Tirana-to-Italy
routes form the second cluster — TIA → BGY (94), TIA → BLQ (66),
TIA → TSF (63), TIA → PSA (62) — making Tirana the most active
non-UK departure hub in the dataset. One Middle Eastern route
appears: DOH → MXP (60).

These high-volume routes will produce the most reliable per-airport
baselines. Low-volume routes will require regularisation to avoid
noise-driven false positives.

### Classical Pipeline

```markdown
## 4. Feature Engineering

### 4.1 Overview

The feature engineering block transforms the two cleaned tables into a
single model-ready feature panel. The cleaning step has already normalised
dtypes and free-text categoricals — here the focus is on semantic alignment,
temporal aggregation, and the construction of signal features that the
anomaly detectors will consume.

---

### 4.2 Dataset Merge

Both cleaned datasets are brought to the same temporal grain (daily,
via `dt.normalize()`) before joining. The Alarms dataset is aggregated
into a daily, airport-keyed panel by one-hot encoding `ALARM_REASON`
and `RISK_FLAG`, summing each dummy per `(day, departure_airport_iata)`,
and adding a `total_alarms_day` scalar. This transforms the raw event
log into a structured feature panel.

The Travelers table serves as the base observation unit. A left join
against the aggregated Alarms panel on `(merge_date,
departure_airport_iata)` produces the master dataset. Traveler rows
with no matching alarm record receive `0` in every alarm column —
semantically correct, as they genuinely had zero alarms that day at
that airport.

Any row whose `departure_date` failed to parse (`NaT`) is dropped
explicitly before the join, as it cannot be temporally aligned.

**Master dataset: 4,987 rows, 38 columns.**

---

### 4.3 Engineered Features

Each feature answers a specific business-level question about whether
a given observation is anomalous relative to its historical context.

| Feature | Business question |
|---------|------------------|
| `alarm_rate` | How alarm-dense is today's traffic? |
| `investigation_rate` | What fraction of entries were investigated? |
| `airport_historical_avg_rate` | What does normal look like at this airport? |
| `rate_deviation` | How far is today from the airport baseline? |
| `alarm_rate_yesterday` | What was yesterday's alarm rate? |
| `rolling_7d_avg_rate` | What is the short-term trend over the past 7 days? |
| `airport_historical_avg_entries` | What is the typical traffic volume at this airport? |
| `traffic_multiplier` | Is today's passenger volume itself abnormal? |
| `is_weekend` | Does day-of-week carry risk? |
| `month` | Is there monthly seasonality? |
| `zone_risk_weight` | What is the geographic risk weight of this zone? |

**Per-airport baseline.** `airport_historical_avg_rate` is computed
over the full observation window. In a production setting this should
be fitted on a strictly earlier training window to avoid look-ahead
bias. In this offline unsupervised exercise the entire window is the
reference set, and we acknowledge this limitation explicitly.

**Lag and rolling features.** Two strictly historical features are
computed per airport to capture short-term momentum. `alarm_rate_yesterday`
uses `shift(1)` to access the previous day's value.
`rolling_7d_avg_rate` uses `.shift(1).rolling(7)` — the `shift` is
critical: a plain `rolling(7).mean()` would include the current day
and produce a textbook look-ahead bias at inference time. The first
observation per airport falls back to the per-airport historical
baseline as a neutral prior.

**Traffic multiplier.** A second baseline on volume rather than rate
answers the complementary question: is today's traffic itself abnormal?
The ratio `entries / airport_historical_avg_entries` is defaulted to
`1.0` when the baseline is zero (new airport with no history) — not
`0`, which would falsely flag the observation as anomalous.

---

### 4.4 Feature Correlation Check

> 📊 *Figure — Correlation Matrix of Engineered Features*

The correlation matrix confirms that the engineered features capture
largely distinct signals. Three observations are worth noting.

`alarm_rate` correlates strongly with `airport_historical_avg_rate`
(r = 0.71) and `rolling_7d_avg_rate` (r = 0.70) — expected, since
both are derived from it. `rate_deviation` is by construction
orthogonal to `airport_historical_avg_rate` (r = 0.00), confirming
it isolates the deviation signal cleanly. `investigation_rate` is
nearly independent of all other features (|r| < 0.07), making it a
genuinely complementary signal. No pair exceeds |r| = 0.90, so no
feature is removed on collinearity grounds at this stage.

---

### 4.5 Feature Selection

The final feature set balances context (raw volumes that anchor the
observation) and signal (deviations from airport-specific baselines).
Three selection principles were applied.

First, raw text and identifier columns are excluded — they carry no
signal for a distance-based or tree-based detector. Second, features
that duplicate information already captured in baseline form are
excluded: `rate_deviation` is kept and raw `alarm_rate` is dropped,
since the deviation carries the relative-risk signal the detector
needs. Third, final dimensionality is kept moderate at 9-10 features.
With a dataset of this size, Isolation Forest is in a regime where
axis-aligned splits are informative, and LOF distances remain
meaningful — curse-of-dimensionality effects become material above
approximately 20 features.

Two optional features (`RISK_FLAG_HIGH`, `zone_risk_weight`) are
included only when present in the master dataset, keeping the
notebook robust to column availability across different perimeters.

**Final feature set:**
`entries`, `total_alarms_day`, `traffic_multiplier`, `rate_deviation`,
`alarm_rate_yesterday`, `rolling_7d_avg_rate`, `investigation_rate`,
`is_weekend`, `month` — plus `RISK_FLAG_HIGH` and `zone_risk_weight`
when available.
```

## 5. Anomaly Detection Models

### 5.1 Overview

Three complementary unsupervised detectors are applied to the engineered
feature panel, each operating under a different inductive bias. A fourth
density-based detector (DBSCAN) is added as an independent extension to
validate the ensemble without a fixed contamination assumption. All detectors
use the same `contamination=0.03` budget — surfacing the top 3% of
observations as anomalies, representing a realistic human review load.

---

### 5.2 Isolation Forest

**Method.** Isolation Forest (Liu, Ting, Zhou, 2008) isolates anomalies
by building a forest of random trees. Each tree recursively partitions the
feature space with random axis-aligned splits. Anomalies — being few and
structurally different — are isolated in fewer splits than normal points,
producing shorter average path lengths. The method is scale-invariant, so
no feature standardisation is required.

**Key hyperparameters:** `n_estimators=100`, `contamination=0.03`,
`max_samples='auto'` (256 per tree, per the original paper).
The severity score is sign-flipped from the native `decision_function`
so that higher values mean more anomalous.

**Result:** 4,987 records analysed → **150 anomalies flagged (3.0%).**

---

### 5.3 Local Outlier Factor

**Method.** LOF (Breunig, Kriegel, Ng, Sander, 2000) compares the local
density around each point with the densities of its k nearest neighbours.
A point in a region noticeably sparser than its neighbours receives a high
LOF score. Unlike Isolation Forest, LOF is distance-based and sensitive to
feature scale — features measured in thousands (`entries`) would otherwise
dominate fractional features (`rate_deviation`). The feature matrix is
standardised with `StandardScaler` before fitting.

**Key hyperparameters:** `n_neighbors=20`, `contamination=0.03`.

**Result:** **150 anomalies flagged. IF–LOF overlap: 9 events (Jaccard 3.09%).**

The low overlap between the two detectors is expected and informative —
Isolation Forest captures global isolation while LOF captures local density
anomalies. Events flagged by both are the most robust candidates for review.

---

### 5.4 Z-score Detector

**Method.** A parametric single-feature detector applied to `rate_deviation`.
Observations where `|z| ≥ 3` are flagged as anomalies. Unlike the two
preceding methods, Z-score makes a strong distributional assumption
(normality) and operates on a single feature — making it interpretable
but less sensitive to multivariate patterns. It serves as the third
independent vote in the consensus layer.

---

### 5.5 DBSCAN — Density-Based Extension

**Method.** DBSCAN (Ester, Kriegel, Sander, Xu, 1996) groups points into
density-connected clusters and labels everything that does not belong to
any cluster as noise — operationally equivalent to an anomaly. The key
distinction from the three detectors above is that DBSCAN does not require
a fixed `contamination` parameter: it lets the data determine how many
anomalies exist by finding which points are too sparse to form a cluster.
This provides a genuinely independent fourth vote.

**Hyperparameter selection.** `min_samples` was set to `2 × d = 20`
(where d = feature dimensionality, following Ester et al. 1996). `eps`
was selected from the k-distance plot knee, fitted on the same
standardised feature matrix used by LOF.

**Result:** 3 clusters found, **98 noise points flagged (1.97%).**

Pairwise Jaccard agreement with DBSCAN confirms its independence from
the other detectors: DBSCAN ∩ IF = 31.91%, DBSCAN ∩ LOF = 7.36%,
DBSCAN ∩ Z-score = 0.00%. The near-zero overlap with Z-score confirms
that the two methods capture structurally different anomaly patterns.

---

### 5.6 Detector Comparison

| Detector         | Inductive bias                 | Scale-sensitive | Contamination-free | Anomalies flagged  |
| ---------------- | ------------------------------ | --------------- | ------------------ | ------------------ |
| Isolation Forest | Global random partitioning     | No              | No                 | 150                |
| LOF              | Local density vs neighbours    | Yes             | No                 | 150                |
| Z-score          | Parametric single-feature tail | No              | No                 | Fixed by threshold |
| DBSCAN           | Density connectivity           | Yes             | **Yes**      | 98                 |

The four detectors are deliberately diverse in their assumptions.
No single detector is considered ground truth — the anomaly confirmation
logic in the following section combines their votes with business rules
to produce a robust final set of confirmed anomalies.

### Multi-Agent Pipeline

## 6. Multi-Agent Pipeline

### 6.1 Architecture and Design Philosophy

The multi-agent implementation replicates the same anomaly-detection logic
as the classical pipeline, but distributes it across five specialised agents
coordinated by a LangGraph orchestrator. The two implementations are
deliberately equivalent at the mathematical level — the same helpers, the
same detectors, the same business rules — so that any difference in output
reflects architectural behaviour rather than algorithmic divergence.

The key operational distinction is **perimeter flexibility**. The classical
pipeline runs on the full dataset with baselines computed globally. The
multi-agent pipeline accepts a user-defined perimeter at runtime —
any combination of airport, month, year, departure country, document type,
age band — and recomputes all baselines on that subset only. This means
that deviations and traffic multipliers reflect what is normal *for that
specific context*, not for the whole dataset. This is the primary
operational justification for the added architectural complexity.

We adopt the **supervisor pattern**: one deterministic orchestrator
coordinates five specialist worker agents that share a typed state object
(`AnomalyState`). Routing is entirely deterministic — we deliberately do
not let the LLM make routing decisions. With a 4B open-weight model,
deterministic control over node transitions is more reliable and easier
to audit in a safety-critical domain like border control.

---

### 6.2 Agent Overview

| Agent                   | Task                                    | LLM used              |
| ----------------------- | --------------------------------------- | --------------------- |
| Data Agent              | Filter source tables by user perimeter  | No                    |
| Baseline Agent          | Recompute engineered features on subset | Yes — interpretation |
| Outlier Detection Agent | Fit IF, LOF, Z-score on subset          | No                    |
| Risk Profiling Agent    | Apply business rules, confirm anomalies | Yes — prioritisation |
| Report Agent            | Generate Markdown narrative report      | Yes — generation     |

**Design principle — LLM only on linguistic tasks.** The four analytical
agents use deterministic Python (pandas, scikit-learn). The LLM is invoked
only where it adds genuine value: interpreting computed statistics,
prioritising findings, and generating natural-language output. This isolates
the model from tasks where small open-weight models are unreliable (numeric
reasoning, structured data manipulation) and confines it to text generation —
the right call for a safety-critical operational context.

---

### 6.3 Shared State

LangGraph passes a single typed `AnomalyState` dictionary between nodes.
Each agent reads only what it needs and returns a partial update — LangGraph
merges updates into the global state, applying the `add` reducer to `log`
and `errors` so that entries from multiple nodes are concatenated rather
than overwritten. The state has three semantic groups.

**Inputs** — user-supplied perimeter filters, optional natural-language
query, and immutable references to the cleaned source tables (`df_trav`,
`df_alar`).

**Intermediate outputs** — what each specialist produces in pipeline order:
filtered subsets → engineered master → scored master (detector columns) →
flagged master (rule columns) → final Markdown report.

**Tracing** — `log` and `errors`, accumulated across all agents, providing
a full execution trace for debugging and audit.

---

### 6.4 Global Agent Rules

A shared set of behavioural constraints is defined once and prepended to
every agent-specific system prompt via simple string concatenation:
`GLOBAL_AGENT_RULES + agent_specific_prompt`. This pattern — sometimes
called a *meta-prompt* — ensures that task-specific instructions never
override baseline safety constraints.

The rules specify both what the agent must do and what it must never do,
giving explicit negative constraints that reinforce control over model
output. Centralising them in a single variable makes the constraints easier
to audit, update, and extend without modifying individual agent prompts.
Constraints include: never invent numbers, never reference demographic
groups, state uncertainty explicitly, and output only what is requested.

---

### 6.5 Reusable Helpers

To guarantee numerical equivalence between the two pipelines, the classical
logic from sections 8–14 is refactored into three pure, side-effect-free
helper functions that each agent calls internally.

- `engineer_features(df_trav_sub, df_alar_sub)` — wraps sections 8 and 9,
  returns a `df_master`-style subset with all engineered features.
- `fit_detectors(df_master_sub, contamination)` — wraps sections 11–13,
  returns detector labels and severity scores. Includes a size-aware
  fallback: subsets with fewer than 30 rows cannot support ML detectors
  reliably and fall back to Z-score only.
- `apply_business_rules(df_master_sub)` — wraps section 14, returns the
  `confirmed_anomaly` column.

A sanity check confirms delta = 0 confirmed anomalies between the helper
outputs and the classical pipeline on the full dataset, validating
equivalence before any perimeter is applied.

---

### 6.6 Specialist Agents

#### Data Agent

The entry point of the pipeline. It receives the perimeter dictionary and
applies each filter as a logical AND over both source tables, returning
filtered subsets to the state. Filtering logic is entirely deterministic —
no LLM is involved. This is a deliberate choice: structured-dictionary
filtering is more reliable and token-efficient than free-form natural
language parsing with a small model. Keeping the LLM out of this step also
means the perimeter is always auditable and reproducible.

#### Baseline Agent

Receives the filtered tables and calls `engineer_features` on the subset,
recomputing per-airport baselines — `airport_historical_avg_rate`,
`rolling_7d_avg_rate`, `traffic_multiplier` — within the user's perimeter.
This is the core of the multi-agent value proposition: the deviation signals
reflect what is normal for that specific context, not globally.

After the Python computation, the agent invokes the LLM using **role
prompting** — assigning the model an explicit expert identity before the
task instruction — to interpret the computed statistics in 2–3 sentences.

#### Outlier Detection Agent

Applies the three-detector ensemble (Isolation Forest, LOF, Z-score) via
`fit_detectors`. The size-aware fallback — Z-score only for subsets under
30 rows — is precisely the kind of operational concern that motivates
agentic decomposition: the classical pipeline assumed a single large dataset,
while the agentic pipeline must handle arbitrary perimeters including
very small ones. No LLM is invoked at this stage.

#### Risk Profiling Agent

Applies the three business rules via `apply_business_rules` and computes
`confirmed_anomaly = (detector == -1) AND (rules_fired >= 1)`. The agent
then invokes the LLM using role prompting to prioritise the top confirmed
anomalies in 2–3 actionable sentences.

#### Report Agent

The only agent where the LLM performs the primary task. It generates a
structured Markdown report from the flagged dataset.

**Prompt design for a small open-weight model.** A 4B model is not reliable
at numerical reasoning over raw tables. The data is therefore pre-digested
in Python before the LLM call: confirmed anomalies are sorted by
`anomaly_severity_iso`, the top-K are selected, and rendered as a compact
structured text block. The LLM only narrates what is already structured —
the standard pattern for small-model agents: keep computation in code,
keep language in the LLM.

The system prompt follows a four-section output contract (Summary, Top
Anomalies, Baseline Context, Recommended Next Steps) with explicit negative
constraints inherited from `GLOBAL_AGENT_RULES` and reinforced locally:
no invented numbers, no vague language, no speculation.

**Defensive fallback.** If the LLM call fails — server down, timeout, or
malformed output — the agent returns a deterministic Markdown report built
directly from the dataframe. This guarantees the pipeline always produces
output, which is essential for a system that must run end-to-end reliably.

---

### 6.7 Orchestrator

The five agents are assembled into a compiled LangGraph graph with a fixed
execution order: Data → Baseline → Outlier → Risk → Report. A single
conditional edge after the Data Agent short-circuits to Report when the
filtered subset is empty, avoiding wasted computation and misleading
intermediate outputs.

The compiled graph is a `Runnable`: `orchestrator.invoke(initial_state)`
returns the final state with all intermediate outputs and the full
execution trace populated. The graph can be rendered as Mermaid for
visualisation.

---

### 6.8 Sample Output — Tirana, January 2024

The pipeline was tested on perimeter `{"airport_iata": "TIA", "month": 1}`.

**Key findings from the generated report:**
1,216 observations were processed for Tirana in January 2024. 31 confirmed
anomalies were identified. The most severe events occurred on 2024-01-07
and 2024-01-11, where alarm rates reached 10–20x the airport baseline of
0.639. A secondary cluster on 2024-01-03 recorded alarm rates of 9.5x
baseline. The report recommended cross-referencing these dates with known
flight schedules and reviewing staffing levels for those periods.

## Comparison Classique vs multi-agent
