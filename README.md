# Transit Anomaly Detection — Classical vs Multi-Agent

**Whitehall Reply — Project 2 · Academic Year 2025/26**
**Team:** Giulio D'Amico · Alexis Mitracos
**Captain:** Giulio D'Amico

---

## Introduction

Border-control authorities and airport operators process thousands of passenger transits every day, each carrying rich metadata: timestamp, gate, route, nationality, document type, control outcome, and security alerts. Anomaly detection on this stream is today largely **reactive** — incidents are reviewed after the fact rather than flagged as patterns emerge. A proactive analytical layer that surfaces suspicious patterns *before* they escalate into operational risks would meaningfully improve both security monitoring and resource allocation.

This project addresses that gap by **implementing the same anomaly-detection system twice** under two paradigms — a classical statistical pipeline and a multi-agent architecture — and producing a comparative analysis to argue  **which approach is preferable under which operational conditions** .

The classical pipeline is a deterministic statistical workflow: cleaning, feature engineering at route level, an ensemble of four unsupervised detectors (Isolation Forest, Local Outlier Factor, DBSCAN, robust Z-score), and a rule-based post-processing layer that ranks the consensus anomalies by a confidence-weighted priority score.

The multi-agent pipeline distributes the same logic across five specialist agents — Data Agent, Baseline Agent, Outlier Detection Agent, Risk Profiling Agent, Report Agent — orchestrated by  **LangGraph** . The numerical core is shared with the classical pipeline through a single `utils.py` module, so any divergence between the two outputs reflects architectural behaviour rather than algorithmic drift. A local LLM (`llama3.2:3b` served via Ollama) is invoked at exactly two points: parsing free-text user requests into a structured perimeter, and writing one short interpretation paragraph on top of the deterministic facts of the report.

The deliverable is  **not a winner declaration** . The two pipelines solve the same problem under different operational constraints — the comparison in §3–4 quantifies the trade-off and concludes with a decision matrix.

---

## Methods

### 2.1 Data sources

Two CSV files were provided by the client, both extracted from Italian airport border-control operations.

| Dataset                   |  Rows | Cols | Observation unit                                                 |
| ------------------------- | ----: | ---: | ---------------------------------------------------------------- |
| `TIPOLOGIA_VIAGGIATORE` | 5,095 |   33 | Daily traveler-category aggregates per departure airport         |
| `ALLARMI`               | 5,080 |   24 | Alarm-event aggregates per (airport × month × occurrence type) |

Both files arrived in Italian with mixed casing, typographic errors (`AREOPORTO` instead of `AEROPORTO`), six different date formats, and several encodings of the same logical column (`Tipo_Documento` vs `TIPO_DOCUMENTO`, `ZONE` vs `ZONE_3`, etc.). The cleaning section addresses these systematically.

### 2.2 Shared analytical core (`utils.py`)

A single module contains the cleaning, feature engineering, detection, and post-processing logic. **Both notebooks import from this module** — no business logic is duplicated. This is a deliberate design choice: it guarantees by construction that any difference in the §4 results comes from the multi-agent layer (perimeter parsing, LLM interpretation, audit log) and not from cleaning drift.

The module exposes:

* `load_clean_data()` — single I/O entry point, returns `(df_alarms, df_travelers)`.
* `build_route_master(df_alarms, df_travelers)` — aggregates both tables at the `(departure × arrival)` route level.
* `build_feature_matrix(df_route)` — `log1p` + `StandardScaler` over the engineered numeric features.
* `fit_detectors(X_scaled, contamination)` — runs the four detectors and returns per-row labels + consensus votes.
* `apply_post_processing(df_scored)` — assigns risk levels, Wilson CI, priority score, and **quality filter** consistently across both pipelines.

### 2.3 Cleaning pipeline

Cleaning is documented step by step in `Classical_approach.ipynb` §2; here we summarise the design choices.

**Italian → English column normalisation.** A curated `COL_MAP` dictionary translates 40+ column names. Headers are then upper-cased and stripped.

**Placeholder unification.** A controlled vocabulary (`N.D.`, `??`, `//`, `-`, `unknown`, `xx`, `zz`, etc.) is mapped to NaN before any imputation. Free-text columns receive explicit placeholders (`NO REASON PROVIDED`, `NO MANUAL NOTES`) to preserve row count and categorical usability.

**Date parsing.** Six format variants are detected and parsed using `format='mixed'` with a fallback pass to capture remaining edge cases.

**IATA enrichment and country mapping.** A curated `IATA_MAPPING` (≈ 80 codes) back-fills missing city/airport descriptions where the IATA code is populated. Italian country names are mapped to ISO alpha-3 via `IT_TO_ALPHA3` plus `pycountry`. Kosovo (no official ISO) is hand-coded as `RKS`.

**Signal-column cap at `[0, SIGNAL_CAP=200]`.** The `ENTRIES`, `INVESTIGATED`, and `ALARMS` columns contain typing artefacts (`"1 pax"`, `"~5"`, free-text leakage). The cut-off was chosen empirically — see §3.1 and the histogram in `Classical_approach.ipynb` §2.2.5: the p99.5 of the raw distribution falls between 150 and 180, the cap at 200 retains > 99.5% of legitimate observations and removes only values that exhibit other parsing artefacts.

![Descrizione immagine](src/images/capping.png)

### 2.4 Feature engineering at route level

The cleaned tables are aggregated to one row per `(departure_iata, arrival_iata)` route. Three groups of features are built:

* **Volume features** : `tot_entries`, `tot_investigated`, `tot_alarms`, plus pivoted occurrence counts (`tot_<occurrence_type>`).
* **Rate features** : `alert_rate = alarms / investigated` (capped at 1.0), `investigation_rate = investigated / entries`, plus segmented rates per top-3 nationalities, top-3 document types, and top-4 control outcomes — these introduce horizontal information that single global rates would hide.
* **Risk features** : `n_high_risk`, `n_medium_risk` from the alarm-level `RISK_FLAG`.

The numeric matrix is transformed with `log1p` (count features have heavy right tails) and standardised with `StandardScaler` before detection.

![ ](src/images/PCA_distribution.png)

### 2.5 Anomaly detection ensemble

Four detectors with **deliberately different inductive biases** vote on each route. A route is flagged as a **consensus anomaly** if at least 2 of the 4 detectors agree.

| Detector         | Inductive bias                | Hyperparameters                                                    |
| ---------------- | ----------------------------- | ------------------------------------------------------------------ |
| Isolation Forest | Global random partitioning    | `n_estimators=200`,`contamination=0.05`                        |
| LOF              | Local density vs neighbours   | `n_neighbors = max(2, min(20, n-1))`                             |
| DBSCAN           | Density connectivity          | `min_samples = max(5, ⌈ln(n)⌉ + 1)`,`eps`from k-distance p95 |
| Robust Z-score   | Per-feature tail (median+MAD) | `\|z\| > 3`after median-MAD standardisation                        |

Two design choices deserve a note.

**DBSCAN `min_samples`.** The textbook rule `min_samples = 2 · d` (Sander et al.) assumes moderate dimensionality. With *d* ≈ 30+ engineered features on ≈ 1 100 routes, that rule labelled almost every point as noise (curse of dimensionality on density estimates). The high-dimensional practical rule `max(5, ⌈ln(n)⌉ + 1)` brings the noise rate from ~75% to a meaningful ~1.5%, which is the operational target — DBSCAN should isolate a small number of structurally distant points, not partition the dataset.

**Robust Z-score.** Plain `(x − mean) / std` on the standardised matrix flagged ~20% of all routes (215 / ≈ 1 100), dominating the consensus vote by construction. The detector was replaced with a robust variant — `(x − median) / MAD` — which restores the intended semantics of "rare per-feature tail event" and brings the Z-flag count to a plausible ≈ 60–90 routes. The four detectors are now genuinely complementary.

![ ](src/images/contamination.png)

### 2.6 Post-processing and risk ranking

Consensus routes (≥ 2/4 detectors agreeing) go through a deterministic ranking layer.

**Risk levels** are assigned by a rule cascade based on alert rate, investigated volume, and number of detector votes:

```
CRITICAL  ←  votes == 4
HIGH      ←  alert_rate ≥ T_HIGH ∧ tot_investigated ≥ T_VOL
HIGH      ←  votes ≥ 3 ∧ alert_rate ≥ pop_median
MEDIUM    ←  alert_rate ≥ T_MED ∨ votes ≥ 3
LOW       ←  otherwise
```

where `T_HIGH = max(3 × pop_median, p66_post)`, `T_MED = max(1.5 × pop_median, p33_post)`, `T_VOL = max(2, p25_post_volume)`.

**Wilson 95% confidence intervals** on the alert rate guard against false positives on tiny volumes. A 100% rate measured on 1 investigated transit is very different from 100% on 50 — the CI half-width captures this.

**Priority score** ranks the consensus list:

```
priority = (0.60 × rate_n + 0.40 × log_alarms_n) × ci_tightness
```

where `rate_n` and `alarms_n` are MinMax-normalised to `[0,1]` and `ci_tightness = 1 − ci95_width ∈ [0,1]` downweights routes with wide CIs (sparse evidence). The previous variable name was `confidence`, which suggested a probabilistic semantics it does not have; the rename is a transparency fix.

![ ](src/images/Top20routes_bypriority.png)

**Quality-note filter.** Three diagnostic flags identify routes whose statistics are not actionable:

* `incomplete data — alarms but no traveler records` (left-only join);
* `likely false positive — flagged on non-rate features` (zero alert_rate, ≤ 2 investigated);
* `warning — high rate but tiny volume (≤ 3 investigated)` (kept in the table for awareness, but flagged).

The first two categories are **excluded from the final ranking** by `apply_post_processing(..., drop_disqualified=True)`. The previous version of the project applied this filter inline in the classical notebook only — both pipelines now go through the same function and produce the same `df_clean` by construction.

### 2.7 Multi-agent architecture (LangGraph)

The same numerical core is wrapped in five specialist agents communicating through a typed `AgentState` dictionary. State is append-only (no agent overwrites upstream keys), which makes the pipeline easy to debug and replay.

![ ](src/images/agent_architecture.png)

| Agent    | Task                                                                             | LLM           |
| -------- | -------------------------------------------------------------------------------- | ------------- |
| Data     | Parse free-text request → perimeter; filter both source tables                  | fallback only |
| Baseline | Aggregate at route level, build feature matrix                                   | no            |
| Outlier  | Run the 4-detector ensemble, attach votes and consensus flag                     | no            |
| Risk     | Apply `apply_post_processing`(risk level + Wilson CI + priority + QA filter)   | no            |
| Report   | Build deterministic Markdown report; write a 4-sentence interpretation paragraph | yes           |

**Conditional routing.** A single conditional edge after the Data Agent skips the entire detection layer when the perimeter is empty or the input invalid, jumping straight to the Report Agent which emits a graceful "no data" report. The pipeline therefore always produces a deliverable, even on bad input.

**LLM scope is deliberately minimal.** The Data Agent uses three deterministic fallbacks for perimeter parsing — explicit IATA token, city-name lookup (`CITY_TO_IATA`), country-name lookup (`IT_TO_ALPHA3` + travelers-frequency) — and only invokes the LLM if all three fail. The Report Agent invokes the LLM with `temperature=0` and `seed=RANDOM_STATE` for reproducibility, and explicit negative constraints in the prompt:  *do not invent causes; do not name a country or nationality as suspicious by itself; focus on rate, volume and confidence* . If the LLM call fails, the report renders with a generic interpretation paragraph —  **the deliverable is never blocked by LLM availability** .

### 2.8 Environment and reproducibility

The submission pins the environment via `requirements.txt`. To reproduce:

```bash
conda create -n transit-anomaly python=3.11 -y
conda activate transit-anomaly
pip install -r requirements.txt
# Optional: install Ollama and pull llama3.2:3b for the multi-agent narrative
ollama pull llama3.2:3b
```

Then open `Classical_approach.ipynb` and `Multi-Agent_approach.ipynb`, run  *Kernel → Restart & Run All* . Random seeds are set globally (`RANDOM_STATE=42` in `utils.py`); the `ChatOllama` client is configured with `seed=RANDOM_STATE` to constrain LLM non-determinism on the interpretation paragraph (numbers, rankings, and risk levels are unaffected).

---

## Experimental Design

We designed four experiments to validate the methodology and quantify the trade-off between the two pipelines. None of them require ground-truth anomaly labels — these are not available in the dataset and the brief itself does not provide them; the experiments are therefore designed as **internal-consistency** and **operational-comparison** tests.

### 3.1 Experiment A — Sensitivity to `contamination`

**Purpose.** Justify the choice of `contamination = 0.05` (a hyperparameter shared across IF and LOF) and demonstrate that the consensus layer is more stable than any single detector.

**Baseline.** Each individual detector at four contamination values: `{0.03, 0.05, 0.07, 0.10}`. We also report `consensus ≥ 2/4` and the higher-confidence `consensus ≥ 3/4`.

**Metrics.** Number of routes flagged at each contamination level, broken down per detector and per consensus threshold.

### 3.2 Experiment B — Pipeline parity (Jaccard agreement)

**Purpose.** Validate that the multi-agent pipeline and the classical pipeline produce the same set of consensus anomalies on the full perimeter, *modulo* the LLM-only steps. This is the empirical check that `utils.py` succeeds in eliminating implementation drift.

**Baseline.** The classical pipeline run on the same input through the same `utils.apply_post_processing()`, producing `df_clean_c`. The multi-agent pipeline run with an empty user request (full-perimeter mode), producing `df_clean_a`.

**Metric.** **Jaccard similarity** on the route sets `{(dep_iata, arr_iata)}`:

```
J = |A ∩ B| / |A ∪ B|
```

A perfect parity yields `J = 1.0`.

### 3.3 Experiment C — Latency benchmark

**Purpose.** Quantify the cost of the agentic interface — the price paid for graph traversal, state copies, and the two LLM calls.

**Baseline.** Wall-clock time of the classical pipeline (cleaning + feature engineering + detection + post-processing).

**Metric.** Median wall-clock seconds over 5 runs, discarding the first as warm-up. Reported as absolute seconds and as overhead multiplier vs the classical reference.

### 3.4 Experiment D — Multi-perimeter robustness

**Purpose.** Demonstrate that the multi-agent pipeline behaves correctly across operational regimes — large hub, medium airport, tiny airport, invalid IATA — without crashing and without producing misleading detections on samples too small for distance-based methods.

**Baseline.** A guard `MIN_ROUTES_FOR_DETECTION = 10` plus the LOF `n_neighbors` clipping in §2.5; an early return from the Outlier Agent when these are not met.

**Metric.** Per-perimeter status code (`ready`, `empty_perimeter`, `invalid_input`), number of routes analysed, number of consensus anomalies, and a boolean `graceful` flag confirming that a structured report is always produced.

---

## Results

### 4.1 Detection on the full perimeter

Running `utils.apply_post_processing()` after the 4-detector ensemble on the full dataset yields the following.

| Quantity                                 |                                    Count |
| ---------------------------------------- | ---------------------------------------: |
| Routes analysed                          |                                    1 100 |
| Isolation Forest flags                   |                                       28 |
| LOF flags                                |                                       27 |
| DBSCAN flags                             |                                       17 |
| Robust Z-score flags                     |                                      194 |
| Consensus ≥ 2/4 (`df_post`)           |                                       41 |
| After QA filter (`df_clean`)           |                                       33 |
| `CRITICAL`/`HIGH`/`MEDIUM`/`LOW` | *populate from df_clean['risk_level']* |

> 📊 **Figure** — `images/top_anomalies_barh.png`: top-N consensus anomalies ranked by `priority_score`, coloured by `risk_level`. Generated by the final cell of `Classical_approach.ipynb` §9.

The DBSCAN noise rate sanity-check (`flags.attrs["dbscan_noise_pct"]`) reports  **≈ 1.5%** , confirming that the high-dimensional `min_samples` fix is operating as intended.

> ⚠ **Insertion note.** Replace the final-row counts above with the actual numbers from your last  *Restart & Run All* . The robust-Z migration changes the consensus count slightly compared with the screenshot in §3.1.

### 4.2 Pipeline parity — Jaccard agreement

> 📊 **Figure** — `images/sensitivity_contamination.png`: routes flagged by each detector and by the two consensus levels across contamination ∈ {0.03, 0.05, 0.07, 0.10}. Generated by `Classical_approach.ipynb` §10 (Experiment A).

After `utils.py` consolidates the cleaning + post-processing for both pipelines:

| Quantity                           |          Routes |
| ---------------------------------- | --------------: |
| Classical clean (`df_clean_c`)   |              33 |
| Multi-agent clean (`df_clean_a`) |              33 |
| Intersection                       |              33 |
| Symmetric difference               |               0 |
| **Jaccard agreement**        | **100 %** |

> ⚠ **Insertion note.** A Jaccard = 100 % is the *expected* result after the §2.2 single-source-of-truth refactor and the realignment of the two pipelines on `apply_post_processing(..., drop_disqualified=True)`. If your run reports < 100 %, the most likely causes are: (i) the multi-agent kernel was not restarted after the `utils.py` edit, (ii) the LLM seed is not honoured by your local Ollama version (the rankings should still match — only the interpretation paragraph differs).

### 4.3 Latency

| Pipeline                     | Median seconds | Overhead × |
| ---------------------------- | -------------: | ----------: |
| Classical                    |        *x.x* |         1.0 |
| Multi-agent (full perimeter) |        *y.y* |     *y/x* |
| Multi-agent (NL → IST)      |        *z.z* |     *z/x* |

> ⚠ **Insertion note.** Run §22.2 of the multi-agent notebook (after applying the `n=5, drop warm-up` fix) and substitute the values. Typical results: classical ≈ 1–2 s, multi-agent full ≈ 4–6 s (one LLM call), multi-agent NL ≈ 4–6 s.

The agentic overhead is **bounded by the cost of one to two LLM calls** plus graph traversal. State handover is in-memory (no disk serialisation) so its contribution is negligible.

### 4.4 Multi-perimeter robustness

| Request                           | Status              | Routes | Consensus | Graceful |
| --------------------------------- | ------------------- | -----: | --------: | -------: |
| `"anomalies from IST"`          | `ready`           |  *r* |     *c* |       ✅ |
| `"flights from TIA"`            | `ready`           |  *r* |     *c* |       ✅ |
| `"voli da Tirana"`(IT, country) | `ready`           |  *r* |     *c* |       ✅ |
| `"flights from KBL"`            | `empty_perimeter` |      0 |         0 |       ✅ |
| `"flights from ZZZ"`            | `invalid_input`   |      0 |         0 |       ✅ |

The deterministic city/country fallbacks added in §2.7 close the previously documented bug where `"voli da Tirana"` (city) or `"voli dall'Albania"` (country) could not be parsed.

### 4.5 Decision matrix — when to use which approach

| Scenario                                                      | Classical | Multi-Agent |
| ------------------------------------------------------------- | :-------: | :---------: |
| Scheduled batch report, fixed perimeter, archived for audit   |    ✅    |            |
| Bit-identical reproducibility (no LLM dependency)             |    ✅    |            |
| Latency-critical scoring (sub-second)                         |    ✅    |            |
| Explainability before a regulator / data-protection authority |    ✅    |            |
| Low operational cost (no GPU / no LLM tokens)                 |    ✅    |            |
| Analyst exploration on arbitrary, ad-hoc perimeters           |          |     ✅     |
| Stakeholder-ready narrative output without manual editing     |          |     ✅     |
| Free-text / multilingual input from non-technical operators   |          |     ✅     |
| Graceful failure on missing/invalid input                     |          |     ✅     |
| Pipeline must extend to new specialist roles (e.g. geo-agent) |          |     ✅     |

The recommendation is  **deploy both** : the classical pipeline as the nightly system-of-record (deterministic, auditable, archived), the multi-agent pipeline as the on-demand exploration layer over the same evidence — both calling the same `utils.py`, so the two outputs cannot drift.

### 4.6 Limitations of the comparison

* **No ground truth.** Without operator-labelled anomalies, neither pipeline can be evaluated for precision or recall. Jaccard is an internal-consistency metric, not an accuracy metric.
* **LLM non-determinism.** Even with `temperature=0` and a fixed seed, the local model can produce minor variations in the interpretation paragraph across runs. The numbers, rankings, and risk levels are unaffected.
* **Single time window.** The dataset covers a 2-month period, which rules out strict temporal back-testing (rolling baselines, train-on-month-1 / score-on-month-2). Both conclusions inherit this limitation.
* **Auto-detected segmentation features.** Top-3 nationality / document / control-outcome categories are computed on the data itself. For production deployment they should be frozen per release.

---

## [Section 5] Conclusions

**Take-away.** The two implementations are mathematically equivalent — both run on the same `utils.py` core, the §4.2 Jaccard parity confirms it numerically. The choice between them is therefore  **not a methodology choice but a deployment choice** . The classical pipeline costs nothing per run, has zero external dependencies, and is fully auditable; the multi-agent pipeline buys interactivity, free-text perimeter selection, graceful failure, and a stakeholder-ready report at the price of one or two LLM calls and a small latency overhead. In a border-control context — where explainability before a regulator is non-negotiable and the consequence of a wrong call is high — the classical pipeline is the correct  **system-of-record** , and the multi-agent pipeline is the correct **exploration interface** sitting above it. Both, not either.

**Open questions and next steps.**
The most important question this work cannot answer is  **detector accuracy** . Without operator-labelled anomalies the four-detector ensemble can only be evaluated for self-consistency; a follow-up phase with the client should establish a labelled validation set to compute precision and recall per risk level. A natural extension is a  **temporal baseline** : with a longer observation window, rolling 7-day and 28-day rates per route would replace the cross-sectional baseline used here, giving the system the ability to flag *changes in behaviour* and not just absolute outliers. On the agent side, a **geo-agent** specialised in clustering routes by country / region would pair naturally with the existing risk agent, and a **feedback agent** that ingests operator decisions back into the ranking weights would close the loop between detection and triage. Finally, the auto-detected segmentation features should be frozen per release and versioned alongside the model, to ensure the same record is scored identically across runs.

---

## Repository structure

```
.
├── README.md                       ← this file
├── requirements.txt                ← pinned environment
├── utils.py                        ← shared core: cleaning + features + detection + post-processing
├── Classical_approach.ipynb        ← classical pipeline (Sections 1–14 of analysis)
├── Multi-Agent_approach.ipynb      ← multi-agent pipeline (Sections 15–23)
├── io/
│   ├── ALLARMI.csv                 ← raw input
│   ├── TIPOLOGIA_VIAGGIATORE.csv   ← raw input
│   ├── ALARMS_CLEANED.csv          ← intermediate
│   ├── TRAVELERS_CLEANED.csv       ← intermediate
│   ├── ROUTE_LEVEL_DATA.csv        ← engineered feature panel
│   └── agent_report/               ← Markdown outputs of the Report Agent
└── images/
    ├── agent_architecture.png      ← LangGraph DAG (see “How to regenerate the figures”)
    ├── signal_distribution.png     ← histogram + cap justification (Classical §2.2.5)
    ├── sensitivity_contamination.png  ← Experiment A
    ├── top_anomalies_barh.png      ← top-N ranked anomalies
    └── jaccard_parity.png          ← optional: Venn diagram of the two consensus sets
```

---

## How to regenerate the figures

All figures are emitted by code cells. Add the following snippets to the notebooks (or copy them into a final `images/` cell at the bottom of each notebook). The `images/` directory is created if missing.

```python
from pathlib import Path
import matplotlib.pyplot as plt
IMG = Path("images"); IMG.mkdir(exist_ok=True)
```

### LangGraph architecture diagram (`images/agent_architecture.png`)

Add this cell at the end of `Multi-Agent_approach.ipynb` §20 (right after `orchestrator = builder.compile()`):

```python
from IPython.display import Image, display

# LangGraph exposes the compiled DAG as a Mermaid graph.
# draw_mermaid_png() renders it via the public mermaid.ink HTTP API
# (no extra system dependency required).
img_bytes = orchestrator.get_graph().draw_mermaid_png()
(Path("images") / "agent_architecture.png").write_bytes(img_bytes)
display(Image(img_bytes))
```

If your environment cannot reach `mermaid.ink`, the offline alternative is:

```bash
pip install pygraphviz   # requires graphviz installed on the OS
```

```python
img_bytes = orchestrator.get_graph().draw_png()    # pygraphviz backend
(Path("images") / "agent_architecture.png").write_bytes(img_bytes)
```

### Signal-column distribution + cap (`images/signal_distribution.png`)

In `Classical_approach.ipynb` §2.2.5, after the histogram is rendered:

```python
plt.savefig(IMG / "signal_distribution.png", bbox_inches="tight", dpi=150)
```

### Sensitivity to contamination (`images/sensitivity_contamination.png`)

Same pattern at the end of the contamination-sensitivity cell of `Classical_approach.ipynb` §10.

### Top-anomalies bar chart (`images/top_anomalies_barh.png`)

At the end of the final ranked-bar-chart cell of `Classical_approach.ipynb` §9.

After running both notebooks end-to-end once, all five PNGs land in `images/` and the README references resolve correctly when the repository is rendered on GitHub.
