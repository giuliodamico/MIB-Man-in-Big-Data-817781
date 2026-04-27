# Classical vs Multi-Agent: Anomaly detection comparison

Classical Approach - Multi-Agent approach - compare them and analize which one is better in some env condition.

Expected a 'Transit anomaly report' with results of analysis.

ANOMALY MEANINGS - PARALLEL OR ML AND USE THE OUYTPUT TO AGENT AI?

## *README.md –  A) Introduction*

This project investigates two alternative approaches for transit anomaly detection in airport and border-control data: a classical machine learning pipeline and a multi-agent architecture. The main objective is to implement the same anomaly detection system twice and then compare the two solutions in order to understand which approach is more suitable under different operational conditions.

The application scenario is motivated by the need to move from reactive anomaly detection to a more proactive analytical system. Border control authorities and airport operators manage large volumes of passenger transit data every day, including information such as timestamps, gates, routes, nationality, document type, control outcomes, and security alerts. In this context, identifying unusual patterns early can help prevent operational issues and support security monitoring.

In the classical implementation, anomaly detection is performed through a structured pipeline including data preparation, feature engineering, historical baseline construction, anomaly detection algorithms, and rule-based post-processing. In the multi-agent implementation, the same logic is distributed across specialized agents, each responsible for a specific task such as querying data, building historical baselines, detecting outliers, applying risk rules, and generating a final report.

The goal of the project is not only to detect anomalies, but also to provide a comparative analysis of the two paradigms in terms of modularity, interpretability, scalability, flexibility, and practical usability. The final output is a transit anomaly report highlighting suspicious patterns and supporting the discussion on the advantages and limitations of each approach.

#### 1. Understand "Anomaly" definition

Understand which kind of anomaly is required; most probably `route + gate + month` (aggregated pattern).

#### 2. Dataset analysis

Translating columns

Columns, granularity of data, n. record, time dimesion, Csv, Categorical variable with high cardinality, alert columns.

## *EDA.py*

Dataset shape, variable type, missing values, cat. cardinality, distribution, time analysis.

Graph of cardinality - A

ANNO_PARTENZA analysis - G

comment on motivo allarmi - G

secure redundant columns - G

What G-A has done in Travelers Dataset :
Listing all columns with missing values : Nationality, document_type, gender, airline, flight_number, control_outcome, operator note, risk code

Check cardinality for all columns to be sure that it's correct and clean

Check redundant columns to fix missing value easily
=> Column that we keep : Typo documento (instead of document type); Fascia Eta (instead of age_group); nationality_3 (instead of nationality); flight number (instead of flight_number); airline% (instead of airline)

Fixing missing values for columns needed

Standardization of date colums

EDA must be the same for both approach.

#### 3. Draft of both pipeline

Classic:
`input data -> preprocessing -> feature engineering -> baseline -> anomaly detection -> rule-based scoring -> final report`

Multi-agent:
`Data Agent -> Baseline Agent -> Outlier Detection Agent -> Risk Profiling Agent -> Report Agent`

Required in section 2 of README.md

#### 4. Fundamental feature

Il PDF suggest transit volumes, non-conformity rates, alert rates for gate/route/nationality/document type, deviation from history mean, % variation from baseline, seasonal feature.

#### 5. Evals methods (**Section 3 Experimental Design**)

qualità e coerenza degli anomaly scores, interpretabilità dei risultati, modularità del sistema, facilità di estensione, costo computazionale, tempo di sviluppo, qualità del report finale.

#### 6. Classical operating baseline

*Uploading data -> Cleaning -> Aggregation -> Historical baseline -> Simple anomaly detection method -> Ranking anomaly*

Multi-Agent has value if we know the steps of the process that each agent should do. Otherwise we're orchestrating nothing.

#### 7. Then, mutuating classical pipeline in agents

When the classic baseline works, we can “agentize” it:

* Agent to upload and prepare,
* Agent build baseline,
* Agent compute anomaly scores,
* Agent apply business rules,
* Agent write final report with natural language.

#### ...

#### Feature engineering done by G-A

Project Update: Data Pipeline & Anomaly Detection MVP
We have successfully completed the end-to-end data preparation and trained our first Machine Learning Proof of Concept (POC) for anomaly detection. Here is the structured breakdown of the pipeline we built:

1. Data Cleaning & Rich Aggregation
   Missing Values Handled: Executed a "brutal but effective" cleaning phase (median imputation for numbers, dropping rows with missing critical dates, dropping unusable text columns like operator_notes).

The "Rich Pivot" Aggregation: To avoid the "Cartesian Product" trap (which would artificially multiply passenger volumes during the merge), we aggregated the ALARMS dataset to a daily level per airport.

Preserving Signal: Instead of just doing a raw count of alarms, we used One-Hot Encoding before aggregating. This allowed us to keep critical business intelligence (like RISK_FLAG_HIGH and ALARM_REASON) while keeping the math safe for the ML algorithm. We then successfully merged TRAVELERS and ALARMS into a single, robust df_master.

2. Feature Engineering (Building the Baselines)
   We moved from raw data to intelligent context. To help the algorithm understand what "normal" looks like, we engineered new features:

traffic_multiplier (Volume Baseline): Measures if today's passenger volume is unusually high or low compared to the airport's historical average.

rate_deviation (Risk Baseline): Measures how far today's alarm rate deviates from the usual alarm rate for that specific airport.

rolling_7d_avg_rate (Time Series Baseline): Captures the recent trend over the last 7 days to give the model temporal awareness.

Temporal Context: Added simple flags like is_weekend.

3. Dimensionality Reduction
   To avoid the "Curse of Dimensionality" and multicollinearity (which blinds the model), we aggressively filtered our 48 available columns down to the 7 most mathematically informative features. We removed all raw IDs, text, and overlapping metrics.
4. Machine Learning: Unsupervised Anomaly Detection
   Primary Model (Isolation Forest): We deployed an Isolation Forest to isolate the top 3% most abnormal flights based on the engineered baselines. It successfully flagged events with extreme combinations (e.g., extremely low passenger volume but massive alarm spikes).

Model Validation (LOF): To prove the robustness of our pipeline, we ran a second algorithm, Local Outlier Factor (LOF).

The Consensus: We cross-referenced both models and found 12 undeniable anomalies that were flagged by both AIs.

### Question for Q&A

Questions for the Data Provider (Business & Operations)

1. The "Entries vs. Alarms" Dilemma (Data Quality vs. True Risk)

"In our dataset, we found instances where the number of daily alarms far exceeded the number of passengers (e.g., 20 alarms for 1 passenger). Should we treat this strictly as a data entry error and clean it out, or is it a valid operational scenario where a single high-risk individual triggers multiple independent database alerts?"

2. Time Series Seasonality

"To give the algorithm temporal context, we built a 7-day rolling average baseline. Based on real-world border control operations, is a weekly cycle the most accurate reflection of traffic patterns, or should we look at monthly or seasonal cycles instead?"

3. Defining "Actionability"

"Our Machine Learning models successfully flagged a top list of highly anomalous days. From an operational standpoint, how would your team ideally consume this information? Do you need a daily automated dashboard, or retroactive monthly audits to spot systemic issues?"

Questions for the TA (Methodology & Machine Learning)

1. Validating Unsupervised Models

"Since we do not have labeled data to measure accuracy or recall, we used a 'Consensus Approach'—running both Isolation Forest and Local Outlier Factor (LOF) and focusing on the events flagged by both. Do you agree with this validation strategy for an MVP, or is there another unsupervised evaluation metric you prefer?"

2. Feature Selection & The Curse of Dimensionality

"To avoid the curse of dimensionality with the Isolation Forest, we aggressively reduced 48 raw columns down to 5-7 engineered baselines (traffic multipliers, rate deviations, risk scores). Are you satisfied with this level of dimensionality reduction, or do you think we aggregated away too much granular signal?"

3. Target Encoding for Categorical Data

"To include 'Nationality' without exploding the dataset with One-Hot Encoding (which Isolation Forest handles poorly), we used Target Encoding to replace countries with their 'Historical Risk Score'. Is this considered a best practice for this specific algorithm in your experience?"

### strategy for AI agent

For the multi-agent architecture on n8n, our strategy is to leverage the existing Python pipeline. The LLM agents will not perform data cleaning or heavy mathematical computations, as this leads to hallucinations and is computationally expensive. Instead, we will provide our df_master dataset (which already contains the mathematical deviation scores) to the n8n agents. > The Agents’ mission will be purely analytical: Agent 1 will read the deviations, Agent 2 will apply business rules to prioritize risks, and Agent 3 will generate the final report in natural language for field teams. Do you agree with this separation of roles between Python (mathematics) and LLMs (cognitive reasoning)



User natural-language query

    │

    ▼

   ┌──────────────────────┐

   │     ORCHESTRATOR     │ ◄── parse perimeter, route nodes,

   │   (LangGraph Graph)  │     handle errors, aggregate state

   └──────────┬───────────┘

    │

   ┌──────────▼───────────┐

   │   1. Data Agent      │ ── filter (df_trav, df_alar) by perimeter

   └──────────┬───────────┘

    ▼

   ┌──────────────────────┐

   │   2. Baseline Agent  │ ── feature engineering on subset

   └──────────┬───────────┘     (rolling means, per-airport baselines)

    ▼

   ┌──────────────────────┐

   │ 3. Outlier Detection │ ── IF + LOF + Z-score on subset

   │       Agent          │     (with size-aware fallback)

   └──────────┬───────────┘

    ▼

   ┌──────────────────────┐

   │ 4. Risk Profiling    │ ── apply 3 business rules + confirmed

   │       Agent          │

   └──────────┬───────────┘

    ▼

   ┌──────────────────────┐

   │   5. Report Agent    │ ── Gemma generates Markdown report

   └──────────┬───────────┘

    ▼

    Transit Anomaly Report
