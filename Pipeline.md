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

EDA must be the same for same approach.

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
