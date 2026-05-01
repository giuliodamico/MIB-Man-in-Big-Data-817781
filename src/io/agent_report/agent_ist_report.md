# Transit Anomaly Report

## Perimeter
{"departure_iata": "IST"}

## Summary
- Routes analysed: **12**
- Consensus anomalies (≥ 2/4 detectors): **1**
- Risk distribution: {'LOW': 1}

## Detector counts
{
  "if_anomaly": 1,
  "lof_anomaly": 1,
  "dbscan_anomaly": 0,
  "z_anomaly": 2,
  "consensus_>=2": 1,
  "consensus_>=3": 0
}

## Top-10 ranked anomalies
|   rank | route   | DEPARTURE_COUNTRY   |   alert_rate | ci95_str       |   tot_investigated |   absolute_alarms |   anomaly_votes | risk_level   |   priority_score | quality_note   |
|-------:|:--------|:--------------------|-------------:|:---------------|-------------------:|------------------:|----------------:|:-------------|-----------------:|:---------------|
|      1 | IST→FCO | Turchia             |     0.382353 | [23.9%, 55.0%] |                 34 |                13 |               2 | LOW          |           0.1582 | ok             |

## Interpretation
The ranked anomalies should be treated as an investigation queue, not as confirmed incidents. Higher-rank items combine high alert rate with meaningful volume and tight confidence intervals.
