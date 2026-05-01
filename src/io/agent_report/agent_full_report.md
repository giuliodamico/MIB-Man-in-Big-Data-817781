# Transit Anomaly Report

## Perimeter
{}

## Summary
- Routes analysed: **557**
- Consensus anomalies (≥ 2/4 detectors): **33**
- Risk distribution: {'HIGH': 15, 'MEDIUM': 12, 'LOW': 5, 'CRITICAL': 1}

## Detector counts
{
  "if_anomaly": 28,
  "lof_anomaly": 27,
  "dbscan_anomaly": 17,
  "z_anomaly": 194,
  "consensus_>=2": 41,
  "consensus_>=3": 9
}

## Top-10 ranked anomalies
|   rank | route   | DEPARTURE_COUNTRY   |   alert_rate | ci95_str        |   tot_investigated |   absolute_alarms |   anomaly_votes | risk_level   |   priority_score | quality_note   |
|-------:|:--------|:--------------------|-------------:|:----------------|-------------------:|------------------:|----------------:|:-------------|-----------------:|:---------------|
|      1 | TIA→BGY | Albania             |     0.188378 | [18.4%, 19.3%]  |              24403 |              4597 |               3 | HIGH         |           0.508  | ok             |
|      2 | TIA→BLQ | Albania             |     0.20016  | [19.5%, 20.6%]  |              20029 |              4009 |               3 | HIGH         |           0.5079 | ok             |
|      3 | TIA→PSA | Albania             |     0.177457 | [17.2%, 18.3%]  |              20929 |              3714 |               2 | MEDIUM       |           0.4912 | ok             |
|      4 | TIA→MXP | Albania             |     0.176944 | [17.1%, 18.3%]  |              18147 |              3211 |               2 | LOW          |           0.4837 | ok             |
|      5 | IST→CTA | Turchia             |     1        | [67.6%, 100.0%] |                  8 |                 8 |               2 | HIGH         |           0.4758 | ok             |
|      6 | TIA→GOA | Albania             |     0.24334  | [23.2%, 25.5%]  |               5330 |              1297 |               2 | MEDIUM       |           0.4748 | ok             |
|      7 | TIA→AOI | Albania             |     0.235972 | [22.4%, 24.8%]  |               4598 |              1085 |               2 | MEDIUM       |           0.4615 | ok             |
|      8 | TIA→TSF | Albania             |     0.170102 | [16.4%, 17.7%]  |              12469 |              2121 |               2 | LOW          |           0.4592 | ok             |
|      9 | TIA→TRN | Albania             |     0.186199 | [17.8%, 19.5%]  |               8405 |              1565 |               2 | MEDIUM       |           0.453  | ok             |
|     10 | SAW→BGY | Turchia             |     0.509804 | [43.1%, 58.8%]  |                153 |                78 |               2 | HIGH         |           0.4328 | ok             |

## Interpretation
The ranked anomalies should be treated as an investigation queue, not as confirmed incidents. Higher-rank items combine high alert rate with meaningful volume and tight confidence intervals.
