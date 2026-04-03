# MIB-Man-in-Big-Data

Team members: Giulio D'Amico - Alexis Mitracos - Marco Astrologo

## Reply Project: Classical vs Multi Agents

This project investigates two alternative approaches for transit anomaly detection in airport and border-control data: a classical machine learning pipeline and a multi-agent architecture. The main objective is to implement the same anomaly detection system twice and then compare the two solutions in order to understand which approach is more suitable under different operational conditions.

The application scenario is motivated by the need to move from reactive anomaly detection to a more proactive analytical system. Border control authorities and airport operators manage large volumes of passenger transit data every day, including information such as timestamps, gates, routes, nationality, document type, control outcomes, and security alerts. In this context, identifying unusual patterns early can help prevent operational issues and support security monitoring.

In the classical implementation, anomaly detection is performed through a structured pipeline including data preparation, feature engineering, historical baseline construction, anomaly detection algorithms, and rule-based post-processing. In the multi-agent implementation, the same logic is distributed across specialized agents, each responsible for a specific task such as querying data, building historical baselines, detecting outliers, applying risk rules, and generating a final report.

The goal of the project is not only to detect anomalies, but also to provide a comparative analysis of the two paradigms in terms of modularity, interpretability, scalability, flexibility, and practical usability. The final output is a transit anomaly report highlighting suspicious patterns and supporting the discussion on the advantages and limitations of each approach.

```markdown
### Data Dictionary (Italian Customs & Aviation Terminology)

**1. Temporal Features (Time & Date).**  
`DATA_PARTENZA`: Departure Date (Full timestamp).  
`ANNO_PARTENZA` / `MESE_PARTENZA` / `GIORNO_PARTENZA`: Year / Month / Day of departure.

**2. Geographical & Airport Features.**  
`AREOPORTO_PARTENZA` / `AREOPORTO_ARRIVO`: Departure / Arrival Airport Code (e.g., FCO, JFK).  
`DESCR_AEREOPORTO_...`: Full Airport Name (e.g., Fiumicino, John F Kennedy).  
`CITTA_PARTENZA` / `CITTA_ARR`: Departure / Arrival City.  
`PAESE_PART` / `PAESE_ARR`: Departure / Arrival Country.  
`CODICE_PAESE_...`: Country Code (e.g., ITA, USA).  
`ZONA` / `3zona`: Geographical routing zone.  

**3. Flight & Airline Info**  
`COMPAGNIA_AEREA` / `compagnia%aerea`: Airline Name.  
`NUMERO_VOLO` / `num volo`: Flight Number.  
`FLAG_TRANSITO`: Indicates if it's a transit flight (Connecting flight).  

**4. Passenger Demographics (Travelers Dataset).**  
`NAZIONALITA` / `3nazionalita`: Passenger Nationality.  
`GENERE`: Gender.  
`FASCIA_ETA` / `FASCIA ETA`: Age Group (e.g., 18-30).  
`TIPO_DOCUMENTO` / `Tipo Documento`: Document used (Passport, ID Card).  

**5. Quantitative Metrics (The Numbers)**  
`TOT` / `tot voli`: Number of alarms for a specific occurenzi.   
`ENTRATI`: Total number of passengers who *entered* the system/country.  
`INVESTIGATI`: Number of passengers *investigated* (Level 1 check).  
`ALLARMATI`: Number of passengers flagged with an *alarm* (Level 2 check/Anomaly).  

**6. Customs & Risk Assessment (The Labels)**  
`OCCORRENZE`: Type of event/occurrence recorded.  
`MOTIVO_ALLARME`: Reason for the alarm/flag. (who has raised the alarm).  
`ESITO_CONTROLLO`: Outcome of the inspection (e.g., Cleared, Rejected, Arrested).  
`codice_rischio` / `flag_rischio`: Risk code/flag (mostly missing data, >98% NaNs).  
`note_operatore`: Customs operator's manual notes (mostly missing data, >98% NaNs).  
```
