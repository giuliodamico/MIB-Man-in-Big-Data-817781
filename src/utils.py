"""
utils.py — Shared utilities for the Transit Anomaly Detection project.

Imported by BOTH Classical_approach.ipynb AND Multi-Agent_approach.ipynb to
guarantee identical cleaning, feature engineering, detection, and
post-processing logic across both pipelines. This is the single source of
truth: any methodological change happens here once, both notebooks pick it up.

Sections:
    1. Constants
    2. Reference dictionaries 
    3. Cleaning             — load_clean_data()
    4. Feature engineering  — build_route_master(), build_feature_matrix()
    5. Anomaly detection    — fit_detectors() [DBSCAN min_samples FIXED]
    6. Post-processing      — wilson_ci(), quality_note(), apply_post_processing()
"""

import re
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy import stats

warnings.filterwarnings("ignore")

# 1. CONSTANTS
RANDOM_STATE  = 42
CONTAMINATION = 0.05

# Signal-column cap. Justified empirically in Classical §2.2.5
# Distribution shows a clear knee at ~150–180; values above are typing/parsing
# errors. Cap at 200 retains > 99.5% of legitimate observations.
SIGNAL_CAP    = 200

DISQUALIFYING_FLAGS = {
    "incomplete data — alarms but no traveler records",
    "likely false positive — flagged on non-rate features",
}
# 2. REFERENCE DICTIONARIES
IATA_MAPPING = {
    'ABJ': ('Abidjan', 'Felix Houphouet Boigny'),
    'ADB': ('Izmir', 'Adnan Menderes'),
    'ADD': ('Addis Ababa', 'Bole International'),
    'ADL': ('Adelaide', 'Adelaide International'),
    'AER': ('Sochi', 'Sochi International'),
    'AKL': ('Auckland', 'Auckland International'),
    'ALA': ('Almaty', 'Almaty International'),
    'ALG': ('Algiers', 'Houari Boumediene'),
    'AMD': ('Ahmedabad', 'Ahmedabad'),
    'AMM': ('Amman', 'Queen Alia International'),
    'ARN': ('Stockholm', 'Stockholm Arlanda'),
    'ASB': ('Ashgabat', 'Ashgabat'),
    'ASM': ('Asmara', 'Asmara International'),
    'ATL': ('Atlanta', 'Hartsfield-Jackson Atlanta International'),
    'AUH': ('Abu Dhabi', 'Abu Dhabi International'),
    'AYT': ('Antalya', 'Antalya'),
    'BAH': ('Manama', 'Bahrain International'),
    'BEG': ('Belgrade', 'Nikola Tesla'),
    'BEY': ('Beirut', 'Beirut-Rafic Hariri International'),
    'BFS': ('Belfast', 'Belfast International'),
    'BHX': ('Birmingham', 'Birmingham International'),
    'BKK': ('Bangkok', 'Suvarnabhumi International'),
    'BLR': ('Bangalore', 'Bangalore International'),
    'BNA': ('Nashville', 'Nashville International'),
    'BNE': ('Brisbane', 'Brisbane International'),
    'BOG': ('Bogota', 'Eldorado International'),
    'BOS': ('Boston', 'Logan International'),
    'BRS': ('Bristol', 'Bristol Airport'),
    'BSB': ('Brasilia', 'Presidente Juscelino Kubitschek'),
    'BUF': ('Buffalo', 'Buffalo Niagara International'),
    'CAI': ('Cairo', 'Cairo International'),
    'CAN': ('Guangzhou', 'Guangzhou Baiyun International'),
    'CEB': ('Cebu', 'Mactan-Cebu International'),
    'CGK': ('Jakarta', 'Soekarno-Hatta International'),
    'CLE': ('Cleveland', 'Hopkins International'),
    'CMN': ('Casablanca', 'Mohammed V International'),
    'CUN': ('Cancun', 'Cancun International'),
    'CVG': ('Cincinnati', 'Cincinnati/Northern Kentucky'),
    'DAR': ('Dar Es Salaam', 'Julius Nyerere International'),
    'DEL': ('New Delhi', 'Indira Gandhi International'),
    'DFW': ('Dallas-Fort Worth', 'Dallas/Fort Worth International'),
    'DMM': ('Dammam', 'King Fahd International'),
    'DOH': ('Doha', 'Hamad International'),
    'DPS': ('Denpasar', 'Ngurah Rai International'),
    'DSS': ('Dakar', 'Blaise Diagne International'),
    'DTW': ('Detroit', 'Detroit Metropolitan Wayne County'),
    'DUR': ('Durban', 'King Shaka International'),
    'DWC': ('Dubai', 'Al Maktoum International'),
    'DXB': ('Dubai', 'Dubai International'),
    'EDI': ('Edinburgh', 'Edinburgh Airport'),
    'ELQ': ('Buraidah', 'Gassim Regional'),
    'EMA': ('Derby', 'East Midlands'),
    'ESB': ('Ankara', 'Esenboga'),
    'EVN': ('Yerevan', 'Zvartnots International'),
    'EWR': ('Newark', 'Newark Liberty International'),
    'EZE': ('Buenos Aires', 'Ezeiza Ministro Pistarini'),
    'FEZ': ('Fes', 'Fes-Saiss'),
    'FIH': ('Kinshasa', "N'Djili International"),
    'FLL': ('Fort Lauderdale', 'Fort Lauderdale/Hollywood International'),
    'FRA': ('Frankfurt', 'Frankfurt International'),
    'FRU': ('Bishkek', 'Manas'),
    'FUK': ('Fukuoka', 'Fukuoka'),
    'GIG': ('Rio de Janeiro', 'Galeao-Antonio Carlos Jobim International'),
    'GLA': ('Glasgow', 'Glasgow International'),
    'GRU': ('Sao Paulo', 'Sao Paulo/Guarulhos International'),
    'GYD': ('Baku', 'Heydar Aliyev International'),
    'GZT': ('Gaziantep', 'Gaziantep'),
    'HAN': ('Hanoi', 'Noibai International'),
    'HGH': ('Hangzhou', 'Hangzhou Xiaoshan International'),
    'HKG': ('Hong Kong', 'Hong Kong International'),
    'HKT': ('Phuket', 'Phuket International'),
    'HND': ('Tokyo', 'Tokyo Haneda International'),
    'HRG': ('Hurghada', 'Hurghada International'),
    'IAD': ('Washington', 'Washington Dulles International'),
    'IAH': ('Houston', 'George Bush Intercontinental'),
    'ICN': ('Seoul', 'Incheon International'),
    'IFN': ('Isfahan', 'Isfahan International'),
    'IKA': ('Tehran', 'Imam Khomeini International'),
    'ISB': ('Islamabad', 'Islamabad International'),
    'IST': ('Istanbul', 'Istanbul Airport'),
    'JAX': ('Jacksonville', 'Jacksonville International'),
    'JED': ('Jeddah', 'King Abdulaziz International'),
    'JFK': ('New York', 'John F. Kennedy International'),
    'KBL': ('Kabul', 'Khwaja Rawash'),
    'KCH': ('Kuching', 'Kuching International'),
    'KIV': ('Chisinau', 'Chisinau International'),
    'KUL': ('Kuala Lumpur', 'Kuala Lumpur International'),
    'KUT': ('Kutaisi', 'Kutaisi International'),
    'KWI': ('Kuwait City', 'Kuwait International'),
    'KZN': ('Kazan', 'Kazan International'),
    'LAD': ('Luanda', '4 de Fevereiro'),
    'LAS': ('Las Vegas', 'McCarran International'),
    'LAX': ('Los Angeles', 'Los Angeles International'),
    'LCY': ('London', 'London City'),
    'LGW': ('London', 'London Gatwick'),
    'LHR': ('London', 'London Heathrow'),
    'LOS': ('Lagos', 'Murtala Muhammed'),
    'LPL': ('Liverpool', 'Liverpool John Lennon'),
    'LRM': ('La Romana', 'Casa de Campo International'),
    'LTN': ('London', 'London Luton'),
    'MAN': ('Manchester', 'Manchester Airport'),
    'MBA': ('Mombasa', 'Moi International'),
    'MCT': ('Muscat', 'Muscat International'),
    'MED': ('Madinah', 'Mohammad Bin Abdulaziz'),
    'MEL': ('Melbourne', 'Melbourne Tullamarine'),
    'MEX': ('Mexico City', 'Benito Juarez International'),
    'MHD': ('Mashad', 'Mashad'),
    'MIA': ('Miami', 'Miami International'),
    'MJI': ('Tripoli', 'Mitiga'),
    'MLE': ('Male', 'Velana International'),
    'MNL': ('Manila', 'Ninoy Aquino International'),
    'MPM': ('Maputo', 'Maputo International'),
    'MRU': ('Plaisance', 'Sir Seewoosagur Ramgoolam International'),
    'MYR': ('Myrtle Beach', 'Myrtle Beach International'),
    'NAV': ('Nevsehir', 'Nevsehir Kapadokya'),
    'NBO': ('Nairobi', 'Jomo Kenyatta International'),
    'NCL': ('Newcastle', 'Newcastle Airport'),
    'NOS': ('Nosy Be', 'Fascene'),
    'NRT': ('Tokyo', 'Narita International'),
    'NSI': ('Yaounde', 'Nsimalen International'),
    'ORD': ('Chicago', "Chicago O'Hare International"),
    'OUA': ('Ouagadougou', 'Ouagadougou'),
    'OXB': ('Bissau', 'Osvaldo Vieira International'),
    'PEK': ('Beijing', 'Beijing Capital International'),
    'PER': ('Perth', 'Perth International'),
    'PEW': ('Peshawar', 'Peshawar'),
    'PHL': ('Philadelphia', 'Philadelphia International'),
    'PKX': ('Beijing', 'Beijing Daxing International'),
    'POA': ('Porto Alegre', 'Salgado Filho'),
    'PRN': ('Pristina', 'Prishtina International'),
    'PVG': ('Shanghai', 'Shanghai Pudong International'),
    'PVR': ('Puerto Vallarta', 'Licenciado Gustavo Diaz Ordaz International'),
    'RAK': ('Marrakech', 'Menara'),
    'REC': ('Recife', 'Guararapes International'),
    'RMF': ('Marsa Alam', 'Marsa Alam International'),
    'RMO': ('Chisinau', 'Chisinau International'),
    'RUH': ('Riyadh', 'King Khaled International'),
    'SAW': ('Istanbul', 'Sabiha Gokcen'),
    'SCL': ('Santiago', 'Arturo Merino Benitez'),
    'SDU': ('Rio de Janeiro', 'Santos Dumont'),
    'SGN': ('Ho Chi Minh City', 'Tan Son Nhat International'),
    'SHJ': ('Sharjah', 'Sharjah International'),
    'SIN': ('Singapore', 'Changi'),
    'SJJ': ('Sarajevo', 'Sarajevo'),
    'SKG': ('Thessaloniki', 'Macedonia International'),
    'SKP': ('Skopje', 'Skopje International'),
    'SKT': ('Sialkot', 'Sialkot International'),
    'SMF': ('Sacramento', 'Sacramento International'),
    'SOF': ('Sofia', 'Sofia'),
    'SPX': ('Giza', 'Sphinx International'),
    'SSA': ('Salvador', 'Luis Eduardo Magalhaes'),
    'SSH': ('Sharm El Sheikh', 'Sharm El-Sheikh International'),
    'STN': ('London', 'London Stansted'),
    'SVO': ('Moscow', 'Sheremetyevo'),
    'SYD': ('Sydney', 'Kingsford Smith'),
    'SYR': ('Syracuse', 'Syracuse Hancock International'),
    'SYZ': ('Shiraz', 'Shiraz'),
    'SZX': ('Shenzhen', 'Shenzhen Baoan International'),
    'TAS': ('Tashkent', 'Tashkent International'),
    'TBS': ('Tbilisi', 'Tbilisi International'),
    'TFU': ('Chengdu', 'Tianfu International'),
    'TGD': ('Podgorica', 'Podgorica Airport'),
    'TIA': ('Tirana', 'Rinas Mother Teresa'),
    'TLV': ('Tel Aviv', 'Ben Gurion International'),
    'TNG': ('Tangier', 'Ibn Battouta'),
    'TPA': ('Tampa', 'Tampa International'),
    'TPE': ('Taipei', 'Taiwan Taoyuan International'),
    'TSA': ('Taipei', 'Taipei Songshan'),
    'TUN': ('Tunis', 'Tunis-Carthage'),
    'VKO': ('Moscow', 'Vnukovo'),
    'YEG': ('Edmonton', 'Edmonton International'),
    'YHZ': ('Halifax', 'Halifax International'),
    'YUL': ('Montreal', 'Montreal-Trudeau International'),
    'YVR': ('Vancouver', 'Vancouver International'),
    'YWG': ('Winnipeg', 'Winnipeg James Armstrong Richardson International'),
    'YYC': ('Calgary', 'Calgary International'),
    'YYZ': ('Toronto', 'Toronto Pearson International'),
    'ZNZ': ('Zanzibar', 'Abeid Amani Karume International'),
    'AGA': ('Agadir',       'Agadir Al Massira'),
    'ANU': ("St. John's",   'V.C. Bird International'),
    'ATQ': ('Amritsar',     'Sri Guru Ram Dass Jee International'),
    'BQH': ('London',       'London Biggin Hill'),
    'CGO': ('Zhengzhou',    'Zhengzhou Xinzheng International'),
    'CKG': ('Chongqing',    'Chongqing Jiangbei International'),
    'FAB': ('Farnborough',  'Farnborough Airport'),
    'ISL': ('Istanbul',     'Ataturk International'),    
    'LBA': ('Leeds',        'Leeds Bradford'),
    'LXR': ('Luxor',        'Luxor International'),
    'LYX': ('Lydd',         'Lydd Airport'),
    'MBJ': ('Montego Bay',  'Sangster International'),
    'NHT': ('Northolt',     'RAF Northolt'),
    'NKG': ('Nanjing',      'Nanjing Lukou International'),
    'OAK': ('Oakland',      'Oakland International'),
    'OXF': ('Oxford',       'London Oxford'),
    'RBA': ('Rabat',        'Rabat-Sale'),
    'SID': ('Sal',          'Amilcar Cabral International'),
    'SLL': ('Salalah',      'Salalah Airport'),
    'WNZ': ('Wenzhou',      'Wenzhou Longwan International'),
    'BZO': ('Bolzano',      'Bolzano'),
    'OLB': ('Olbia',        'Costa Smeralda'),
    'VBS': ('Brescia',      'Brescia Montichiari'),
    'AOI': ('Ancona', 'Falconara'),
    'BDS': ('Brindisi', 'Papola Casale'),
    'BGY': ('Bergamo', 'Orio al Serio'),
    'BLQ': ('Bologna', 'Guglielmo Marconi'),
    'BRI': ('Bari', 'Palese'),
    'CAG': ('Cagliari', 'Elmas'),
    'CIA': ('Roma', 'Ciampino'),
    'CIY': ('Comiso', 'Comiso'),
    'CTA': ('Catania', 'Fontanarossa'),
    'CUF': ('Cuneo', 'Levaldigi'),
    'FCO': ('Roma', 'Fiumicino'),
    'FLR': ('Firenze', 'Firenze Peretola'),
    'GOA': ('Genova', 'Cristoforo Colombo'),
    'LIN': ('Milano', 'Linate'),
    'MXP': ('Milano', 'Malpensa'),
    'NAP': ('Napoli', 'Napoli Capodichino'),
    'PEG': ('Perugia', 'Sant Egidio'),
    'PMF': ('Parma', 'Parma'),
    'PMO': ('Palermo', 'Punta Raisi'),
    'PSA': ('Pisa', 'Galileo Galilei'),
    'PSR': ('Pescara', 'Liberi'),
    'REG': ('Reggio Calabria', 'Tito Menniti'),
    'RMI': ('Rimini', 'Miramare'),
    'SUF': ('Lamezia Terme', 'Lamezia Terme'),
    'TRN': ('Torino', 'Sandro Pertini'),
    'TRS': ('Trieste', 'Ronchi Dei Legionari'),
    'TSF': ('Treviso', "Treviso-Sant'Angelo"),
    'VCE': ('Venezia', 'Venezia Marco Polo'),
    'VRN': ('Verona', 'Verona Villafranca'),
}
COL_MAP = {
    "OCCORRENZE":"OCCURRENCES","AREOPORTO_ARRIVO":"ARRIVAL_AIRPORT_IATA",
    "AREOPORTO_PARTENZA":"DEPARTURE_AIRPORT_IATA","ANNO_PARTENZA":"DEPARTURE_YEAR",
    "MESE_PARTENZA":"DEPARTURE_MONTH","DATA_PARTENZA":"DEPARTURE_DATE",
    "DESCR_AEREOPORTO_ARR":"ARRIVAL_AIRPORT_DESCRIPTION",
    "DESCR_AEREOPORTO_PART":"DEPARTURE_AIRPORT_DESCRIPTION",
    "CITTA_ARR":"ARRIVAL_CITY","CITTA_PARTENZA":"DEPARTURE_CITY",
    "CODICE_PAESE_ARR":"ARRIVAL_COUNTRY_CODE","CODICE_PAESE_PART":"DEPARTURE_COUNTRY_CODE",
    "PAESE_ARR":"ARRIVAL_COUNTRY","PAESE_PART":"DEPARTURE_COUNTRY",
    "ZONA":"ZONE","TOT":"TOTAL","MOTIVO_ALLARME":"ALARM_REASON",
    "note_operatore":"OPERATOR_NOTES","flag_rischio":"RISK_FLAG",
    "Paese Partenza":"DEPARTURE_COUNTRY_FULL",
    "CODICE PAESE ARR":"ARRIVAL_COUNTRY_CODE_FULL",
    "3zona":"ZONE_3","paese%arr":"ARRIVAL_COUNTRY_PERCENTAGE",
    "tot voli":"TOTAL_FLIGHTS","NAZIONALITA":"NATIONALITY",
    "GIORNO_PARTENZA":"DEPARTURE_DAY","ENTRATI":"ENTRIES",
    "INVESTIGATI":"INVESTIGATED","ALLARMATI":"ALARMS",
    "TIPO_DOCUMENTO":"DOCUMENT_TYPE","GENERE":"GENDER",
    "FASCIA_ETA":"AGE_GROUP","FLAG_TRANSITO":"TRANSIT_FLAG",
    "COMPAGNIA_AEREA":"AIRLINE","NUMERO_VOLO":"FLIGHT_NUMBER",
    "ESITO_CONTROLLO":"CONTROL_OUTCOME","codice_rischio":"RISK_CODE",
    "Tipo_Documento":"DOCUMENT_TYPE2","Tipo Documento":"DOCUMENT_TYPE2",
    "FASCIA_ETA2":"AGE_GROUP2","FASCIA ETA":"AGE_GROUP2",
    "3nazionalita":"NATIONALITY_3","compagnia%aerea":"AIRLINE%",
    "num volo":"FLIGHT NUMBER",
}
OCCURRENCES_MAP = {
    'viaggiatori entrati nel sistema':'entries',
    'voli disponibili in ingresso al sistema':'available_inbound_flights',
    'voli solo visualizzati, ma non investigati':'viewed_not_investigated_flights',
    'voli investigati (sdi/nsis - interpol - tsc)':'investigated_flights',
    'viaggiatori investigati':'investigated_entries',
    'voli con allarmi':'flights_with_alerts',
    'viaggiatori con allarmi':'entries_with_alerts',
    'allarmi generati da sdi/nsis':'generated_alerts',
    'allarmi generati da bcs':'generated_alerts',
    'allarmi generati da interpol':'generated_alerts',
    'allarmi generati':'generated_alerts',
    'allarmi chiusi':'closed_alerts',
    'allarmi chiusi con azione (cc.xx)':'closed_alerts',
    'allarmi rilevanti':'relevant_alerts',
    'allarmi non chiusi':'open_alerts',
    'nulla a procedere sdi':'negative_outcome',
    'nulla a procedere nsis':'negative_outcome',
    'nulla a procedere int':'negative_outcome',
    'errata segnalazione sdi':'negative_outcome',
    'errata segnalazione nsis':'negative_outcome',
    'errata segnalazione bcs':'negative_outcome',
    'respinto/a':'negative_outcome',
    'mancato aggiornamento sdi':'missing_update',
    'mancato aggiornamento schengen nsis':'missing_update',
    'notifica atti/provv':'other',
    'inammissibilita schengen - art.24':'other',
    'altro':'other',
    '???':np.nan,
    'n/c':np.nan,
    'ali armatt':np.nan,
}
IT_TO_ALPHA3 = {
    "Albania": "ALB", "Algeria": "DZA", "Antigua e Barbuda": "ATG",
    "Arabia Saudita": "SAU", "Argentina": "ARG", "Armenia": "ARM",
    "Azerbaigian": "AZE", "Bahrain": "BHR", "Brasile": "BRA", "Canada": "CAN",
    "Capo Verde": "CPV", "Cina": "CHN", "Corea del Sud": "KOR", "Cossovo": "RKS",
    "Egitto": "EGY", "Emirati Arabi Uniti": "ARE", "Etiopia": "ETH",
    "Georgia": "GEO", "Giamaica": "JAM", "Giappone": "JPN", "Giordania": "JOR",
    "Hong Kong": "HKG", "India": "IND", "Iran": "IRN", "Israele": "ISR",
    "Kazakistan": "KAZ", "Kenya": "KEN", "Kuwait": "KWT", "Libano": "LBN",
    "Libia": "LBY", "Macedonia": "MKD", "Maldive": "MDV", "Marocco": "MAR",
    "Mauritius": "MUS", "Messico": "MEX", "Moldavia": "MDA", "Montenegro": "MNE",
    "Oman": "OMN", "Qatar": "QAT", "Regno Unito": "GBR",
    "Repubblica Dominicana": "DOM", "Senegal": "SEN", "Serbia": "SRB",
    "Singapore": "SGP", "Stati Uniti": "USA", "Taiwan": "TWN",
    "Tanzania": "TZA", "Tunisia": "TUN", "Turchia": "TUR", "Uzbekistan": "UZB",
    "Afghanistan": "AFG", "Angola": "AGO", "Australia": "AUS",
    "Bosnia Herzegovina": "BIH", "Bulgaria": "BGR", "Burkina Faso": "BFA",
    "Camerun": "CMR", "Cile": "CHL", "Colombia": "COL",
    "Congo (Kinshasa)": "COD", "Costa d'Avorio": "CIV", "Eritrea": "ERI",
    "Filippine": "PHL", "Germania": "DEU", "Grecia": "GRC",
    "Guinea-Bissau": "GNB", "Indonesia": "IDN", "Kirghizistan": "KGZ",
    "Madagascar": "MDG", "Malesia": "MYS", "Mozambico": "MOZ", "Nigeria": "NGA",
    "Nuova Zelanda": "NZL", "Pakistan": "PAK", "Russia": "RUS",
    "Sudafrica": "ZAF", "Svezia": "SWE", "Thailandia": "THA",
    "Turkmenistan": "TKM", "Vietnam": "VNM",
}
PLACEHOLDER_NULL = {
    "n.d.", "?", "", "//", "-", "unknown", "nd", "n/a", "null", "unk",
    "zz", "none", "xx", "???", " ", "ND", "UNK", "XX", "ZZ", "EU",
}

CITY_TO_IATA = {city.upper(): iata for iata, (city, _) in IATA_MAPPING.items()}
_COUNTRY_TO_IATA_CACHE: dict = {}

# 3. CLEANING
def load_clean_data(io_dir: str = "io"):
    df_alarms    = pd.read_csv(f"{io_dir}/ALLARMI.csv")
    df_travelers = pd.read_csv(f"{io_dir}/TIPOLOGIA_VIAGGIATORE.csv")

    df_alarms    = df_alarms.rename(columns=COL_MAP)
    df_travelers = df_travelers.rename(columns=COL_MAP)

    df_travelers = _clean_travelers(df_travelers)
    df_alarms    = _clean_alarms(df_alarms)

    for df in (df_alarms, df_travelers):
        df["DEPARTURE_DATE"] = _parse_dates(df["DEPARTURE_DATE"])
    df_alarms    = df_alarms.dropna(subset=["DEPARTURE_DATE"]).reset_index(drop=True)
    df_travelers = df_travelers.dropna(subset=["DEPARTURE_DATE"]).reset_index(drop=True)

    return df_alarms, df_travelers

def _parse_dates(s: pd.Series) -> pd.Series:
    raw = (s.astype(str).str.strip()
           .str.replace(r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})",
                        r"\1 \2", regex=True))
    out = pd.to_datetime(raw, format="mixed", dayfirst=True, errors="coerce")
    return out.fillna(pd.to_datetime(raw, errors="coerce"))

def _clean_placeholders(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin(PLACEHOLDER_NULL), np.nan)

def _clean_travelers(df: pd.DataFrame) -> pd.DataFrame:
    """Travelers cleaning — mirrors Classical §2.3."""
    df = df.copy()

    # IATA enrichment for departure/arrival
    iata_to_city    = {k: v[0] for k, v in IATA_MAPPING.items()}
    iata_to_airport = {k: v[1] for k, v in IATA_MAPPING.items()}

    for prefix in ("DEPARTURE", "ARRIVAL"):
        col = f"{prefix}_AIRPORT_IATA"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
            df[f"{prefix}_CITY"]                = df[col].map(iata_to_city)
            df[f"{prefix}_AIRPORT_DESCRIPTION"] = df[col].map(iata_to_airport)

    # Arrival is always Italy (per the brief)
    df["ARRIVAL_COUNTRY"]      = "ITALY"
    df["ARRIVAL_COUNTRY_CODE"] = "ITA"

    # Departure country → ISO alpha-3
    if "DEPARTURE_COUNTRY" in df.columns:
        df["DEPARTURE_COUNTRY"]      = df["DEPARTURE_COUNTRY"].astype(str).str.strip()
        df["DEPARTURE_COUNTRY_CODE"] = df["DEPARTURE_COUNTRY"].map(IT_TO_ALPHA3)

    # ZONE
    if "ZONE" in df.columns:
        df["ZONE"] = pd.to_numeric(df["ZONE"].replace(["-1", "99", "??"], np.nan),
                                   errors="coerce")
        df.loc[(df["ZONE"] < 0) | (df["ZONE"] > 10), "ZONE"] = np.nan
        df["ZONE"] = df["ZONE"].astype("Int64")
        df.drop(columns=["ZONE_3"], inplace=True, errors="ignore")

    # GENDER
    if "GENDER" in df.columns:
        gender_map = {
            "MASCHIO": "M", "MALE": "M", "1": "M",
            "FEMMINA": "F", "FEMALE": "F", "2": "F",
            " ": "N.D.", "-": "N.D.", "//": "N.D.", "X": "N.D.",
            "N/B": "N.D.", "ND": "N.D.", "UNKNOWN": "N.D.", "UNKN": "N.D.", "?": "N.D.",
        }
        df["GENDER"] = df["GENDER"].astype(str).str.upper().replace(gender_map).fillna("N.D.")

    # OPERATOR_NOTES
    if "OPERATOR_NOTES" in df.columns:
        df["OPERATOR_NOTES"] = (df["OPERATOR_NOTES"].astype(str).str.upper().str.strip()
                                .replace("NAN", np.nan).fillna("NO_NOTES"))

    # TRANSIT_FLAG
    if "TRANSIT_FLAG" in df.columns:
        transit_map = {"SINGOLA TRATTA": "1", "MULTITRATTA": "2",
                       "TRIPLA TRATTA": "3", "TRANSITO": "STOPOVER"}
        df["TRANSIT_FLAG"] = (df["TRANSIT_FLAG"].astype(str).str.upper()
                              .replace(transit_map).replace("N/C", np.nan))

    # NATIONALITY
    if "NATIONALITY" in df.columns:
        df["NATIONALITY"] = df["NATIONALITY"].fillna("Unknown")
        df["NATIONALITY"] = df["NATIONALITY"].replace(list(PLACEHOLDER_NULL), "Unknown")
        if "NATIONALITY_3" in df.columns:
            df.drop(columns=["NATIONALITY_3"], inplace=True, errors="ignore")

    # Drop redundant/duplicate columns
    for redundant in ("DEPARTURE_YEAR", "DEPARTURE_MONTH",
                      "AGE_GROUP", "DOCUMENT_TYPE", "AIRLINE", "FLIGHT_NUMBER"):
        df.drop(columns=[redundant], inplace=True, errors="ignore")

    # Signal columns: cleanup + cap at SIGNAL_CAP
    # Justification of the cap is documented in Classical §2.2.5.
    for c in ("ENTRIES", "INVESTIGATED", "ALARMS"):
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace(r"(?i)pax", "", regex=True)
                     .str.replace(r"\s+", "", regex=True))
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if all(c in df.columns for c in ("ENTRIES", "INVESTIGATED", "ALARMS")):
        mask = (df["ENTRIES"].between(0, SIGNAL_CAP) &
                df["INVESTIGATED"].between(0, SIGNAL_CAP) &
                df["ALARMS"].between(0, SIGNAL_CAP))
        df = df[mask].copy()
        for c in ("ENTRIES", "INVESTIGATED", "ALARMS"):
            df[c] = df[c].astype(int)

    return df

def _clean_alarms(df: pd.DataFrame) -> pd.DataFrame:
    """Alarms cleaning — mirrors Classical §2.4."""
    df = df.copy()
    df.drop(columns=["DEPARTURE_MONTH", "DEPARTURE_YEAR",
                     "ARRIVAL_COUNTRY_CODE",
                     "ARRIVAL_COUNTRY_CODE_PERCENTAGE"],
            inplace=True, errors="ignore")

    if "OCCURRENCES" in df.columns:
        df["OCCURRENCES"] = (df["OCCURRENCES"].astype(str).str.lower().str.strip()
                             .map(OCCURRENCES_MAP))

    if "ZONE" in df.columns:
        df["ZONE"] = pd.to_numeric(df["ZONE"].replace(["-1", "99", "??"], np.nan),
                                   errors="coerce")
        df.loc[(df["ZONE"] < 0) | (df["ZONE"] > 10), "ZONE"] = np.nan
        if "ZONE_3" in df.columns:
            df["ZONE"] = df["ZONE"].astype("Int64").combine_first(df["ZONE_3"])
            df.drop(columns=["ZONE_3"], inplace=True, errors="ignore")

    text_fill = {"ALARM_REASON": "NO REASON PROVIDED",
                 "OPERATOR_NOTES": "NO MANUAL NOTES"}
    for col, placeholder in text_fill.items():
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.upper()
                       .replace({"NAN": np.nan}).fillna(placeholder))

    if "RISK_FLAG" in df.columns:
        df["RISK_FLAG"] = (df["RISK_FLAG"]
                           .replace({"ALTO": "HIGH RISK", "MEDIO": "MEDIUM RISK"})
                           .fillna("LOW RISK"))

    iata_to_city    = {k: v[0] for k, v in IATA_MAPPING.items()}
    iata_to_airport = {k: v[1] for k, v in IATA_MAPPING.items()}

    iata_dep = df["DEPARTURE_AIRPORT_IATA"].astype(str).str.upper().str.strip()
    df["DEPARTURE_AIRPORT_IATA"]        = iata_dep
    df["DEPARTURE_CITY"]                = iata_dep.map(iata_to_city)
    df["DEPARTURE_AIRPORT_DESCRIPTION"] = iata_dep.map(iata_to_airport)

    country = (_clean_placeholders(df["DEPARTURE_COUNTRY_FULL"])
               .combine_first(_clean_placeholders(df["DEPARTURE_COUNTRY"])))
    df["DEPARTURE_COUNTRY"]      = country
    df["DEPARTURE_COUNTRY_CODE"] = country.map(IT_TO_ALPHA3)
    df.drop(columns=["DEPARTURE_COUNTRY_FULL"], inplace=True, errors="ignore")

    iata_arr = df["ARRIVAL_AIRPORT_IATA"].astype(str).str.upper().str.strip()
    df["ARRIVAL_AIRPORT_IATA"]        = iata_arr
    df["ARRIVAL_CITY"]                = iata_arr.map(iata_to_city)
    df["ARRIVAL_AIRPORT_DESCRIPTION"] = iata_arr.map(iata_to_airport)
    df["ARRIVAL_COUNTRY"]             = "Italia"
    df["ARRIVAL_COUNTRY_CODE"]        = "ITA"

    if "TOTAL_FLIGHTS" in df.columns:
        s = (df["TOTAL_FLIGHTS"].astype(str).str.strip()
             .str.replace(r"\s*voli\s*$", "", regex=True).str.lstrip("~"))
        s = pd.to_numeric(s, errors="coerce")
        df["TOTAL_FLIGHTS"] = s.where((s >= 0) & (s <= 1000)).fillna(0).round().astype(int)
    df.drop(columns=["TOTAL"], inplace=True, errors="ignore")

    return df

# 4. FEATURE ENGINEERING
def build_route_master(df_alarms: pd.DataFrame,
                       df_travelers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate at route level. Identical for both pipelines."""
    pivot = (df_alarms.pivot_table(
                index=["DEPARTURE_AIRPORT_IATA", "ARRIVAL_AIRPORT_IATA"],
                columns="OCCURRENCES", values="TOTAL_FLIGHTS",
                aggfunc="sum", fill_value=0))
    pivot.columns = [f"tot_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    desc = (df_alarms
            .groupby(["DEPARTURE_AIRPORT_IATA", "ARRIVAL_AIRPORT_IATA"])
            .agg(DEPARTURE_CITY=("DEPARTURE_CITY", "first"),
                 DEPARTURE_COUNTRY=("DEPARTURE_COUNTRY", "first"),
                 DEPARTURE_COUNTRY_CODE=("DEPARTURE_COUNTRY_CODE", "first"),
                 ARRIVAL_CITY=("ARRIVAL_CITY", "first"),
                 ZONE=("ZONE", lambda s: s.mode().iat[0] if s.notna().any() else np.nan),
                 n_high_risk=("RISK_FLAG", lambda s: (s == "HIGH RISK").sum()),
                 n_medium_risk=("RISK_FLAG", lambda s: (s == "MEDIUM RISK").sum()),
                 alarm_reason_top=("ALARM_REASON",
                                   lambda s: s.mode().iat[0] if s.notna().any() else np.nan))
            .reset_index())
    df_a_route = pivot.merge(desc, on=["DEPARTURE_AIRPORT_IATA", "ARRIVAL_AIRPORT_IATA"])

    top_nat = [v for v in df_travelers["NATIONALITY"].value_counts().head(3).index
               if pd.notna(v) and v != "Unknown"][:3]
    top_doc = ([v for v in df_travelers["DOCUMENT_TYPE2"].value_counts().head(3).index
                if pd.notna(v)][:3]
               if "DOCUMENT_TYPE2" in df_travelers.columns else [])
    top_co  = ([v for v in df_travelers["CONTROL_OUTCOME"].value_counts().head(4).index
                if pd.notna(v)][:4]
               if "CONTROL_OUTCOME" in df_travelers.columns else [])

    def _safe(num, den): return num / den if den > 0 else np.nan

    def agg_one(g):
        inv, alm = g["INVESTIGATED"].sum(), g["ALARMS"].sum()
        intensity = _safe(alm, inv)
        out = {"tot_entries": g["ENTRIES"].sum(),
               "tot_investigated": inv,
               "tot_alarms": alm,
               "n_records": len(g),
               "alarm_intensity": intensity,
               "alert_rate": (min(intensity, 1.0) if pd.notna(intensity) else np.nan),
               "investigation_rate": _safe(inv, g["ENTRIES"].sum())}
        for nat in top_nat:
            sub = g[g["NATIONALITY"] == nat]
            out[f"alert_rate_{nat}"] = _safe(sub["ALARMS"].sum(), sub["INVESTIGATED"].sum())
            out[f"pct_nat_{nat}"]    = (g["NATIONALITY"] == nat).mean()
        for doc in top_doc:
            sub = g[g["DOCUMENT_TYPE2"] == doc]
            out[f"alert_rate_doc_{doc}"] = _safe(sub["ALARMS"].sum(),
                                                 sub["INVESTIGATED"].sum())
        if top_co:
            co_norm = g["CONTROL_OUTCOME"].value_counts(normalize=True, dropna=True)
            for outcome in top_co:
                out[f"pct_{outcome}"] = co_norm.get(outcome, 0.0)
        return pd.Series(out)

    df_t_route = (df_travelers
                  .groupby(["DEPARTURE_AIRPORT_IATA", "ARRIVAL_AIRPORT_IATA"])
                  .apply(agg_one).reset_index())

    df_route = df_a_route.merge(df_t_route,
                                on=["DEPARTURE_AIRPORT_IATA", "ARRIVAL_AIRPORT_IATA"],
                                how="outer", indicator=True)
    df_route["data_coverage"] = df_route["_merge"].astype(str)
    df_route = df_route.drop(columns="_merge")

    num_cols = df_route.select_dtypes(include="number").columns
    df_route[num_cols] = df_route[num_cols].fillna(0)
    df_route["route"] = (df_route["DEPARTURE_AIRPORT_IATA"]
                         + "→" + df_route["ARRIVAL_AIRPORT_IATA"])
    return df_route

def build_feature_matrix(df_route: pd.DataFrame):
    """log1p + StandardScaler. Identical for both pipelines."""
    count_features = [c for c in df_route.columns if c.startswith("tot_")]
    rate_features  = [c for c in df_route.columns
                      if c.startswith("alert_rate") or c.startswith("pct_")]
    risk_features  = [c for c in ("n_high_risk", "n_medium_risk")
                      if c in df_route.columns]
    extras         = [c for c in ("alarm_intensity", "investigation_rate")
                      if c in df_route.columns]
    feature_cols   = count_features + rate_features + risk_features + extras
    X       = df_route[feature_cols].fillna(0).astype(float)
    X_log   = np.log1p(X.values)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_log),
                            columns=feature_cols, index=X.index)
    return feature_cols, X_scaled

# 5. ANOMALY DETECTION  —  DBSCAN min_samples FIX
def fit_detectors(X_scaled: pd.DataFrame,
                  contamination: float = CONTAMINATION,
                  random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    4-detector ensemble (IF, LOF, DBSCAN, Z-score) → consensus voting.

    HIGH-PRIORITY FIX (DBSCAN):
      The classical-pipeline rule `min_samples = 2 * d` from Sander et al.
      assumes moderate dimensionality. With d ≈ 30+ features, 2*d ≈ 60 on
      ~1.1k routes labels almost everything as noise (curse of dimensionality
      on density estimates). We now use `max(5, ceil(ln(n)) + 1)`, which is
      the practical rule for high-d data, and the DBSCAN vote stops being
      a constant 1 by construction.

      The function also exposes `out.attrs['dbscan_noise_pct']` so the
      notebooks can sanity-check the new behaviour (target: 5–25 % noise).
    """
    n, d = X_scaled.shape
    out  = pd.DataFrame(index=X_scaled.index)

    # 1) Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=random_state).fit(X_scaled)
    out["if_anomaly"] = (iso.predict(X_scaled) == -1).astype(int)
    out["if_score"]   = -iso.decision_function(X_scaled)

    # 2) LOF — k clipped to [2, 20] and never larger than n-1
    lof_k = max(2, min(20, n - 1))
    lof = LocalOutlierFactor(n_neighbors=lof_k, contamination=contamination)
    out["lof_anomaly"] = (lof.fit_predict(X_scaled) == -1).astype(int)
    out["lof_score"]   = -lof.negative_outlier_factor_

    # 3) DBSCAN  — FIXED min_samples for high-d data
    min_samples = max(5, int(np.ceil(np.log(n))) + 1)
    if n >= min_samples + 1:
        nn = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        eps = float(np.quantile(
            np.sort(nn.kneighbors(X_scaled)[0][:, -1]), 0.95))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        out["dbscan_anomaly"] = (db.labels_ == -1).astype(int)
        out.attrs["dbscan_min_samples"] = int(min_samples)
        out.attrs["dbscan_eps"]         = float(eps)
        out.attrs["dbscan_noise_pct"]   = float((db.labels_ == -1).mean())
    else:
        out["dbscan_anomaly"] = 0
        out.attrs["dbscan_noise_pct"] = float("nan")

    # 4) Z-score on the standardised matrix (max |z| > 3 across features)
    med  = np.median(X_scaled.values, axis=0)
    mad  = stats.median_abs_deviation(X_scaled.values, axis=0, scale="normal")
    mad  = np.where(mad < 1e-9, 1e-9, mad)
    Z    = np.abs((X_scaled.values - med) / mad)
    out["z_max"]     = Z.max(axis=1)
    out["z_anomaly"] = (out["z_max"] > 3).astype(int)

    det_cols = ["if_anomaly", "lof_anomaly", "dbscan_anomaly", "z_anomaly"]
    out["anomaly_votes"]     = out[det_cols].sum(axis=1)
    out["anomaly_consensus"] = (out["anomaly_votes"] >= 2).astype(int)
    return out

# 6. POST-PROCESSING
def wilson_ci(rate: float, n: int, z: float = 1.96):
    if n == 0 or pd.isna(rate):
        return np.nan, np.nan
    k = round(rate * n)
    centre = (k + z * z / 2) / (n + z * z)
    margin = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0, centre - margin), min(1, centre + margin)

def quality_note(rate: float, vol: float) -> str:
    if vol == 0 and rate > 0:        return "incomplete data — alarms but no traveler records"
    if rate == 0 and vol <= 2:       return "likely false positive — flagged on non-rate features"
    if rate >= 0.30 and vol <= 3:    return "warning — high rate but tiny volume (≤3 investigated)"
    return "ok"

def apply_post_processing(df_scored: pd.DataFrame, drop_disqualified: bool = True):
    """
    Identical post-processing for BOTH pipelines.

    Returns
    -------
    df_post  : full consensus subset, with risk_level + Wilson CI + priority.
               Always returned for audit / appendix.
    df_clean : df_post minus rows whose quality_note ∈ DISQUALIFYING_FLAGS.
               This is what the final ranked report should use.

    HIGH-PRIORITY FIX: the classical notebook used to filter df_clean inline
    inside the report cell, while the multi-agent built `top_10` directly
    from df_post (no quality filter). Both pipelines now go through this
    single function and produce the same df_clean by construction.
    """
    df_post = df_scored[df_scored["anomaly_consensus"] == 1].copy()
    if df_post.empty:
        empty = df_post.copy()
        return empty, empty

    pop_median   = df_scored["alert_rate"].median()
    p66          = df_post["alert_rate"].quantile(0.66)
    p33          = df_post["alert_rate"].quantile(0.33)
    vol_p25      = df_post["tot_investigated"].quantile(0.25)
    T_HIGH       = max(3.0 * pop_median, p66)
    T_MED        = max(1.5 * pop_median, p33)
    T_VOL        = max(2, vol_p25)

    def _classify(r):
        if r["anomaly_votes"] == 4:
            return "CRITICAL"
        if r["alert_rate"] >= T_HIGH and r["tot_investigated"] >= T_VOL:
            return "HIGH"
        if r["anomaly_votes"] >= 3 and r["alert_rate"] >= pop_median:
            return "HIGH"
        if r["alert_rate"] >= T_MED or r["anomaly_votes"] >= 3:
            return "MEDIUM"
        return "LOW"

    df_post["risk_level"]   = df_post.apply(_classify, axis=1)
    df_post["quality_note"] = df_post.apply(
        lambda r: quality_note(r["alert_rate"], r["tot_investigated"]), axis=1)

    ci = df_post.apply(lambda r: wilson_ci(r["alert_rate"],
                                           int(r["tot_investigated"])), axis=1)
    df_post[["ci95_low", "ci95_high"]] = pd.DataFrame(ci.tolist(), index=df_post.index)
    df_post["ci95_str"] = df_post.apply(
        lambda r: (f"[{r['ci95_low']:.1%}, {r['ci95_high']:.1%}]"
                   if pd.notna(r["ci95_low"]) else "n/a"), axis=1)
    df_post["ci_width"]     = (df_post["ci95_high"] - df_post["ci95_low"]).fillna(1.0)
    df_post["ci_tightness"] = (1 - df_post["ci_width"]).clip(lower=0)

    df_post["absolute_alarms"]     = df_post["alert_rate"] * df_post["tot_investigated"]
    df_post["absolute_alarms_log"] = np.log1p(df_post["absolute_alarms"])
    pop_min, pop_max = df_scored["alert_rate"].min(), df_scored["alert_rate"].max()
    df_post["rate_n"]   = ((df_post["alert_rate"] - pop_min)
                           / max(pop_max - pop_min, 1e-9))
    df_post["alarms_n"] = MinMaxScaler().fit_transform(df_post[["absolute_alarms_log"]])
    df_post["priority_score"] = ((0.60 * df_post["rate_n"]
                                  + 0.40 * df_post["alarms_n"])
                                 * df_post["ci_tightness"]).round(4)

    df_post = df_post.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df_post["rank"] = df_post.index + 1

    if drop_disqualified:
        df_clean = df_post[~df_post["quality_note"].isin(DISQUALIFYING_FLAGS)].copy()
    else:
        df_clean = df_post.copy()

    return df_post, df_clean