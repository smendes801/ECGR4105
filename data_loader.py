import pandas as pd
import numpy as np
import requests
from sgp4.api import Satrec

# --- Data Acquisition and Preprocessing ---

def mean_motion_to_altitude(mean_motion):
    """
    Convert mean motion to orbital altitude.
    Args:
        mean_motion (float): Mean motion (revolutions per day)
    Returns:
        float: Orbital altitude in kilometers
    """
    mu = 3.986e14  # Earth's gravitational parameter (m³/s²)
    revs_per_sec = mean_motion / 86400
    semi_major_axis = (mu / ((2 * np.pi * revs_per_sec)**2))**(1/3)
    altitude_km = (semi_major_axis / 1000) - 6371  # Earth radius in km
    return altitude_km


def calculate_derived_features(df):
    """
    Calculate additional derived features for the dataset.
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Dataframe with derived features
    """
    df['ALTITUDE'] = df['MEAN_MOTION'].apply(mean_motion_to_altitude)
    df['SSO_FLAG'] = df['INCLINATION'].apply(lambda x: 1 if 96 < x < 102 else 0)
    return df


def load_data():
    """
    Load and preprocess satellite and debris data from online sources.
    Returns:
        pd.DataFrame: Combined and processed dataframe
    """
    data_sources = {
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=analyst&FORMAT=csv": "satellite",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=csv": "debris",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=csv": "debris",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=csv": "debris",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=csv": "debris"
    }
    all_dfs = []
    for url, label in data_sources.items():
        try:
            df = pd.read_csv(url)
            df['label'] = label
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    tle_url = "https://celestrak.org/NORAD/elements/gp-last.php?SPECIAL=IMPENDING&FORMAT=tle"
    tle_text = requests.get(tle_url).text
    tle_lines = tle_text.strip().splitlines()
    impending_data = []
    for i in range(0, len(tle_lines), 3):
        if i + 2 >= len(tle_lines):
            break
        name = tle_lines[i].strip()
        line1 = tle_lines[i + 1].strip()
        line2 = tle_lines[i + 2].strip()
        try:
            sat = Satrec.twoline2rv(line1, line2)
            data_point = {
                'MEAN_MOTION': sat.no_kozai,
                'ECCENTRICITY': sat.ecco,
                'INCLINATION': sat.inclo,
                'RA_OF_ASC_NODE': sat.nodeo,
                'ARG_OF_PERICENTER': sat.argpo,
                'MEAN_ANOMALY': sat.mo,
                'BSTAR': sat.bstar,
                'MEAN_MOTION_DOT': sat.ndot,
                'MEAN_MOTION_DDOT': sat.nddot,
                'label': 'impending_reentry'
            }
            impending_data.append(data_point)
        except Exception as e:
            print(f"Error parsing TLE for {name}: {e}")
            continue
    df_impending = pd.DataFrame(impending_data)
    combined_df = pd.concat(all_dfs + [df_impending], ignore_index=True)
    combined_df = calculate_derived_features(combined_df)
    return combined_df 