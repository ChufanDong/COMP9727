import pandas as pd
import os

def load_anime_data(path="data/archive/animes.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Anime file not found: {path}")
    df = pd.read_csv(path)
    print(f"[load_anime_data] Loaded {len(df)} anime records.")
    return df

def load_profile_data(path="data/archive/profiles.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile file not found: {path}")
    df = pd.read_csv(path)
    print(f"[load_profile_data] Loaded {len(df)} user profiles.")
    return df

def load_review_data(path="data/archive/reviews.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Review file not found: {path}")
    df = pd.read_csv(path)
    print(f"[load_review_data] Loaded {len(df)} reviews.")
    return df