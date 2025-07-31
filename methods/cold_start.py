from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from preprocessing.load_cleaned import (
    get_clean_animes,
    get_clean_profiles,
)
from preprocessing.preprocess_pipline import profile_preprocess

CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")

WEIGHT_SCORE = 0.5     
WEIGHT_MEMBERS = 0.3    
WEIGHT_RECENCY = 0.2    

def _build_popularity_table(animes: pd.DataFrame) -> pd.DataFrame:
    """Return *animes* sorted by blended recency‑popularity score (highest first)."""
    df = animes.copy()

    for col in ("score", "members", "aired_year"):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in animes DataFrame.")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    scaler = MinMaxScaler()
    df[["norm_score", "norm_members"]] = scaler.fit_transform(df[["score", "members"]])

    this_year = datetime.now().year
    year_span = (this_year - df["aired_year"]).replace(0, 1)  # avoid /0
    df["recency"] = 1 - (year_span / year_span.max())

    df["pop_score"] = (
        WEIGHT_SCORE   * df["norm_score"] +
        WEIGHT_MEMBERS * df["norm_members"] +
        WEIGHT_RECENCY * df["recency"]
    )

    return df.sort_values("pop_score", ascending=False).reset_index(drop=True)

_POP_TABLE: pd.DataFrame | None = None


def cold_start_top_n(n: int = 20) -> pd.DataFrame:
    """Return the **top‑n** titles as a DataFrame (columns: uid, title, pop_score)."""
    global _POP_TABLE
    if _POP_TABLE is None:
        _POP_TABLE = _build_popularity_table(get_clean_animes())
    return _POP_TABLE.head(n)[["uid", "title", "pop_score"]]

def recommend_for_cold_start_profiles(
    profiles: pd.DataFrame,
    n: int = 20,
) -> Dict[str, List[Tuple[int, float]]]:
    """Return {profile_id: [(uid, pop_score), …]} for all cold‑start users."""
    # Define cold‑start: explicit flag **or** very sparse favourites list
    mask = profiles["is_cold_start"] | (profiles["favorites_anime"].apply(len) <= 3)
    cold_profiles = profiles[mask]

    if cold_profiles.empty:
        return {}

    top_n = cold_start_top_n(n)
    uid_score_pairs = list(zip(top_n["uid"].tolist(), top_n["pop_score"].tolist()))

    return {pid: uid_score_pairs for pid in cold_profiles["profile"]}

def _cli_demo(n: int = 5):
    """Print sample recommendations for the first *n* cold‑start users."""
    profiles = profile_preprocess(get_clean_profiles())
    recs = recommend_for_cold_start_profiles(profiles, n=20)

    print("\n=== Cold‑Start Demo ===")
    for i, (pid, rec_list) in enumerate(recs.items()):
        if i >= n:
            break
        print(f"\nUser {pid} → top {len(rec_list)} starter picks:")
        for uid, score in rec_list[:10]:  # show first 10 for brevity
            print(f"  • UID {uid}  (blend={score:.3f})")


if __name__ == "__main__":
    _cli_demo()