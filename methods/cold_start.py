from __future__ import annotations

import os, ast, re
import sys
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")

sys.path.append(PROJECT_ROOT)

from preprocessing.load_cleaned import get_clean_animes, get_clean_profiles 
from preprocessing.preprocess_pipline import profile_preprocess 

WEIGHT_SCORE = 0.5   
WEIGHT_MEMBERS = 0.3 
WEIGHT_RECENCY = 0.2 

def to_list(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
        try:                     
            return [g.strip() for g in ast.literal_eval(raw)]
        except (ValueError, SyntaxError):
            pass
    # fallback: split on | or ,
    return [g.strip() for g in re.split(r"[|,]", str(raw)) if g.strip()]

def _build_popularity_table(animes: pd.DataFrame) -> pd.DataFrame:
    df = animes.drop_duplicates(subset="uid").copy()   

    for col in ("score", "members", "aired_year"):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in animes DataFrame.")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "genre" in df.columns:
        df["genre_list"] = df["genre"].apply(to_list)
    elif "genres" in df.columns:
        df["genre_list"] = df["genres"].apply(to_list)
    else:
        df["genre_list"] = [[] for _ in range(len(df))]

    scaler = MinMaxScaler()
    df[["norm_score", "norm_members"]] = scaler.fit_transform(
        df[["score", "members"]]
    )

    this_year = datetime.now().year
    year_span = (this_year - df["aired_year"]).replace(0, 1)
    df["recency"] = 1 - (year_span / year_span.max())

    df["pop_score"] = (
        WEIGHT_SCORE   * df["norm_score"]   +
        WEIGHT_MEMBERS * df["norm_members"] +
        WEIGHT_RECENCY * df["recency"]
    )

    return df.sort_values("pop_score", ascending=False).reset_index(drop=True)

_POP_TABLE: pd.DataFrame | None = None

def _pop_table() -> pd.DataFrame:
    global _POP_TABLE
    if _POP_TABLE is None:
        _POP_TABLE = _build_popularity_table(get_clean_animes())
    return _POP_TABLE

def cold_start_top_n(n: int = 20) -> pd.DataFrame:
    return _pop_table().head(n)[["uid", "title", "pop_score"]]


def recommend_by_genre(preferred_genres: Sequence[str], n: int = 20) -> pd.DataFrame:

    if not preferred_genres:
        return cold_start_top_n(n)

    wanted = {g.lower().strip() for g in preferred_genres}

    def _matches(row) -> bool:
        glist = row.get("genre_list", [])
        if not isinstance(glist, list):         
            return False
        return any(g.lower().strip() in wanted 
                for g in glist)

    subset = _pop_table().loc[_pop_table().apply(_matches, axis=1)]
    return subset.head(n)[["uid", "title", "pop_score"]] if not subset.empty else cold_start_top_n(n)


def recommend_for_cold_start_profiles(profiles: pd.DataFrame, n: int = 20) -> Dict[str, List[Tuple[int, float]]]:

    mask = profiles["is_cold_start"] | (profiles["favorites_anime"].apply(len) <= 3)
    cold_profiles = profiles[mask]

    if cold_profiles.empty:
        return {}

    top_n = cold_start_top_n(n)
    uid_score_pairs = list(zip(top_n["uid"].tolist(), top_n["pop_score"].tolist()))
    return {pid: uid_score_pairs for pid in cold_profiles["profile"]}

def interactive_recommend(n: int = 20) -> pd.DataFrame:

    print("\n=== Quick Anime Finder ===")
    print("Type a few genres (comma‑separated) or hit ENTER for universal picks.\n")

    raw = input("Genres: ").strip()
    genres = [g.strip() for g in raw.split(",") if g.strip()]

    print() 
    recs = recommend_by_genre(genres, n=n)

    if not genres or recs.equals(cold_start_top_n(n)):
        print("Showing universally popular starters…\n")
    else:
        print(f"Top {len(recs)} picks for {', '.join(genres)}:\n")

    return recs


def _cli_demo(n_users: int = 5, n_recs: int = 10):

    profiles = profile_preprocess(get_clean_profiles())
    recs = recommend_for_cold_start_profiles(profiles, n=n_recs)

    print("\n=== Cold‑Start Demo ===")
    for i, (pid, rec_list) in enumerate(recs.items()):
        if i >= n_users:
            break
        print(f"\nUser {pid} → starter picks:")
        for uid, score in rec_list:
            print(f"  • UID {uid} (blend={score:.3f})")

if __name__ == "__main__":
    df = interactive_recommend(n=20)
    print(df.to_string(index=False))
