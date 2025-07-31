import pandas as pd
import ast
import re

def clean_animes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["synopsis"] = df["synopsis"].fillna("")
    df["synopsis"] = df["synopsis"].apply(
        lambda x: re.sub(r"\s+", " ", x).strip()
    )

    df["score"] = df["score"].fillna(-1)
    df["ranked"] = df["ranked"].fillna(-1)
    df["episodes"] = df["episodes"].fillna(0)

    df["genre"] = df["genre"].fillna("")

    df["genre"] = df["genre"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    )

    df["aired_year"] = df["aired"].str.extract(r'(\d{4})')

    """ print(df["genre"].iloc[0][0])
    print(type(df["genre"].iloc[0])) """

    return df


def clean_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_parse_list(x):
        try:
            if isinstance(x, str):
                return ast.literal_eval(x)
            elif isinstance(x, list):
                return x
            else:
                return []
        except:
            return []

    df["favorites_anime"] = df["favorites_anime"].apply(safe_parse_list)
    df["favorites_count"] = df["favorites_anime"].apply(len)
    df["is_cold_start"] = df["favorites_count"] < 3
    df["gender"] = df["gender"].fillna("NULL")
    df["birthday"] = df["birthday"].fillna("NULL")

    df = df[df["favorites_count"] > 0]  #去除无 favorites 的用户

    """ print(df["favorites_anime"].iloc[0][0])
    print(type(df["favorites_anime"].iloc[0])) """

    df = df.drop_duplicates(subset=["profile"])

    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    expected_cols = ["uid","profile","anime_uid","score","scores"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    
    df = df[expected_cols]
    df = df.dropna()
    df["score"] = df["score"].astype(int)

    return df
