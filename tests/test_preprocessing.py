""" RUN: python animeRecommender/tests/test_preprocessing.py """
import unittest
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IO.load_csv import load_anime_data, load_profile_data, load_review_data
from preprocessing.clean import clean_animes, clean_profiles, clean_reviews


def inspect_dataframe(df: pd.DataFrame, title: str = "DataFrame Inspection", n_rows: int = 1):
    print(f"\n----- {title} -----")
    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"Type    : {df[col].dtype}")
        print(f"Missing : {df[col].isnull().sum()} out of {len(df)}")
        print(f"Sample  : {df[col].head(n_rows).tolist()}")


class TestPreprocessing(unittest.TestCase):

    def test_animes_clean_and_inspect(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "archive", "animes.csv"))
        df = clean_animes(load_anime_data(path))

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("aired_year", df.columns)

        print("\n----- Cleaned Anime Data Preview -----")
        print(df.head())

        inspect_dataframe(df, "Auto Inspect: Cleaned Anime Columns")

    def test_profiles_clean_and_inspect(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "archive", "profiles.csv"))
        df = clean_profiles(load_profile_data(path))

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("favorites_anime", df.columns)
        self.assertIn("favorites_count", df.columns)
        self.assertIn("is_cold_start", df.columns)
        self.assertTrue(isinstance(df["favorites_anime"].iloc[0], list))
        self.assertEqual((df["favorites_count"] == 0).sum(), 0)

        print("\n----- Cleaned Profiles Preview -----")
        print(df.head())

        inspect_dataframe(df, "Auto Inspect: Cleaned Profiles")

    def test_reviews_clean_and_inspect(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "archive", "reviews.csv"))
        df = clean_reviews(load_review_data(path))

        self.assertIn("uid", df.columns)
        self.assertIn("anime_uid", df.columns)
        self.assertIn("score", df.columns)
        self.assertEqual(df["score"].dtype, "int32")
        self.assertEqual(df.isnull().sum().sum(), 0)

        print("\n----- Cleaned Reviews Preview -----")
        print(df.head())

        inspect_dataframe(df, "Auto Inspect: Cleaned Reviews")


if __name__ == "__main__":
    unittest.main()
