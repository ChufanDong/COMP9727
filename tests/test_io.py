import unittest
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IO.load_csv import load_anime_data, load_profile_data, load_review_data


class TestIOModule(unittest.TestCase):

    def test_load_anime_data(self):
        anime_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "archive", "animes.csv"))
        df = load_anime_data(anime_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("title", df.columns)
        self.assertGreater(len(df), 1000)
        print(df.head(10)) 

    def test_load_profile_data(self):
        profile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "archive", "profiles.csv"))
        df = load_profile_data(profile_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("favorites_anime", df.columns)
        self.assertGreater(len(df), 1000)
        print(df.head(10)) 

    def test_load_review_data(self):
        review_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "archive", "reviews.csv"))
        df = load_review_data(review_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 1000)
        print(df.head(10)) 



if __name__ == "__main__":
    unittest.main()
