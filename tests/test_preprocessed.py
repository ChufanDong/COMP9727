""" RUN: python animeRecommender/tests/test_preprocessed.py """
import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.load_cleaned import (
    get_clean_animes,
    get_clean_profiles,
    get_clean_reviews,
    get_all_cleaned_data
)

class TestPreprocessedAccess(unittest.TestCase):

    def test_get_clean_profiles(self):
        df = get_clean_profiles()
        self.assertFalse(df.empty)
        self.assertIn("favorites_anime", df.columns)
        print(df.head())

    def test_get_all_cleaned_data(self):
        anime_df, profile_df, review_df = get_all_cleaned_data()
        self.assertGreater(len(anime_df), 0)
        self.assertGreater(len(profile_df), 0)
        self.assertGreater(len(review_df), 0)

        print("\nAnime preview:")
        print(anime_df.head())

        print("\nProfile preview:")
        print(profile_df.head())

        print("\nReview preview:")
        print(review_df.head())


if __name__ == "__main__":
    unittest.main()
