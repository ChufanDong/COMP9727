import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import load_cleaned
from preprocessing import text_preprocess

def anime_preprocess(animes: pd.DataFrame):
    """Process anime data."""

    # Process synopsis and aired dates
    print("[preprocessing] Processing anime data...")
    animes['synopsis'] = animes['synopsis'].apply(text_preprocess.text_process)
    animes['duration'] = animes['aired'].apply(text_preprocess.transform_quarter)
    print("[preprocessing] Anime data processed successfully.")
    
    return animes

def profile_preprocess(profiles: pd.DataFrame):
    """Process profile data."""
    print("[preprocessing] Processing profile data...")
    print("[preprocessing] Profile data processed successfully.")
    return profiles

def review_preprocess(reviews: pd.DataFrame):
    """Process review data."""
    print("[preprocessing] Processing review data...")
    print("[preprocessing] Review data processed successfully.")
    return reviews

def final_preprocess(animes: pd.DataFrame, profiles: pd.DataFrame, reviews: pd.DataFrame):
    """Output the final cleaned datasets."""
    
    animes = anime_preprocess(animes)
    profiles = profile_preprocess(profiles)
    reviews = review_preprocess(reviews)

    return animes, profiles, reviews


if __name__ == "__main__":
    # Example usage
    animes, profiles, reviews = load_cleaned.get_all_cleaned_data()

    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)
    print("Processed Animes:")
    print(animes.head())
    print("\nProcessed Profiles:")
    print(profiles.head())
    print("\nProcessed Reviews:")
    print(reviews.head())

