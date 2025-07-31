import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import clean
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

def _drop_ids_fromlist(fav_anime: list, ani_ids: set):
    """Remove anime IDs from favorites list that don't exist in animes dataset."""
    if not fav_anime:
        return fav_anime
    return [int(anime_id) for anime_id in fav_anime if int(anime_id) in ani_ids]

def drop_unreachable_animes(profiles: pd.DataFrame, animes: pd.DataFrame):
    """Remove anime IDs from profiles' favorites_anime that don't exist in animes dataset."""
    animes_ids = set(animes["uid"].tolist())
    profiles['favorites_anime'] = profiles['favorites_anime'].apply(
        lambda x: _drop_ids_fromlist(x, animes_ids)
    )
    # debugging
    # print("[preprocessing] Dropping unreachable animes from profiles...")
    # print(profiles)
    profiles["favorites_count"] = profiles["favorites_anime"].apply(len)
    profiles["is_cold_start"] = profiles["favorites_count"] < 3

    profiles = profiles[profiles["favorites_count"] > 0]  #去除无 favorites 的用户

    # debugging
    # print("[preprocessing] Unreachable animes dropped from profiles.")
    # print(profiles)
    return profiles

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

    profiles = drop_unreachable_animes(profiles, animes)

    return animes, profiles, reviews


if __name__ == "__main__":
    # Example usage
    animes, profiles, reviews = load_cleaned.get_all_cleaned_data()

    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)
    print("Processed Animes:")
    print(animes.head())
    print("\nProcessed Profiles:")
    print(profiles['is_cold_start'].value_counts())
    print(profiles.head())
    print("\nProcessed Reviews:")
    print(reviews.head())

