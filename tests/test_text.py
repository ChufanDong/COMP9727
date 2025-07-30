import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.load_cleaned import (
    get_clean_animes,
    get_clean_profiles,
    get_clean_reviews,
)

from preprocessing.preprocess_pipline import final_preprocess

def run_pipeline():
    """Run the entire preprocessing pipeline."""
    
    # Load cleaned data
    animes = get_clean_animes()
    profiles = get_clean_profiles()
    reviews = get_clean_reviews()

    # Final preprocessing
    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)

    # Optionally save or return processed data
    return animes, profiles, reviews
if __name__ == "__main__":
    animes, profiles, reviews = run_pipeline()
    
    print("Processed Animes:")
    print(animes)
    print("\nProcessed Profiles:")
    print(profiles)
    print("\nProcessed Reviews:")
    print(reviews)