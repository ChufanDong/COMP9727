import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.load_cleaned import get_clean_animes, get_clean_profiles, get_clean_reviews
import pandas as pd
from typing import List, Tuple, Optional
def split_profile(profile: pd.DataFrame, 
                  train_size: float = 0.8,
                  test_size: float = 0.2, 
                  ) -> Tuple[pd.DataFrame]:
    if train_size + test_size != 1.0:
        raise ValueError("train_size and test_size must sum to 1.0")
        
    train_profiles = []
    test_profiles = []

    for idx, row in profile.iterrows():
        favorites = row['favorites_anime']
        if isinstance(favorites, list) and len(favorites) > 0:
            n_train = int(len(favorites) * train_size)
            if n_train == 0:
                n_train = 1
            
            train_favorites = favorites[:n_train]
            test_favorites = favorites[n_train:]
            
            # Create train profile
            train_row = row.copy()
            train_row['favorites_anime'] = train_favorites
            train_profiles.append(train_row)
            
            # Create test profile
            test_row = row.copy()
            test_row['favorites_anime'] = test_favorites
            test_profiles.append(test_row)

    train_df = pd.DataFrame(train_profiles)
    test_df = pd.DataFrame(test_profiles)

    return train_df, test_df

def split_test():
    # Load the cleaned profiles
    profiles = get_clean_profiles()

    # Split the profiles into train and test sets
    train_profiles, test_profiles = split_profile(profiles)


if __name__ == "__main__":
    split_test()