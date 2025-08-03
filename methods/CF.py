import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from surprise import accuracy
from collections import defaultdict
from typing import Optional, Dict, Any, List
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import json
from preprocessing.load_cleaned import (
    get_clean_animes,
    get_clean_profiles,
    get_clean_reviews,
)
from preprocessing.preprocess_pipline import (
    anime_preprocess, 
    profile_preprocess, 
    review_preprocess,
    final_preprocess
)



import pandas as pd

def build_svd_training(
    reviews: pd.DataFrame,
    train_profiles: pd.DataFrame,
    test_profiles: pd.DataFrame,
    score_for_fav: int = 7
) -> pd.DataFrame:
    """
    Build enhanced rating dataset suitable for SVD training:
    1. Remove ratings not in training user list
    2. Assign fixed scores to training users' favorite anime list
    3. Remove ratings that might leak test set favorites

    Parameters
    ----------
    reviews : pd.DataFrame
        Original rating data, at least contains ['uid','profile','anime_uid','score']
    train_profiles : pd.DataFrame
        Training set user favorites DataFrame, at least contains ['profile','favorites_anime']
    test_profiles : pd.DataFrame
        Test set user favorites DataFrame
    score_for_fav : int
        Fixed rating for training set favorite anime (default 7)

    Returns
    -------
    pd.DataFrame
        Enhanced clean rating data, ready for SVD training
    """

    # 1️⃣ Keep only training set users
    train_users = set(train_profiles['profile'])
    reviews_filtered = reviews[reviews['profile'].isin(train_users)].copy()

    # 2️⃣ Build fixed ratings for training set favorites
    extra_rows = []
    for _, row in train_profiles.iterrows():
        user = row['profile']
        for anime in row['favorites_anime']:
            extra_rows.append({
                'uid': None,  # Optional: SVD doesn't depend on uid
                'profile': user,
                'anime_uid': int(anime),
                'score': score_for_fav
            })
    train_extra_df = pd.DataFrame(extra_rows)

    # 3️⃣ Merge enhanced ratings
    augmented_reviews = pd.concat([reviews_filtered, train_extra_df], ignore_index=True)

    # Same user-anime duplicate ratings: keep original rating, discard 7-score padding
    augmented_reviews = augmented_reviews.drop_duplicates(
        subset=['profile', 'anime_uid'],
        keep='first'
    )

    # 4️⃣ Remove test set conflicting ratings (avoid data leakage)
    test_conflicts = set()
    for _, row in test_profiles.iterrows():
        user = row['profile']
        for anime in row['favorites_anime']:
            test_conflicts.add((user, int(anime)))

    mask_conflict = augmented_reviews.apply(
        lambda x: (x['profile'], x['anime_uid']) in test_conflicts, axis=1
    )
    cleaned_reviews = augmented_reviews[~mask_conflict].reset_index(drop=True)

    cleaned_reviews = cleaned_reviews.drop_duplicates().reset_index(drop=True)

    print("Original rating count:", len(reviews))
    print("Training user rating count:", len(reviews_filtered))
    print("Enhanced rating count:", len(augmented_reviews))
    print("Cleaned rating count after conflict removal:", len(cleaned_reviews))

    return cleaned_reviews

from surprise import SVD, Dataset, Reader

def train_svd_model(
    training_data,
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    biased=True,
    random_state=42
):
    """
    Train rating matrix using Surprise's SVD model

    Parameters
    ----------
    training_data : pd.DataFrame
        At least contains ['profile','anime_uid','score'] three columns
    n_factors : int
        Number of latent factors
    n_epochs : int
        Number of training iterations
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    model : surprise.SVD
        Trained SVD model
    trainset : surprise.Trainset
        Surprise internal training set, can be used for prediction
    """
    # Build Surprise dataset
    reader = Reader(rating_scale=(training_data['score'].min(), training_data['score'].max()))
    data = Dataset.load_from_df(training_data[['profile','anime_uid','score']], reader)
    trainset = data.build_full_trainset()

    # Train SVD model
    model = SVD(n_factors=n_factors,n_epochs=n_epochs,lr_all=lr_all,reg_all=reg_all,biased=biased,random_state=random_state)
    model.fit(trainset)

    return model, trainset

from methods.cold_start import recommend_for_cold_start_profiles

def recommend_for_user(model, trainset, user_id, top_k=10, cold_start_recs=None):
    """
    Generate Top-K recommendation list for a specific user
    """
    if cold_start_recs and user_id in cold_start_recs:
        return cold_start_recs[user_id][:top_k]
    # Get all items (original IDs) in the training set
    all_items = set(trainset._raw2inner_id_items.keys())

    # Get items rated by the user
    try:
        rated_items = {
            trainset.to_raw_iid(inner_iid)
            for (inner_uid, inner_iid, _) in trainset.all_ratings()
            if trainset.to_raw_uid(inner_uid) == user_id
        }

    except ValueError:
        # User does not exist in the training set (cold start)
        return cold_start_recs.get(user_id, []) if cold_start_recs else []

    # Set of unrated items
    candidates = all_items - rated_items

    # Predict scores for unrated items
    predictions = [(iid, model.predict(user_id, iid).est) for iid in candidates]

    # Sort by predicted scores in descending order, take Top-K
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]

def recommend_for_all_users(model, trainset, train_profiles, top_k=10):
    """
    Generate Top-K recommendation dictionary for all users in training set
    
    Returns
    -------
    dict[str, list[tuple[int,float]]]
        {profile: [(anime_uid, predicted_score), ...]}
    """
    cold_start_recs = recommend_for_cold_start_profiles(train_profiles, n=top_k)
    user_recommendations = {}

    # Iterate through all user original IDs in training set
    all_users = [trainset.to_raw_uid(inner_id) for inner_id in trainset.all_users()]

    for user_id in tqdm(all_users, desc="Generating Recommendations"):
        recs = recommend_for_user(model, trainset, user_id, top_k=top_k, cold_start_recs=cold_start_recs)
        user_recommendations[user_id] = recs

    return user_recommendations

def recommend_for_all_users_fast(model, trainset, train_profiles, top_k=10):
    """
    Batch generate Top-K recommendations for all users
    Cold start users directly use popular Top-K recommendations
    """
    import numpy as np
    from methods.cold_start import recommend_for_cold_start_profiles

    num_users, num_items = trainset.n_users, trainset.n_items

    # 1️⃣ First generate cold start user recommendation dictionary
    cold_start_recs = recommend_for_cold_start_profiles(train_profiles, n=top_k)
    cold_start_users = set(cold_start_recs.keys())

    # 2️⃣ Calculate complete prediction matrix (only for non-cold-start users)
    pred_matrix = np.dot(model.pu, model.qi.T)
    if model.biased:  # Only add bias when biased=True
        pred_matrix += model.trainset.global_mean
        pred_matrix += model.bu[:, np.newaxis]
        pred_matrix += model.bi[np.newaxis, :]

    # 3️⃣ Clip to rating range, ensure consistency with model.predict
    min_rating, max_rating = trainset.rating_scale
    pred_matrix = np.clip(pred_matrix, min_rating, max_rating)

    # 3️⃣ Mask already rated items
    rated_mask = np.zeros_like(pred_matrix, dtype=bool)
    for inner_uid, inner_iid, _ in trainset.all_ratings():
        rated_mask[inner_uid, inner_iid] = True
    pred_matrix[rated_mask] = -np.inf

    # 4️⃣ Batch Top-K recommendations
    top_k_indices = np.argpartition(-pred_matrix, top_k, axis=1)[:, :top_k]
    row_indices = np.arange(num_users)[:, None]
    top_k_sorted_idx = top_k_indices[
        row_indices, np.argsort(-pred_matrix[row_indices, top_k_indices])
    ]

    # 5️⃣ Build recommendation dictionary
    recommendations = {}
    for inner_uid in range(num_users):
        user_id = trainset.to_raw_uid(inner_uid)

        # If cold start user → directly use popular recommendations
        if user_id in cold_start_users:
            recommendations[user_id] = cold_start_recs[user_id][:top_k]
            continue

        # Non-cold-start user → SVD matrix recommendations
        top_items = top_k_sorted_idx[inner_uid]
        recs = [
            (int(trainset.to_raw_iid(inner_iid)), float(pred_matrix[inner_uid, inner_iid]))
            for inner_iid in top_items
        ]
        recommendations[user_id] = recs

    return recommendations


def SVD_recommend(train_profiles, test_prefiles, reviews, top_k=10):
    """
    Automated SVD recommendation pipeline, generate Top-K recommendation dictionary
    """

    train_profiles = train_profiles[train_profiles['is_cold_start'] == False].reset_index(drop=True)
    test_profiles = test_prefiles[test_prefiles['is_cold_start'] == False].reset_index(drop=True)
    reviews = reviews

    # 4️⃣ Build SVD training data
    training_data = build_svd_training(
        reviews=reviews,
        train_profiles=train_profiles,
        test_profiles=test_profiles,
        score_for_fav=8
    )
    print("Final SVD training set size:", len(training_data))

    # 5️⃣ Train SVD model
    model, trainset = train_svd_model(training_data, n_factors=150, n_epochs=30, lr_all=0.005, reg_all=0.05, biased=False, random_state=42)

    # 6️⃣ Generate recommendation dictionary for all users
    recommendations = recommend_for_all_users_fast(model, trainset, train_profiles, top_k=top_k)
    print(f"Generated Top-{top_k} recommendations for {len(recommendations)} users in total")

    return recommendations

def auto_recommend_dump(top_k=10):
    # 1️⃣ Load basic cleaned data
    profiles = get_clean_profiles()
    reviews = get_clean_reviews()

    # 2️⃣ Secondary preprocessing
    profiles = profile_preprocess(profiles)
    reviews = review_preprocess(reviews)

    # 3️⃣ Select necessary columns
    #profiles = profiles[['profile', 'favorites_anime']]
    #reviews = reviews[['uid', 'profile', 'anime_uid', 'score']]

    # 4️⃣ Split training and test sets
    from preprocessing.split_dataset import split_profile
    train_profiles, test_profiles = split_profile(profiles, train_size=0.5, test_size=0.5)

    # 5️⃣ Build SVD training data
    training_data = build_svd_training(
        reviews=reviews,
        train_profiles=train_profiles,
        test_profiles=test_profiles,
        score_for_fav=8
    )
    print("Final SVD training set size:", len(training_data))

    # 6️⃣ Train SVD model
    model, trainset = train_svd_model(training_data,n_factors=150,n_epochs=30,lr_all=0.005,reg_all=0.05,biased=False,random_state=42)

    testset = []
    for _, row in test_profiles.iterrows():
        user_id = str(row['profile'])
        for anime_id in row['favorites_anime']:
            testset.append((user_id, str(anime_id), 8))  # Assume favorite equals perfect score

    # Predict and calculate RMSE
    from surprise import accuracy
    pred = model.test(testset)
    rmse = accuracy.rmse(pred, verbose=True)
    print(f"RMSE on split test set favorites: {rmse:.4f}")

    # 7️⃣ Generate recommendation dictionary for all users
    #recommendations = recommend_for_all_users(model, trainset, train_profiles, top_k=top_k)
    recommendations = recommend_for_all_users_fast(model, trainset, train_profiles, top_k=top_k)
    print(f"Generated Top-{top_k} recommendations for {len(recommendations)} users in total")

    # 8️⃣ Optional: Print examples
    for user, recs in list(recommendations.items())[:3]:
        print(f"{user}: {recs}")

    with open("methods/svd_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False)

    return recommendations

def read_json_recommendations(file_path: str) -> Dict[str, List]:
    """
    Read recommendation results from JSON file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    rec_dict = auto_recommend_dump(top_k=10)

    with open("methods/svd_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(rec_dict, f, ensure_ascii=False)
