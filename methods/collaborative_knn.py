import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from typing import List, Optional, Dict, Any
from scipy.sparse import csr_matrix  # Sparse matrix support
from tqdm import tqdm  # Progress bar

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
from preprocessing.split_dataset import split_profile
from evaluation.precision_at_k import evaluate_precision_at_k
from evaluation.recall_at_k import evaluate_recall_at_k

def collaborative_recommend(
    profiles: pd.DataFrame,
    top_k: int = 10
) -> Dict[str, List[tuple]]:
    """
    Generate recommendations for each user profile based on favorite anime.
    Returns a dict mapping profile_id -> list of (anime_uid, score).
    """
    # Drop cold start users
    # profiles = profiles[profiles['favorites_anime'].apply(lambda favs: len(favs) > 3)].reset_index(drop=True)

    recommender = CollaborativeFilteringRecommender(profiles)
    recommendations = {}
    #print(f'The current k value is {top_k}.')
    # Progress bar
    for row in tqdm(profiles.itertuples(index=False), total=len(profiles), desc="Processing users"):
        recs = recommender.recommend_for_user(row.favorites_anime, top_k)
        recommendations[row.profile] = recs
    return recommendations


class CollaborativeFilteringRecommender:
    """
    Item-based collaborative filtering using binary preference matrix (favorites).
    """
    def __init__(
        self,
        profiles: pd.DataFrame,
        nn_params: Optional[Dict[str, Any]] = None
    ):
        nn_params = nn_params or {'metric': 'cosine', 'algorithm': 'auto'}

        # Extract user IDs and all unique anime IDs from favorites
        self.profile_ids = profiles['profile'].tolist()
        self.anime_ids = sorted({uid for favs in profiles['favorites_anime'] for uid in favs})

        # Create index mappings
        anime_idx = {uid: i for i, uid in enumerate(self.anime_ids)}
        profile_idx = {pid: i for i, pid in enumerate(self.profile_ids)}

        # Build sparse binary preference matrix (anime x user)
        row_idx, col_idx, data = [], [], []
        for _, row in profiles.iterrows():
            pid = row['profile']
            p_i = profile_idx[pid]
            for uid in row['favorites_anime']:
                if uid in anime_idx:
                    row_idx.append(anime_idx[uid])
                    col_idx.append(p_i)
                    data.append(1)

        self.ratings_matrix = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(len(self.anime_ids), len(self.profile_ids))
        )
        self.nn_model = NearestNeighbors(algorithm=nn_params['algorithm'], metric=nn_params['metric']) # set metrics
        self.nn_model.fit(self.ratings_matrix) 

    def recommend_for_user(self, liked_uids: List[int], top_k: int = 10) -> List[tuple]:
        """
        Recommend anime for a user given their list of favorite UIDs.
        Returns a list of (anime_uid, aggregated_similarity).
        """
        valid = [uid for uid in liked_uids if uid in self.anime_ids]
        #if not valid:
        #    return []

        idxs = [self.anime_ids.index(uid) for uid in valid]
        #user_vec = self.ratings_matrix[idxs].sum(axis=0)   # compare between sum and mean
        user_vec = self.ratings_matrix[idxs].mean(axis=0)
        user_vec = np.array(user_vec).reshape(1, -1)

        distances, indices = self.nn_model.kneighbors(
            user_vec, n_neighbors=top_k + len(valid)
        )
        sims = 1 - distances.flatten()

        recs = []
        for sim, i in zip(sims, indices.flatten()):
            uid = self.anime_ids[i]
            if uid in valid:
                continue
            recs.append((int(uid), float(sim)))
            if len(recs) >= top_k:
                break
        return recs


if __name__ == "__main__":
    profiles = profile_preprocess(get_clean_profiles())
    print("Profiles loaded and preprocessed.")

    profiles = profiles[profiles['favorites_anime'].apply(lambda favs: len(favs) > 3)].reset_index(drop=True)
    print("Cold Start Users dropped")

    profiles, test = split_profile(profiles, 0.5, 0.5)

    recs = collaborative_recommend(profiles, top_k=50)
    #for pid, rec_list in recs.items():                             # print the result in terminal
    #    print(f"Recommendations for {pid}:")
    #    for uid, score in rec_list:
    #        print(f"  - UID {uid}, score {score:.4f}")

    #precision_results = evaluate_precision_at_k(recs, test, k=5)
    precision_results = evaluate_recall_at_k(recs, test, k=10)

    overall_precision = sum(precision_results.values()) / len(precision_results) if precision_results else 0.0
    print(f"Overall Precision at 5: {overall_precision:.4f}")
    print("Evaluation completed.")
