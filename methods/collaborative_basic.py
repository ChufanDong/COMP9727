import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from typing import List, Optional, Dict, Any

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
from scipy.sparse import csr_matrix  # Sparse matrix support
from tqdm import tqdm  # Progress bar


def collaborative_recommend(
    profiles: pd.DataFrame,
    top_k: int = 10
) -> Dict[str, List[tuple]]:
    """
    Generate recommendations for each user profile based on favorite anime.
    Returns a dict mapping profile_id -> list of (anime_uid, score).
    """
    # drop cold start users
    profiles = profiles[profiles['favorites_anime'].apply(lambda favs: len(favs) > 3)].reset_index(drop=True)

    recommender = CollaborativeFilteringRecommender(profiles)
    recommendations = {}
    # Wrap the iteration in a progress bar
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
        nn_params = nn_params or {'n_neighbors': 10, 'metric': 'cosine', 'algorithm': 'auto'}

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
        self.nn_model = NearestNeighbors(algorithm='brute', metric='cosine')
        self.nn_model.fit(self.ratings_matrix)

    def recommend_similar_items(self, target_uid: int, top_k: int = 10) -> List[tuple]:
        """
        Given an anime UID, recommend top_k similar anime based on binary favorites.
        Returns a list of (anime_uid, similarity_score).
        """
        if target_uid not in self.anime_ids:
            raise ValueError(f"Anime UID {target_uid} not found.")

        idx = self.anime_ids.index(target_uid)
        query_vec = self.ratings_matrix[idx].reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=top_k + 1)

        sims = 1 - distances.flatten()
        recs = []
        for sim, i in zip(sims, indices.flatten()):
            uid = self.anime_ids[i]
            if uid == target_uid:
                continue
            recs.append((uid, float(sim)))
            if len(recs) >= top_k:
                break
        return recs

    def recommend_for_user(self, liked_uids: List[int], top_k: int = 10) -> List[tuple]:
        """
        Recommend anime for a user given their list of favorite UIDs.
        Returns a list of (anime_uid, aggregated_similarity).
        """
        valid = [uid for uid in liked_uids if uid in self.anime_ids]
        if not valid:
            return []

        idxs = [self.anime_ids.index(uid) for uid in valid]
        user_vec = self.ratings_matrix[idxs].sum(axis=0)
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
            recs.append((uid, float(sim)))
            if len(recs) >= top_k:
                break
        return recs


if __name__ == "__main__":
    profiles = profile_preprocess(get_clean_profiles())
    print("Profiles loaded and preprocessed.")

    recs = collaborative_recommend(profiles, top_k=10)
    for pid, rec_list in recs.items():
        print(f"Recommendations for {pid}:")
        for uid, score in rec_list:
            print(f"  - UID {uid}, score {score:.4f}")
