import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple
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


class ItemBasedRecommender:
    """
    Item-based collaborative filtering. Make recommendations based on item similarities (cosine) from the binary
    favorites matrix, then for each user aggregate similarity from their liked items, and output top k recommendations
    as required.
    """
    def __init__(self, profiles: pd.DataFrame):
        # mappings
        self.profile_ids = profiles['profile'].tolist()
        self.anime_ids = sorted({uid for favs in profiles['favorites_anime'] for uid in favs})
        anime_idx = {uid: i for i, uid in enumerate(self.anime_ids)}    # anime list
        profile_idx = {pid: i for i, pid in enumerate(self.profile_ids)}    # user list

        # build item × user binary matrix (anime as rows)
        row_idx, col_idx, data = [], [], []
        for _, row in profiles.iterrows():
            pid = row['profile']
            p_i = profile_idx[pid]
            for uid in row['favorites_anime']:
                if uid in anime_idx:
                    row_idx.append(anime_idx[uid])  # item as row
                    col_idx.append(p_i)            # user as column
                    data.append(1)
        self.item_user_matrix = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(len(self.anime_ids), len(self.profile_ids))
        )  # shape: items × users

        # Precompute item-item cosine similarity matrix (item × item)
        self.item_similarity = cosine_similarity(self.item_user_matrix, dense_output=False)

    def recommend_for_user(self, liked_uids: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        valid = [uid for uid in liked_uids if uid in self.anime_ids]
        if not valid:
            return []

        idxs = [self.anime_ids.index(uid) for uid in valid]  # indices of liked items

        # Sum similarities from each liked item to all items
        agg_scores = np.array(self.item_similarity[:, idxs].sum(axis=1)).flatten()

        # Zero out already liked items
        for i in idxs:
            agg_scores[i] = 0.0

        # Build and sort recommendations
        recs = []
        for item_idx, score in enumerate(agg_scores):
            if score > 0:
                recs.append((int(self.anime_ids[item_idx]), float(score)))
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:top_k]


def collaborative_recommend(
    profiles: pd.DataFrame,
    top_k: int = 10
) -> Dict[str, List[tuple]]:
    """
    Generate recommendations for each user profile based on favorite anime using clean item-based CF.
    Returns a dict mapping profile_id -> list of (anime_uid, score).
    """
    recommender = ItemBasedRecommender(profiles)
    recommendations: Dict[str, List[tuple]] = {}
    for row in tqdm(profiles.itertuples(index=False), total=len(profiles), desc="Processing users"):
        recs = recommender.recommend_for_user(row.favorites_anime, top_k)
        recommendations[row.profile] = recs
    return recommendations


if __name__ == "__main__":
    profiles = profile_preprocess(get_clean_profiles())
    print("Profiles loaded and preprocessed.")

    profiles = profiles[profiles['favorites_anime'].apply(lambda favs: len(favs) > 9)].reset_index(drop=True)
    print("Cold Start Users dropped")

    profiles, test = split_profile(profiles, 0.5, 0.5)

    recs = collaborative_recommend(profiles, top_k=10)

    # You can uncomment to inspect individual recommendations:
    # for pid, rec_list in recs.items():
    #     print(f"Recommendations for {pid}:")
    #     for uid, score in rec_list:
    #         print(f"  - UID {uid}, score {score:.4f}")

    # Evaluate (recall in this example)
    precision_results = evaluate_recall_at_k(recs, test, k=10)
    overall_recall = sum(precision_results.values()) / len(precision_results) if precision_results else 0.0
    print(f"Overall Recall at 10: {overall_recall:.4f}")
    print("Evaluation completed.")