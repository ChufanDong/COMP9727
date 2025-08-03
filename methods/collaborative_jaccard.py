import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import os
import sys

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


def compute_item_jaccard_similarity(ratings_matrix: np.ndarray) -> np.ndarray:
    """
    ratings_matrix: binary (items x users) numpy array
    Returns item-item Jaccard similarity matrix.
    """
    bin_mat = (ratings_matrix > 0).astype(int)
    intersection = bin_mat @ bin_mat.T  # (items x items)
    item_sums = bin_mat.sum(axis=1)  # per item
    union = item_sums[:, None] + item_sums[None, :] - intersection  # broadcasting
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard = intersection / union
        jaccard[union == 0] = 0.0
    return jaccard  # shape: (n_items, n_items)


class JaccardCFRecommender:
    """
    Item-based collaborative filtering using Jaccard similarity on binary favorites.
    """
    def __init__(
        self,
        profiles: pd.DataFrame
    ):
        self.profile_ids = profiles['profile'].tolist()
        self.anime_ids = sorted({uid for favs in profiles['favorites_anime'] for uid in favs})

        # index mappings
        anime_idx = {uid: i for i, uid in enumerate(self.anime_ids)}
        profile_idx = {pid: i for i, pid in enumerate(self.profile_ids)}

        num_anime = len(self.anime_ids)
        num_profiles = len(self.profile_ids)

        # Build binary item-user matrix (items x users)
        bin_matrix = np.zeros((num_anime, num_profiles), dtype=int)
        for _, row in profiles.iterrows():
            pid = row['profile']
            p_i = profile_idx[pid]
            for uid in row['favorites_anime']:
                if uid in anime_idx:
                    bin_matrix[anime_idx[uid], p_i] = 1

        self.bin_matrix = bin_matrix  # keep for reference
        # Precompute item-item Jaccard similarity
        self.similarity_matrix = compute_item_jaccard_similarity(self.bin_matrix)

        # Precompute liked indices per profile
        self.profile_likes_idxs: Dict[str, List[int]] = {}
        for _, row in profiles.iterrows():
            valid = [uid for uid in row['favorites_anime'] if uid in self.anime_ids]
            self.profile_likes_idxs[row['profile']] = [self.anime_ids.index(uid) for uid in valid]

    def recommend_similar_items(self, target_uid: int, top_k: int = 10) -> List[tuple]:
        if target_uid not in self.anime_ids:
            raise ValueError(f"Anime UID {target_uid} not found.")
        idx = self.anime_ids.index(target_uid)
        sims = self.similarity_matrix[idx]  # similarities to all items
        candidates = []
        for i, sim in enumerate(sims):
            if i == idx:
                continue
            candidates.append((int(self.anime_ids[i]), float(sim)))
        # sort by similarity
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        return candidates

    def recommend_for_user(self, liked_uids: List[int], top_k: int = 10) -> List[tuple]:
        valid = [uid for uid in liked_uids if uid in self.anime_ids]
        if not valid:
            return []

        liked_idxs = [self.anime_ids.index(uid) for uid in valid]
        # Score candidate items by average Jaccard similarity to all liked items
        scores = defaultdict(list)  # item -> list of sims from each liked
        for liked_idx in liked_idxs:
            sims = self.similarity_matrix[liked_idx]  # similarity of this liked item to all
            for i, sim in enumerate(sims):
                if i in liked_idxs:
                    continue  # skip already liked
                scores[i].append(sim)

        aggregated = []
        for item_idx, sim_list in scores.items():
            if not sim_list:
                continue
            avg_sim = float(np.mean(sim_list))
            aggregated.append((int(self.anime_ids[item_idx]), avg_sim))

        # sort and take top_k
        aggregated = sorted(aggregated, key=lambda x: x[1], reverse=True)[:top_k]
        return aggregated


def collaborative_recommend(
    profiles: pd.DataFrame,
    top_k: int = 10
) -> Dict[str, List[tuple]]:
    recommender = JaccardCFRecommender(profiles)
    recommendations = {}
    for row in tqdm(profiles.itertuples(index=False), total=len(profiles), desc="Processing users"):
        recs = recommender.recommend_for_user(row.favorites_anime, top_k)
        recommendations[row.profile] = recs
    return recommendations


if __name__ == "__main__":
    profiles = profile_preprocess(get_clean_profiles())
    print("Profiles loaded and preprocessed.")

    profiles = profiles[profiles['favorites_anime'].apply(lambda favs: len(favs) > 3)].reset_index(drop=True)
    print("Cold Start Users dropped")

    profiles, test = split_profile(profiles, 0.5, 0.5)
    print("Split applied.")

    recs = collaborative_recommend(profiles, top_k=50)

    precision_results = evaluate_precision_at_k(recs, test, k=10)
    precision_results = evaluate_recall_at_k(recs, test, k=10)

    overall_precision = sum(precision_results.values()) / len(precision_results) if precision_results else 0.0
    print(f"Overall Precision at 5: {overall_precision:.4f}")
    print("Evaluation completed.")