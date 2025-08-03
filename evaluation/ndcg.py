from typing import List, Dict, Tuple, Union
from collections import defaultdict
import pandas as pd
import ast
import os
import sys
import numpy as np
from sklearn.metrics import ndcg_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def ndcg_at_n(predictions: List[Tuple[int, float]], ground_truth: List[int], n: int) -> float:
    """
    Calculate NDCG at n using the actual prediction scores.
    
    :param predictions: List of (item_id, score) tuples, sorted by score desc
    :param ground_truth: List of true items.
    :param n: The number of top items to consider for NDCG calculation.
    :return: NDCG at n.
    """
    if not ground_truth or n <= 0:
        return 0.0
    
    # Consider only the top n predictions
    pred_at_n = predictions[:n]
    if not pred_at_n:
        return 0.0
    
    ground_truth_set = set(ground_truth)
    
    # Extract items and scores
    pred_items = [item for item, _ in pred_at_n]
    pred_scores = [score for _, score in pred_at_n]
    
    # Create relevance scores (1 for relevant, 0 for irrelevant)
    relevance_scores = [1.0 if item in ground_truth_set else 0.0 for item in pred_items]
    
    # Check if there are any relevant items in predictions
    if not any(rel > 0 for rel in relevance_scores):
        return 0.0
    
    # Use prediction scores as y_score and relevance as y_true
    y_true = np.array([relevance_scores])
    y_score = np.array([pred_scores])
    
    return ndcg_score(y_true, y_score, k=n)


def batch_ndcg_at_n(predictions: Dict[str, List[Tuple[int, float]]], 
                   ground_truth: Dict[str, List[int]], 
                   n: int) -> Dict[str, float]:
    """
    Calculate NDCG at n for multiple users efficiently.
    """
    ndcgs = {}
    for user, preds in predictions.items():
        true_items = ground_truth.get(user, [])
        ndcgs[user] = ndcg_at_n(preds, true_items, n)
    
    return ndcgs

def evaluate_ndcg_at_k(predictions: Dict[str, List[Tuple[int, float]]], 
                      ground_truth: pd.DataFrame, 
                      k: int) -> Dict[str, float]:
    """
    Evaluate NDCG at k for all users.
    """
    def parse_anime_ids(anime_ids_str: Union[str, list]) -> List[int]:
        """Parse anime IDs from string representation or list."""
        if isinstance(anime_ids_str, str):
            try:
                # Handle string representation of list
                parsed = ast.literal_eval(anime_ids_str)
                return [int(x) for x in parsed] if isinstance(parsed, list) else [int(parsed)]
            except (ValueError, SyntaxError):
                # Handle comma-separated string
                return [int(x.strip()) for x in anime_ids_str.split(',') if x.strip()]
        elif isinstance(anime_ids_str, list):
            return [int(x) for x in anime_ids_str]
        else:
            return [int(anime_ids_str)]
    
    # Process ground truth more efficiently
    ground_truth_dict = {}
    for _, row in ground_truth.iterrows():
        user_id = str(row['profile'])
        anime_ids = parse_anime_ids(row['favorites_anime'])
        ground_truth_dict[user_id] = anime_ids
    
    return batch_ndcg_at_n(predictions, ground_truth_dict, k)

# def main():
#     # Load the cleaned profiles and animes
#     profiles = get_clean_profiles().drop_duplicates(['profile'])
#     animes = get_clean_animes()
#     # Preprocess the animes
#     animes = anime_preprocess(animes)

#     # Split the dataset into train and test sets
#     train_profiles, test_profiles = split_profile(profiles, 0.8, 0.2)

#     # Initialize the content-based recommender
#     recommender = ContentBasedRecommender(animes)

#     # Generate recommendations for each user in the test set
#     content_based_recommendations = content_based_recommend(recommender, animes, train_profiles)

#     # Evaluate NDCG at k
#     ndcg_results = evaluate_ndcg_at_k(content_based_recommendations, test_profiles, k=5)
    
#     # Print NDCG results
#     for user, ndcg in ndcg_results.items():
#         print(f"User {user}: NDCG at 5 = {ndcg:.4f}")
#     # print overall NDCG
#     overall_ndcg = sum(ndcg_results.values()) / len(ndcg_results) if ndcg_results else 0.0
#     print(f"Overall NDCG at 5: {overall_ndcg:.4f}")
#     print("Evaluation completed.")

# if __name__ == "__main__":
#     main()
