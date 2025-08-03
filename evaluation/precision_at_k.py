from typing import List, Dict, Tuple, Union
from collections import defaultdict
import pandas as pd
import ast
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def precision_at_n(predictions: List[int], ground_truth: List[int], n: int) -> float:
    """
    Calculate precision at n for a set of predictions.
    
    :param predictions: List of predicted items.
    :param ground_truth: List of true items.
    :param n: The number of top items to consider for precision calculation.
    :return: Precision at n.
    """
    if not ground_truth or n <= 0:
        return 0.0
    
    # Consider only the top n predictions
    pred_at_n = predictions[:n]
    if not pred_at_n:
        return 0.0
    
    # Use set intersection for faster lookup
    ground_truth_set = set(ground_truth)
    correct_predictions = sum(1 for item in pred_at_n if item in ground_truth_set)
    
    return correct_predictions / len(pred_at_n)


def batch_precision_at_n(predictions: Dict[str, List[Tuple[int, float]]], 
                        ground_truth: Dict[str, List[int]], 
                        n: int) -> Dict[str, float]:
    """
    Calculate precision at n for multiple users efficiently.
    """
    precisions = {}
    for user, preds in predictions.items():
        true_items = ground_truth.get(user, [])
        pred_items = [item for item, _ in preds]
        precisions[user] = precision_at_n(pred_items, true_items, n)
    
    return precisions

def evaluate_precision_at_k(predictions: Dict[str, List[Tuple[int, float]]], 
                           ground_truth: pd.DataFrame, 
                           k: int) -> Dict[str, float]:
    """
    Evaluate precision at k for all users.
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
    
    return batch_precision_at_n(predictions, ground_truth_dict, k)


# def main():
#     # Load the cleaned profiles and animes
#     profiles = get_clean_profiles().drop_duplicates(['profile'])
#     animes = get_clean_animes()
#     # Preprocess the animes
#     animes = anime_preprocess(animes)

#     # Split the dataset into train and test sets
#     train_profiles, test_profiles = split_profile(profiles, 0.5, 0.5)

#     # Initialize the content-based recommender
#     recommender = ContentBasedRecommender(animes)

#     # Generate recommendations for each user in the test set
#     content_based_recommendations = content_based_recommend(recommender, animes, train_profiles)

#     # Evaluate precision at k
#     precision_results = evaluate_precision_at_k(content_based_recommendations, test_profiles, k=10)
    
#     # Print precision results
#     for user, precision in precision_results.items():
#         print(f"User {user}: Precision at 10 = {precision:.4f}, {test_profiles[test_profiles['profile'] == user]['favorites_count'].values[0]} favorites")
#     # print overall precision
#     overall_precision = sum(precision_results.values()) / len(precision_results) if precision_results else 0.0
#     print(f"Overall Precision at 10: {overall_precision:.4f}")
#     print("Evaluation completed.")
# if __name__ == "__main__":
#     main()



