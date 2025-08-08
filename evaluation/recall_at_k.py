from typing import List, Dict, Tuple, Union
import pandas as pd
import ast

def recall_at_n(predictions: List[int], ground_truth: List[int], n: int) -> float:
    """
    Calculate recall at n for a set of predictions.
    :param predictions: List of predicted items.
    :param ground_truth: List of true items.
    :param n: The number of top items to consider.
    :return: Recall at n.
    """
    if not ground_truth or n <= 0:
        return 0.0

    pred_at_n = predictions[:n]
    if not pred_at_n:
        return 0.0

    ground_truth_set = set(ground_truth)
    correct = sum(1 for item in pred_at_n if item in ground_truth_set)
    return correct / len(ground_truth_set)


def batch_recall_at_n(predictions: Dict[str, List[Tuple[int, float]]],
                      ground_truth: Dict[str, List[int]],
                      n: int) -> Dict[str, float]:
    """
    Calculate recall at n for multiple users.
    """
    recalls = {}
    for user, preds in predictions.items():
        true_items = ground_truth.get(user, [])
        pred_items = [item for item, _ in preds]
        recalls[user] = recall_at_n(pred_items, true_items, n)
    return recalls


def evaluate_recall_at_k(predictions: Dict[str, List[Tuple[int, float]]],
                         ground_truth: pd.DataFrame,
                         k: int) -> Dict[str, float]:
    """
    Evaluate recall at k for all users.
    """
    def parse_anime_ids(anime_ids_str: Union[str, list]) -> List[int]:
        if isinstance(anime_ids_str, str):
            try:
                parsed = ast.literal_eval(anime_ids_str)
                return [int(x) for x in parsed] if isinstance(parsed, list) else [int(parsed)]
            except (ValueError, SyntaxError):
                return [int(x.strip()) for x in anime_ids_str.split(',') if x.strip()]
        elif isinstance(anime_ids_str, list):
            return [int(x) for x in anime_ids_str]
        else:
            return [int(anime_ids_str)]

    ground_truth_dict = {}
    for _, row in ground_truth.iterrows():
        user_id = str(row['profile'])
        anime_ids = parse_anime_ids(row['favorites_anime'])
        ground_truth_dict[user_id] = anime_ids

    return batch_recall_at_n(predictions, ground_truth_dict, k)