import sys
import os
from typing import Dict, List, Tuple
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.precision_at_k import evaluate_precision_at_k
from evaluation.ndcg import evaluate_ndcg_at_k
from evaluation.recall_at_k import evaluate_recall_at_k

def evaluation_pipeline(recommendations: Dict[str, List[Tuple[int, float]]], test_profiles: pd.DataFrame, k: int = 10) -> Dict:
    """
    Evaluate the recommendations using precision, NDCG, and recall.
    
    :param recommendations: Dictionary of user recommendations.
    :param test_profiles: DataFrame of test profiles.
    :param k: Number of top recommendations to consider.
    :return: Dictionary with evaluation metrics.
    """
    precision_results = evaluate_precision_at_k(recommendations, test_profiles, k)
    overall_precision = sum(precision_results.values()) / len(precision_results)
    ndcg_results = evaluate_ndcg_at_k(recommendations, test_profiles, k)
    overall_ndcg = sum(ndcg_results.values()) / len(ndcg_results)
    recall_results = evaluate_recall_at_k(recommendations, test_profiles, k)
    overall_recall = sum(recall_results.values()) / len(recall_results)
    
    overall = {
        "precision": overall_precision,
        "ndcg": overall_ndcg,
        "recall": overall_recall
    }
    results = {
        "precision": precision_results,
        "ndcg": ndcg_results,
        "recall": recall_results
    }

    return {"overall": overall, "results": results}