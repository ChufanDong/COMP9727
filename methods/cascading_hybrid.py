import os
import sys
from typing import List, Dict, Tuple, Union
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from methods.CF import auto_recommend_dump, read_json_recommendations
from methods.collaborative_knn import CollaborativeFilteringRecommender, collaborative_recommend
from methods.content_based import ContentBasedRecommender, content_based_recommend
from methods.cold_start import recommend_for_cold_start_profiles
import pandas as pd
import numpy as np
from preprocessing.load_cleaned import get_all_cleaned_data
from preprocessing.preprocess_pipline import final_preprocess
from preprocessing.split_dataset import split_profile
from evaluation.precision_at_k import evaluate_precision_at_k
from evaluation.ndcg import evaluate_ndcg_at_k
from evaluation.recall_at_k import evaluate_recall_at_k

import matplotlib.pyplot as plt
from tqdm import tqdm

def get_collaborative_recommendations(profiles: pd.DataFrame, top_k: int = 10) -> Dict[str, List[Tuple[int, float]]]:
    """
    Generate recommendations using collaborative filtering.
    """
    recommendations = collaborative_recommend(profiles, top_k)
    return recommendations

def get_content_based_recommendations(animes: pd.DataFrame, profiles: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
    """
    Generate recommendations using content-based filtering.
    """
    content_recommender = ContentBasedRecommender(animes)
    recommendations = content_based_recommend(content_recommender, animes, profiles)
    return recommendations

def normalize_scores(recommendations: Dict[str, List[Tuple[int, float]]]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Normalize scores in the recommendations to a 0-1 range.
    """
    normalized_recs = {}
    for user, recs in recommendations.items():
        if not recs:
            normalized_recs[user] = []
            continue
        
        scores = [score for _, score in recs]
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            normalized_recs[user] = [(anime, 1.0) for anime, _ in recs]
        else:
            normalized_recs[user] = [(anime, (score - min_score) / (max_score - min_score)) for anime, score in recs]
    
    return normalized_recs

def cascading_hybrid_recommendations(
    animes: pd.DataFrame,
    profiles: pd.DataFrame,
    reviews: pd.DataFrame,
    top_k: int = 10,
    cf_top_m: int = 30,
    lambda_content: float = 0.3
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Cascading hybrid:
    1. Get top-M from collaborative filtering.
    2. Re-rank those candidates by blending in content similarity.
    """
    # Cold start fallback
    cold_start_recs = recommend_for_cold_start_profiles(profiles, top_k)

    # Step 1: collaborative filtering candidates (top M)
    collab_recs_full = get_collaborative_recommendations(profiles, top_k=cf_top_m)
    collab_recs_norm = normalize_scores(collab_recs_full)

    # Step 2: get content-based scores for all users (full), normalized
    content_recommender = ContentBasedRecommender(animes)
    content_recs_full = content_based_recommend(content_recommender, animes, profiles)
    content_recs_norm = normalize_scores(content_recs_full)

    # Step 3: cascade & blend over CF candidates
    hybrid_recs = defaultdict(list)
    for user in profiles['profile']:
        if user in cold_start_recs:
            hybrid_recs[user] = cold_start_recs[user]
            continue

        cf_items = dict(collab_recs_norm.get(user, []))  # item -> cf_score
        content_items = dict(content_recs_norm.get(user, []))  # item -> content_score

        combined = {}
        for item, cf_score in cf_items.items():
            content_score = content_items.get(item, 0.0)
            blended_score = (1 - lambda_content) * cf_score + lambda_content * content_score
            combined[item] = blended_score

        # Sort and take top_k
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        hybrid_recs[user] = sorted_items

    return hybrid_recs

def main():
    # Load cleaned data
    animes, profiles, reviews = get_all_cleaned_data()
    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)

    # Split profiles into training and testing sets
    train_profiles, test_profiles = split_profile(profiles, train_size=0.5, test_size=0.5)

    # Parameters for cascading hybrid
    top_k = 10
    cf_top_m = 30  # number of CF candidates to cascade
    lambda_content = 0.2  # strength of content re-ranking

    # Generate cascading hybrid recommendations
    hybrid_recs = cascading_hybrid_recommendations(
        animes, train_profiles, reviews,
        top_k=top_k,
        cf_top_m=cf_top_m,
        lambda_content=lambda_content
    )

    # Evaluate precision at k
    precision_results = evaluate_precision_at_k(hybrid_recs, test_profiles, k=top_k)
    
    # Evaluate NDCG at k
    ndcg_results = evaluate_ndcg_at_k(hybrid_recs, test_profiles, k=top_k)
    
    overall_precision = sum(precision_results.values()) / len(precision_results) if precision_results else 0.0
    overall_ndcg = sum(ndcg_results.values()) / len(ndcg_results) if ndcg_results else 0.0
    print(f"Cascading Hybrid Overall Precision@{top_k}: {overall_precision:.4f}")
    print(f"Cascading Hybrid Overall NDCG@{top_k}: {overall_ndcg:.4f}")

    # (Optional) plot results for this single configuration
    plt.figure(figsize=(8, 4))
    metrics = ['Precision', 'NDCG']
    values = [overall_precision, overall_ndcg]
    plt.bar(metrics, values, alpha=0.7)
    plt.ylim(0, 1)
    plt.title('Cascading Hybrid Performance')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig('cascading_hybrid_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame({
        'metric': ['precision_at_10', 'ndcg_at_10'],
        'value': [overall_precision, overall_ndcg]
    })
    results_df.to_csv('cascading_hybrid_results.csv', index=False)
    print("Results saved to 'cascading_hybrid_results.csv'")

if __name__ == "__main__":
    main()