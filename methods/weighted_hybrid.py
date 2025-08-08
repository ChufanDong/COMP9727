import os
import sys
from typing import List, Dict, Tuple
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods.collaborative_knn import collaborative_recommend
from methods.content_based import ContentBasedRecommender, content_based_recommend
from methods.cold_start import recommend_for_cold_start_profiles
import pandas as pd

def get_collaborative_recommendations(profiles: pd.DataFrame, top_k: int = 10) -> Dict[str, List[Tuple[int, float]]]:
    """
    Generate recommendations using collaborative filtering.
    """
    recommendations = collaborative_recommend(profiles, top_k)
    # recommendations = SVD_recommend(train_profiles=train_profiles, test_profiles=test_profiles, reviews=reviews, top_k=top_k)
    # recommendations = read_json_recommendations(r"methods\svd_recommendations.json")
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

def hybrid_recommendations(
    animes: pd.DataFrame, 
    profiles: pd.DataFrame, 
    reviews: pd.DataFrame,
    a: float = 0.5,
    b: float = 0.5,
    top_k: int = 10
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Generate hybrid recommendations combining collaborative and content-based methods.
    """
    n = top_k

    # Get cold start recommendations
    cold_start_recs = recommend_for_cold_start_profiles(profiles, n)

    # Get collaborative recommendations
    collab_recs = get_collaborative_recommendations(profiles, n)
    
    # Get content-based recommendations
    content_recs = get_content_based_recommendations(animes, profiles)

    # Normalize scores
    collab_recs = normalize_scores(collab_recs)
    content_recs = normalize_scores(content_recs)
    
    alpha = a  # Weight for collaborative recommendations
    beta = b   # Weight for content-based recommendations
    # Combine recommendations
    hybrid_recs = defaultdict(list)
    for user in profiles['profile']:
        if user in cold_start_recs:
            hybrid_recs[user] = cold_start_recs[user]
        else:
            collab_items = collab_recs.get(user, [])
            content_items = content_recs.get(user, [])
            
            # Combine and weight recommendations
            combined_items = defaultdict(float)
            for item, score in collab_items:
                combined_items[item] += alpha * score
            for item, score in content_items:
                combined_items[item] += beta * score
            
            # Sort by score and take top K
            sorted_items = sorted(combined_items.items(), key=lambda x: x[1], reverse=True)[:top_k]
            hybrid_recs[user] = sorted_items
    return hybrid_recs
