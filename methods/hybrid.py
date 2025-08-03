import os
import sys
from typing import List, Dict, Tuple, Union
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods.CF import auto_recommend_dump, read_json_recommendations
from methods.collaborative_basic import CollaborativeFilteringRecommender, collaborative_recommend
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

def get_collaborative_recommendations(profiles: pd.DataFrame, top_k: int = 10) -> Dict[str, List[Tuple[int, float]]]:
    """
    Generate recommendations using collaborative filtering.
    """
    recommendations = collaborative_recommend(profiles, top_k)
    # recommendations = auto_recommend_dump(top_k)
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

def main():
    # Load cleaned data
    animes, profiles, reviews = get_all_cleaned_data()
    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)

    # Split profiles into training and testing sets
    train_profiles, test_profiles = split_profile(profiles, train_size=0.5, test_size=0.5)

    # Create parameter grid for a and b weights
    weight_params = []
    for a in np.arange(0.0, 1.1, 0.1):  # From 0.0 to 1.0 with step 0.1
        for b in np.arange(0.0, 1.1, 0.1):
            if abs(a + b - 1.0) < 0.01:  # Ensure a + b = 1.0 (approximately)
                weight_params.append({'a': round(a, 1), 'b': round(b, 1)})
    
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    best_precision = 0.0
    best_ndcg = 0.0
    best_params_precision = None
    best_params_ndcg = None
    
    # Store results for plotting
    a_values = []
    b_values = []
    precision_scores = []
    ndcg_scores = []
    
    for params in tqdm(weight_params, desc="Testing different weight combinations"):
        print(f"Using weights: a={params['a']}, b={params['b']}")
        
        # Generate hybrid recommendations with current weights
        hybrid_recs = hybrid_recommendations(
            animes, train_profiles, reviews, 
            a=params['a'], b=params['b'], top_k=10
        )

        # Evaluate precision at k
        precision_results = evaluate_precision_at_k(hybrid_recs, test_profiles, k=10)
        
        # Evaluate NDCG at k
        ndcg_results = evaluate_ndcg_at_k(hybrid_recs, test_profiles, k=10)
        
        overall_precision = sum(precision_results.values()) / len(precision_results) if precision_results else 0.0
        overall_ndcg = sum(ndcg_results.values()) / len(ndcg_results) if ndcg_results else 0.0
        print(f"Overall Precision at 10: {overall_precision:.4f}, Overall NDCG at 10: {overall_ndcg:.4f}")
        
        # Store results
        a_values.append(params['a'])
        b_values.append(params['b'])
        precision_scores.append(overall_precision)
        ndcg_scores.append(overall_ndcg)
        
        if overall_precision > best_precision:
            best_precision = overall_precision
            best_params_precision = params
        if overall_ndcg > best_ndcg:
            best_ndcg = overall_ndcg
            best_params_ndcg = params

    print(f"Best Precision: {best_precision:.4f} with weights {best_params_precision}")
    print(f"Best NDCG: {best_ndcg:.4f} with weights {best_params_ndcg}")
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Precision scores
    plt.subplot(1, 3, 1)
    scatter1 = plt.scatter(a_values, precision_scores, c=b_values, cmap='viridis', alpha=0.7)
    plt.xlabel('Weight a (Collaborative)')
    plt.ylabel('Precision@10')
    plt.title('Precision@10 vs Weight a')
    plt.colorbar(scatter1, label='Weight b (Content-based)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: NDCG scores
    plt.subplot(1, 3, 2)
    scatter2 = plt.scatter(a_values, ndcg_scores, c=b_values, cmap='viridis', alpha=0.7)
    plt.xlabel('Weight a (Collaborative)')
    plt.ylabel('NDCG@10')
    plt.title('NDCG@10 vs Weight a')
    plt.colorbar(scatter2, label='Weight b (Content-based)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of precision scores
    plt.subplot(1, 3, 3)
    # Reshape data for heatmap
    unique_a = sorted(list(set(a_values)))
    unique_b = sorted(list(set(b_values)))
    precision_matrix = np.zeros((len(unique_b), len(unique_a)))
    
    for i, a in enumerate(a_values):
        a_idx = unique_a.index(a)
        b_idx = unique_b.index(b_values[i])
        precision_matrix[b_idx, a_idx] = precision_scores[i]
    
    im = plt.imshow(precision_matrix, cmap='viridis', aspect='auto')
    plt.xlabel('Weight a (Collaborative)')
    plt.ylabel('Weight b (Content-based)')
    plt.title('Precision@10 Heatmap')
    plt.xticks(range(len(unique_a)), [f'{a:.1f}' for a in unique_a])
    plt.yticks(range(len(unique_b)), [f'{b:.1f}' for b in unique_b])
    plt.colorbar(im)
    
    plt.tight_layout()
    plt.savefig('hybrid2_weight_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'weight_a': a_values,
        'weight_b': b_values,
        'precision_at_10': precision_scores,
        'ndcg_at_10': ndcg_scores
    })
    results_df.to_csv('hybrid2_weight_parameter_results.csv', index=False)
    print("Results saved to 'hybrid2_weight_parameter_results.csv'")

if __name__ == "__main__":
    main()
