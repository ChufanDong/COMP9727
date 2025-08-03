import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.load_cleaned import get_all_cleaned_data
from preprocessing.preprocess_pipline import final_preprocess, anime_preprocess, profile_preprocess, review_preprocess
from preprocessing.split_dataset import split_profile
from methods.content_based import ContentBasedRecommender, content_based_recommend
from evaluation.precision_at_k import evaluate_precision_at_k
from evaluation.ndcg import evaluate_ndcg_at_k
from evaluation.recall_at_k import evaluate_recall_at_k

def main():
    animes, profiles, reviews = get_all_cleaned_data()
    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)
    
    # Split profiles into training and testing sets
    train_profiles, test_profiles = split_profile(profiles, train_size=0.5, test_size=0.5)
    
    # initialize content-based recommender
    content_recommender = ContentBasedRecommender(animes)
    # Get recommendations using content-based filtering
    recommendations = content_based_recommend(content_recommender, animes, train_profiles)
    
    # Evaluate precision at k
    precision_results = evaluate_precision_at_k(recommendations, test_profiles, k=10)
    
    # Evaluate NDCG at k
    ndcg_results = evaluate_ndcg_at_k(recommendations, test_profiles, k=10)

    # Evaluate recall at k
    recall_results = evaluate_recall_at_k(recommendations, test_profiles, k=10)
    
    for user, precision in precision_results.items():
        print(f"User: {user}, Precision@10: {precision:.4f}")
    for user, ndcg in ndcg_results.items():
        print(f"User: {user}, NDCG@10: {ndcg:.4f}")
    for user, recall in recall_results.items():
        print(f"User: {user}, Recall@10: {recall:.4f}")
    
    print("Evaluation complete.")
    print("Overall precision:", sum(precision_results.values()) / len(precision_results))
    print("Overall NDCG:", sum(ndcg_results.values()) / len(ndcg_results))
    print("Overall recall:", sum(recall_results.values()) / len(recall_results))

if __name__ == "__main__":
    main()