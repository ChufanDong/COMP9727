import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.load_cleaned import get_all_cleaned_data
from preprocessing.preprocess_pipline import final_preprocess, anime_preprocess, profile_preprocess, review_preprocess
from preprocessing.split_dataset import split_profile
from methods.CF import read_json_recommendations
from evaluation.precision_at_k import evaluate_precision_at_k
from evaluation.ndcg import evaluate_ndcg_at_k


def main():
    animes, profiles, reviews = get_all_cleaned_data()
    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)
    train_profiles, test_profiles = split_profile(profiles, train_size=0.8, test_size=0.2)
    # get recommendations
    recommendations = read_json_recommendations(r"methods\svd_recommendations.json")
    # evaluate precision at k
    precision_results = evaluate_precision_at_k(recommendations, test_profiles, k=10)
    # evaluate ndcg at k
    ndcg_results = evaluate_ndcg_at_k(recommendations, test_profiles, k=10)
    
    for user, precision in precision_results.items():
        print(f"User: {user}, Precision@10: {precision:.4f}")
    for user, ndcg in ndcg_results.items():
        print(f"User: {user}, NDCG@10: {ndcg:.4f}")
    print("Evaluation complete.")
    print("overall precision:", sum(precision_results.values()) / len(precision_results))
    print("overall ndcg:", sum(ndcg_results.values()) / len(ndcg_results))

if __name__ == "__main__":
    main()
    