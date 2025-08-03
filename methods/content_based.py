import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Any, List
from sklearn.neighbors import NearestNeighbors
from preprocessing.load_cleaned import (
    get_clean_animes,
    get_clean_profiles,
    get_clean_reviews,
    get_all_cleaned_data
)
from preprocessing.preprocess_pipline import (
    anime_preprocess, 
    profile_preprocess, 
    review_preprocess,
    final_preprocess
)
from preprocessing.split_dataset import split_profile
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from evaluation.precision_at_k import evaluate_precision_at_k
from evaluation.ndcg import evaluate_ndcg_at_k
from methods.cold_start import cold_start_top_n, recommend_for_cold_start_profiles

class ContentBasedRecommender:
    """
    Content-based recommendation using global TF-IDF + NearestNeighbors for fast Top-k.
    """

    def __init__(
        self,
        animes: pd.DataFrame,
        tfidf_params: Optional[Dict[str, Any]] = None,
        nn_params: Optional[Dict[str, Any]] = None
    ):
        # Default parameters
        tfidf_params = tfidf_params or {'stop_words': 'english', 'max_features': 9800}
        nn_params    = nn_params    or {'n_neighbors': 10, 'metric': 'cosine', 'algorithm': 'auto'}

        # Data preparation
        self.animes = self._prepare(animes)
        self.uids = self.animes['uid'].tolist()
        self.genres = self.animes['genre'].tolist()

        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.animes['synopsis'])

        # NearestNeighbors index
        self.nn_model = NearestNeighbors(**nn_params)
        self.nn_model.fit(self.tfidf_matrix)

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df[['uid', 'synopsis', 'genre']].dropna(subset=['synopsis', 'genre']).copy()
        df['genre'] = df['genre'].apply(lambda g: [s.strip() for s in g] if isinstance(g, (list, tuple)) else [])
        return df.reset_index(drop=True)

    def recommend_similar(
        self,
        target_uid: int,
        top_k: int = 10,
        genre_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Recommend anime most similar to the target_uid based on content.
        If genre_filter is specified, only anime belonging to this genre will be kept in candidates.

        :param target_uid: The anime UID to query for
        :param top_k: Number of recommendations to return
        :param genre_filter: Optional, specify genre, results will only keep this genre
        :return: DataFrame, columns = ['uid','synopsis','genre','similarity']
        """
        if target_uid not in self.uids:
            raise ValueError(f"UID {target_uid} is not in the dataset.")

        idx = self.uids.index(target_uid)
        query_vec = self.tfidf_matrix[idx].toarray()  

        # Search more candidates to handle genre filtering
        k_search = min(len(self.uids), top_k * 3)
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=k_search)

        # Convert distance to similarity
        sims = 1 - distances.flatten()
        idxs = indices.flatten()

        results = []
        for sim, i in zip(sims, idxs):
            if self.uids[i] == target_uid:
                continue
            if genre_filter and genre_filter not in self.genres[i]:
                continue
            results.append({
                'uid': self.uids[i],
                'synopsis': self.animes.at[i, 'synopsis'],
                'genre': self.genres[i],
                'similarity': float(sim)
            })
            if len(results) >= top_k:
                break

        return pd.DataFrame(results)
        

    def recommend_for_user(
        self,
        liked_uids: List[int],
        top_k: int = 10,
        genre_filter: Optional[str] = None
    ) -> List[tuple]:

        # Filter valid UIDs
        valid_uids = [uid for uid in liked_uids if uid in self.uids]
        if not valid_uids:
            raise ValueError("No valid anime UIDs in the list.")
        
        # Calculate user preference vector (average of liked anime vectors)
        idxs = [self.uids.index(uid) for uid in valid_uids]
        user_vec = self.tfidf_matrix[idxs].mean(axis=0).A

        # Search for similar anime
        k_search = min(len(self.uids), top_k * 3)
        distances, indices = self.nn_model.kneighbors(user_vec, n_neighbors=k_search)
        sims = 1 - distances.flatten()
        idxs = indices.flatten()

        # Filter results
        results = []
        for sim, i in zip(sims, idxs):
            uid = self.uids[i]
            if uid in liked_uids:
                continue
            if genre_filter and genre_filter not in self.genres[i]:
                continue
            results.append((uid, float(sim)))
            if len(results) >= top_k:
                break

        return results



def content_based_recommend(Recommender: ContentBasedRecommender, animes: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    """Generate content-based recommendations based on anime genres."""
    recommender = Recommender
    recommendations = defaultdict(list)

    cold_start = recommend_for_cold_start_profiles(profiles, n=10)
    if cold_start:
        for pid, recs in cold_start.items():
            recommendations[pid] = recs
    
    profiles = profiles[profiles['is_cold_start']==False].reset_index(drop=True)
    
    for _, profile in tqdm(profiles.iterrows(), total=len(profiles), desc="Generating recommendations"):
        liked_animes = [int(uid) for uid in profile['favorites_anime']]
        # print(f"Profile {profile['profile']} likes animes: {liked_animes}")
        if not liked_animes:
            continue
        try:
            recs = recommender.recommend_for_user(liked_animes, top_k=10)
        except ValueError as e:
            print(f"Error for profile {profile['profile']}: {e}")
            continue
        recommendations[profile['profile']]= recs

    return recommendations


def main():
    """Run the preprocessing pipeline for animes, profiles, and reviews."""
    
    # Load cleaned data
    animes, profiles, reviews = get_all_cleaned_data()
    # Preprocess each dataset
    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)
    #split profiles into training and testing sets
    train_profiles, test_profiles = split_profile(profiles, train_size=0.8, test_size=0.2)

    print("Preprocessing complete. Animes and profiles are ready for recommendation.")
    
    # prepare different parameters for the TFIDF
    TFIDF_params = []
    for i in range(9800, 100000, 100):
        TFIDF_params.append({'stop_words': 'english', 'max_features': i})
    
    import matplotlib.pyplot as plt
    
    best_precision = 0.0
    best_ndcg = 0.0
    best_params_precision = None
    best_params_ndcg = None
    
    # Store results for plotting
    max_features_list = []
    precision_scores = []
    ndcg_scores = []
    
    for params in tqdm(TFIDF_params, desc="Testing different TFIDF parameters"):
        print(f"Using TFIDF parameters: {params}")
        recommender = ContentBasedRecommender(animes, tfidf_params=params)
        # Get recommendations using content-based filtering
        recommendations = content_based_recommend(recommender, animes, train_profiles)
        
        # Evaluate precision at k
        precision_results = evaluate_precision_at_k(recommendations, test_profiles, k=10)
        
        # Evaluate NDCG at k
        ndcg_results = evaluate_ndcg_at_k(recommendations, test_profiles, k=10)
        
        overall_precision = sum(precision_results.values()) / len(precision_results) if precision_results else 0.0
        overall_ndcg = sum(ndcg_results.values()) / len(ndcg_results) if ndcg_results else 0.0
        print(f"Overall Precision at 10: {overall_precision:.4f}, Overall NDCG at 10: {overall_ndcg:.4f}")
        
        # Store results
        max_features_list.append(params['max_features'])
        precision_scores.append(overall_precision)
        ndcg_scores.append(overall_ndcg)
        
        if overall_precision > best_precision:
            best_precision = overall_precision
            best_params_precision = params
        if overall_ndcg > best_ndcg:
            best_ndcg = overall_ndcg
            best_params_ndcg = params

    print(f"Best Precision: {best_precision:.4f} with params {best_params_precision}")
    print(f"Best NDCG: {best_ndcg:.4f} with params {best_params_ndcg}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(max_features_list, precision_scores, 'b-', marker='o', markersize=2)
    plt.xlabel('Max Features')
    plt.ylabel('Precision@10')
    plt.title('Precision@10 vs Max Features')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=best_params_precision['max_features'], color='r', linestyle='--', 
                label=f'Best: {best_params_precision["max_features"]}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(max_features_list, ndcg_scores, 'g-', marker='o', markersize=2)
    plt.xlabel('Max Features')
    plt.ylabel('NDCG@10')
    plt.title('NDCG@10 vs Max Features')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=best_params_ndcg['max_features'], color='r', linestyle='--',
                label=f'Best: {best_params_ndcg["max_features"]}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tfidf_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'max_features': max_features_list,
        'precision_at_10': precision_scores,
        'ndcg_at_10': ndcg_scores
    })
    results_df.to_csv('tfidf_parameter_results.csv', index=False)
    print("Results saved to 'tfidf_parameter_results.csv'")

if __name__ == "__main__":
    main()