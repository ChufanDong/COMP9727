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
)
from preprocessing.preprocess_pipline import (
    anime_preprocess, 
    profile_preprocess, 
    review_preprocess,
    final_preprocess
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

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
        tfidf_params = tfidf_params or {'stop_words': 'english', 'max_features': 5000}
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
    profiles = profiles[profiles['is_cold_start'] == False].reset_index(drop=True).sample(frac=1).reset_index(drop=True)
    
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
    animes = get_clean_animes()
    profiles = get_clean_profiles()
    # reviews = get_clean_reviews()

    # Preprocess each dataset
    animes = anime_preprocess(animes)
    profiles = profile_preprocess(profiles)
    # reviews = review_preprocess(reviews)
    # animes = pd.read_csv(r"E:\UNSW\COMP9727\Proj\animeRecommender\animeRecommender\data\processed_animes.csv")


    print("Preprocessing complete. Animes and profiles are ready for recommendation.")
    # debugging
    recommends = content_based_recommend(animes, profiles)
    for profile, recs in recommends.items():
        print(f"Recommendations for {profile}:")
        for uid, sim in recs:
            print(f"  - UID: {uid}, Similarity: {sim:.4f}")
    # Optionally save or return processed data
    return animes, profiles

if __name__ == "__main__":
    main()