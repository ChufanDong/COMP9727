import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from methods.cold_start import recommend_for_cold_start_profiles

class ContentBasedRecommender:
    """
    A simplified and more robust Content-Based Recommender that uses a single,
    global model instead of per-genre models. This approach avoids data fragmentation
    and builds a stronger, more generalized model.
    """
    def __init__(
        self,
        animes: pd.DataFrame,
        tfidf_params: Optional[Dict[str, Any]] = None,
        nn_params: Optional[Dict[str, Any]] = None,
        text_weight: float = 0.5,
        numerical_weight: float = 0.5
    ):
        # 1. Use more robust default parameters
        self.tfidf_params = tfidf_params or {
            'stop_words': 'english',
            'max_features': 10000,
            'min_df': 2,
            'max_df': 0.7,
            'ngram_range': (1, 1),
            'sublinear_tf': True,
            'norm': 'l2'
        }
        self.nn_params = nn_params or {
            'n_neighbors': 50,      # Increase neighbors to get a larger candidate pool
            'metric': 'cosine',
            'algorithm': 'brute',  
            'n_jobs': -1
        }

        self.text_weight = text_weight
        self.numerical_weight = numerical_weight

        # 2. Prepare and store data
        self.animes = self._prepare_data(animes)
        self.uid_to_idx = {uid: i for i, uid in enumerate(self.animes['uid'])}

        # 3. Build a single, unified feature matrix
        self.feature_matrix = self._build_feature_matrix()

        # 4. Fit a single NearestNeighbors model
        self.nn_model = NearestNeighbors(**self.nn_params)
        self.nn_model.fit(self.feature_matrix)
        print("Successfully built a unified Content-Based Recommender.")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the anime dataframe for recommendation."""
        df = df.copy()
        # Ensure 'genre' is a list of strings
        df['genre_str'] = df['genre'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        # Combine text features for a richer representation
        df['combined_text'] = df['synopsis'].fillna('') + ' ' + df['genre_str']
        return df.reset_index(drop=True)

    def _build_feature_matrix(self):
        """Enhanced feature matrix with better weighting."""
        # TF-IDF for combined text
        vectorizer = TfidfVectorizer(**self.tfidf_params)
        text_features = vectorizer.fit_transform(self.animes['combined_text'])
        
        # Enhanced numerical features
        scaler = MinMaxScaler()
        
        # Use more numerical features
        numerical_cols = []
        if 'score' in self.animes.columns:
            numerical_cols.append('score')
        if 'popularity' in self.animes.columns:
            numerical_cols.append('popularity')
        if 'members' in self.animes.columns:
            numerical_cols.append('members')
        
        if numerical_cols:
            numerical_features = self.animes[numerical_cols].fillna(0)
            scaled_numerical = scaler.fit_transform(numerical_features)
            
            return hstack([text_features * self.text_weight, scaled_numerical * self.numerical_weight]).tocsr()
        else:
            return text_features

    def recommend(self, favorites_anime: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Recommends animes by aggregating similarities from each favorite item.
        This approach preserves diverse user interests.
        """
        if not favorites_anime:
            return []

        # Use a dictionary to aggregate scores for potential recommendations
        aggregated_scores = defaultdict(float)
        # Use a set for fast lookups of animes the user already likes
        watched_uids = set(favorites_anime)

        # Find valid favorite animes that exist in our model
        valid_favorites_idx = [self.uid_to_idx[uid] for uid in favorites_anime if uid in self.uid_to_idx]

        if not valid_favorites_idx:
            return []

        # Get the feature vectors for all favorite animes at once
        favorite_vectors = self.feature_matrix[valid_favorites_idx]

        # Find neighbors for all favorite animes in a single batch call
        distances, indices = self.nn_model.kneighbors(favorite_vectors)

        # Aggregate the results
        for i in range(len(valid_favorites_idx)):
            # The first neighbor is the item itself, so we start from the second (j=1)
            for j in range(1, len(indices[i])):
                neighbor_idx = indices[i][j]
                neighbor_uid = self.animes.iloc[neighbor_idx]['uid']

                # Only recommend animes the user hasn't already favorited
                if neighbor_uid not in watched_uids:
                    similarity = 1 - distances[i][j]
                    # Add similarity to the anime's score
                    aggregated_scores[neighbor_uid] += similarity

        if not aggregated_scores:
            return []

        # Sort the aggregated recommendations by score
        sorted_recs = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)

        return [(uid, float(score)) for uid, score in sorted_recs[:top_k]]

def content_based_recommend(Recommender: ContentBasedRecommender, animes: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    """Generate content-based recommendations - optimized for speed."""
    recommender = Recommender
    recommendations = defaultdict(list)

    # Handle cold start profiles efficiently
    cold_start_profiles = profiles[profiles['is_cold_start'] == True]
    if not cold_start_profiles.empty:
        cold_start = recommend_for_cold_start_profiles(cold_start_profiles, n=10)
        if cold_start:
            recommendations.update(cold_start)
    
    # Process non-cold start profiles
    active_profiles = profiles[profiles['is_cold_start'] == False]
    
    # Batch process profiles for better performance
    for _, profile in tqdm(active_profiles.iterrows(), total=len(active_profiles), 
                          desc="Generating Content_based recommendations", 
                          mininterval=1.0):
        
        liked_animes = profile['favorites_anime']
        if not liked_animes:
            continue
            
        # Convert to int list efficiently
        try:
            liked_animes = [int(uid) for uid in liked_animes]
            recs = recommender.recommend(liked_animes, top_k=10)
            if recs:  # Only add if recommendations were generated
                recommendations[profile['profile']] = recs
        except (ValueError, Exception) as e:
            print(f"Error processing profile {profile['profile']}: {e}")
            continue

    return recommendations