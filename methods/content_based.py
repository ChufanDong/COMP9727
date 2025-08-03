import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
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
from evaluation.evaluation_pipline import evaluation_pipeline
from methods.cold_start import cold_start_top_n, recommend_for_cold_start_profiles
from sklearn.metrics.pairwise import cosine_similarity
import ast
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
            'max_features': 20000,    # 增加特征数
            'min_df': 2,              # 过滤低频词
            'max_df': 0.7,            # 过滤高频词  
            'ngram_range': (1, 2),
            'sublinear_tf': True,     # 使用次线性TF
            'norm': 'l2'              # L2归一化
        }
        self.nn_params = nn_params or {
            'n_neighbors': 50,      # Increase neighbors to get a larger candidate pool
            'metric': 'cosine',
            'algorithm': 'brute',   # 'brute' is often best for sparse data
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
            
            # Better weighting: 60% text, 40% numerical
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

def test_content_based_recommend():
    """
    Enhanced test function for content-based recommender with comprehensive 
    performance analysis and result persistence.
    """
    print("=" * 60)
    print("CONTENT-BASED RECOMMENDER PERFORMANCE TEST")
    print("=" * 60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    animes, profiles, reviews = get_all_cleaned_data()
    animes, profiles, reviews = final_preprocess(animes, profiles, reviews)
    train_profiles, test_profiles = split_profile(profiles, train_size=0.5, test_size=0.5)
    
    # Filter profiles
    cold_start_profiles = train_profiles[train_profiles['is_cold_start'] == True]
    normal_profiles = train_profiles[train_profiles['is_cold_start'] == False]
    
    print(f"Dataset Statistics:")
    print(f"  - Total animes: {len(animes)}")
    print(f"  - Cold start profiles: {len(cold_start_profiles)}")
    print(f"  - Normal profiles: {len(normal_profiles)}")
    print(f"  - Test profiles: {len(test_profiles)}")
    print()

    # Define weight configurations to test
    weight_configs = [
        # Fine-grained testing around expected optimal range
        (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2),
        # Extreme cases
        (0.9, 0.1), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9),
        # Balanced approaches
        (0.55, 0.45), (0.65, 0.35), (0.75, 0.25)
    ]
    
    results = []
    best_precision = 0
    best_ndcg = 0
    best_recall = 0
    best_precision_config = None
    best_ndcg_config = None
    best_recall_config = None
    
    print(f"Testing {len(weight_configs)} weight configurations...")
    print("-" * 60)

    for i, (text_weight, numerical_weight) in enumerate(weight_configs, 1):
        print(f"[{i}/{len(weight_configs)}] Testing weights: text={text_weight:.2f}, numerical={numerical_weight:.2f}")
        
        try:
            # Create recommender with current weights
            recommender = ContentBasedRecommender(
                animes, 
                text_weight=text_weight,
                numerical_weight=numerical_weight
            )
            
            # Generate recommendations
            print("  Generating recommendations...")
            content_based_results = content_based_recommend(recommender, animes, normal_profiles)
            
            # Evaluate performance
            print("  Evaluating performance...")
            eva_R = evaluation_pipeline(
                content_based_results, 
                test_profiles, 
                k=10, 
            )

            precision = eva_R['overall']['precision']
            ndcg = eva_R['overall']['ndcg']
            recall = eva_R['overall']['recall']
            
            result = {
                'text_weight': text_weight,
                'numerical_weight': numerical_weight,
                'precision@10': precision,
                'ndcg@10': ndcg,
                'recall@10': recall,
                'num_users_with_recs': len(content_based_results),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            

            print(f"  ✓ Precision@10: {precision:.4f}")
            print(f"  ✓ NDCG@10: {ndcg:.4f}")
            print(f"  ✓ Recall@10: {recall:.4f}")
            print(f"  ✓ Users with recommendations: {len(content_based_results)}")
            
            # Track best performance
            if precision > best_precision:
                best_precision = precision
                best_precision_config = (text_weight, numerical_weight)
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_ndcg_config = (text_weight, numerical_weight)
                
            if recall > best_recall:
                best_recall = recall
                best_recall_config = (text_weight, numerical_weight)
                
        except Exception as e:
            print(f"  ✗ Error with weights ({text_weight}, {numerical_weight}): {e}")
            continue
        
        print()

    # Save results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to JSON
    json_filename = f"content_based_test_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'test_info': {
                'total_animes': len(animes),
                'normal_profiles': len(normal_profiles),
                'test_profiles': len(test_profiles),
                'test_timestamp': datetime.now().isoformat()
            },
            'results': results,
            'best_performance': {
                'precision': {'value': best_precision, 'config': best_precision_config},
                'ndcg': {'value': best_ndcg, 'config': best_ndcg_config},
                'recall': {'value': best_recall, 'config': best_recall_config}
            }
        }, f, indent=2)
    
    # Save to CSV for easy analysis
    results_df = pd.DataFrame(results)
    csv_filename = f"content_based_test_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    # Print detailed results table
    print("\nDetailed Results:")
    print("Text | Num  | Precision@10 | NDCG@10   | Recall@10 | Users")
    print("-" * 58)
    for result in sorted(results, key=lambda x: x['precision@10'], reverse=True):
        print(f"{result['text_weight']:.2f} | {result['numerical_weight']:.2f} | "
              f"{result['precision@10']:.6f}   | {result['ndcg@10']:.6f}  | "
              f"{result['recall@10']:.6f}    | {result['num_users_with_recs']:5d}")
    
    # Best performance summary
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"  Best Precision@10: {best_precision:.6f} with weights {best_precision_config}")
    print(f"  Best NDCG@10: {best_ndcg:.6f} with weights {best_ndcg_config}")
    print(f"  Best Recall@10: {best_recall:.6f} with weights {best_recall_config}")
    
    # Statistical analysis
    if results:
        precisions = [r['precision@10'] for r in results]
        ndcgs = [r['ndcg@10'] for r in results]
        recalls = [r['recall@10'] for r in results]
        
        print(f"\n📊 STATISTICAL ANALYSIS:")
        print(f"  Precision@10 - Mean: {np.mean(precisions):.6f}, Std: {np.std(precisions):.6f}")
        print(f"  NDCG@10 - Mean: {np.mean(ndcgs):.6f}, Std: {np.std(ndcgs):.6f}")
        print(f"  Recall@10 - Mean: {np.mean(recalls):.6f}, Std: {np.std(recalls):.6f}")
        print(f"  Best precision improvement: {((best_precision - np.min(precisions)) / np.min(precisions) * 100):.2f}%")
        print(f"  Best NDCG improvement: {((best_ndcg - np.min(ndcgs)) / np.min(ndcgs) * 100):.2f}%")
        print(f"  Best recall improvement: {((best_recall - np.min(recalls)) / np.min(recalls) * 100):.2f}%")
    
    print(f"\n💾 Results saved to:")
    print(f"  - {json_filename}")
    print(f"  - {csv_filename}")
    
    # Generate visualization if matplotlib is available
    try:
        _create_performance_plots(results, timestamp)
        print(f"  - content_based_performance_plots_{timestamp}.png")
    except ImportError:
        print("  Note: Install matplotlib and seaborn for visualization")
    except Exception as e:
        print(f"  Note: Could not generate plots: {e}")
    
    return results, best_precision_config, best_ndcg_config, best_recall_config

def _create_performance_plots(results, timestamp):
    """Create comprehensive performance visualization plots including recall."""
    if not results:
        return
        
    # Create a 2x3 subplot layout for more comprehensive visualization
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Plot 1: Precision vs Text Weight
    ax1.scatter(df['text_weight'], df['precision@10'], alpha=0.7, s=60, color='blue')
    ax1.set_xlabel('Text Weight')
    ax1.set_ylabel('Precision@10')
    ax1.set_title('Precision@10 vs Text Weight')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: NDCG vs Text Weight  
    ax2.scatter(df['text_weight'], df['ndcg@10'], alpha=0.7, s=60, color='orange')
    ax2.set_xlabel('Text Weight')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('NDCG@10 vs Text Weight')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Recall vs Text Weight
    ax3.scatter(df['text_weight'], df['recall@10'], alpha=0.7, s=60, color='green')
    ax3.set_xlabel('Text Weight')
    ax3.set_ylabel('Recall@10')
    ax3.setTitle('Recall@10 vs Text Weight')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision Heatmap
    precision_pivot = df.pivot_table(values='precision@10', 
                                   index='numerical_weight', 
                                   columns='text_weight', 
                                   fill_value=0)
    sns.heatmap(precision_pivot, annot=True, fmt='.4f', cmap='Blues', ax=ax4)
    ax4.set_title('Precision@10 Heatmap')
    
    # Plot 5: NDCG Heatmap
    ndcg_pivot = df.pivot_table(values='ndcg@10', 
                               index='numerical_weight', 
                               columns='text_weight', 
                               fill_value=0)
    sns.heatmap(ndcg_pivot, annot=True, fmt='.4f', cmap='Oranges', ax=ax5)
    ax5.set_title('NDCG@10 Heatmap')
    
    # Plot 6: 3D-style scatter plot showing all three metrics
    scatter = ax6.scatter(df['precision@10'], df['ndcg@10'], 
                         c=df['recall@10'], cmap='viridis', s=100, alpha=0.7, 
                         edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Precision@10')
    ax6.set_ylabel('NDCG@10') 
    ax6.set_title('Performance Landscape (color = Recall@10)')
    cbar = plt.colorbar(scatter, ax=ax6, label='Recall@10')
    ax6.grid(True, alpha=0.3)
    
    # Add text annotations for best performing points
    best_precision_idx = df['precision@10'].idxmax()
    best_ndcg_idx = df['ndcg@10'].idxmax()
    best_recall_idx = df['recall@10'].idxmax()
    
    # Annotate best precision point
    ax6.annotate(f'Best Precision\n({df.loc[best_precision_idx, "text_weight"]:.2f}, {df.loc[best_precision_idx, "numerical_weight"]:.2f})',
                xy=(df.loc[best_precision_idx, 'precision@10'], df.loc[best_precision_idx, 'ndcg@10']),
                xytext=(10, 10), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plot_filename = f"content_based_performance_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create an additional summary plot
    _create_summary_plot(df, timestamp)

def _create_summary_plot(df, timestamp):
    """Create a summary plot showing metric trends."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All metrics vs text weight
    ax1.plot(df['text_weight'], df['precision@10'], 'o-', label='Precision@10', alpha=0.7)
    ax1.plot(df['text_weight'], df['ndcg@10'], 's-', label='NDCG@10', alpha=0.7)
    ax1.plot(df['text_weight'], df['recall@10'], '^-', label='Recall@10', alpha=0.7)
    ax1.set_xlabel('Text Weight')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('All Metrics vs Text Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation matrix heatmap
    corr_matrix = df[['text_weight', 'numerical_weight', 'precision@10', 'ndcg@10', 'recall@10']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax2)
    ax2.set_title('Metrics Correlation Matrix')
    
    plt.tight_layout()
    summary_filename = f"content_based_summary_plots_{timestamp}.png"
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - {summary_filename}")

if __name__ == "__main__":
    test_content_based_recommend()