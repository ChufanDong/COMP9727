import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd
import ast

def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'archive'))

    animes_path   = os.path.join(DATA_DIR, 'animes.csv')     
    profiles_path = os.path.join(DATA_DIR, 'profiles.csv')
    reviews_path  = os.path.join(DATA_DIR, 'reviews.csv')

    animes   = pd.read_csv(animes_path)
    profiles = pd.read_csv(profiles_path, converters={'favorites': ast.literal_eval})
    reviews  = pd.read_csv(reviews_path)
    return animes, profiles, reviews

def prepare_genre_series(animes):
    genres = (animes.dropna(subset=['genre'])
                     .assign(genre=lambda df: df['genre'].apply(ast.literal_eval))
                     .explode('genre')['genre'])
    return genres

def plot_genre_distribution(genres, top_n=15):
    top_genres = genres.value_counts().head(top_n)
    plt.figure(figsize=(8,5))
    top_genres[::-1].plot(kind='barh')
    plt.title(f"Top {top_n} Genres by Anime Count")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution(reviews):
    plt.figure()
    reviews['score'].hist(bins=10)
    plt.title("Distribution of Review Scores")
    plt.xlabel("Score (1–10)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.show()

def plot_episode_vs_popularity(animes):
    subset = animes[['episodes','members']].dropna()
    plt.figure()
    plt.scatter(subset['episodes'], subset['members'], alpha=0.3)
    plt.title("Episode Count vs Popularity (Members)")
    plt.xlabel("Episodes")
    plt.ylabel("Members")
    plt.tight_layout()
    plt.show()

def plot_score_drift(animes):
    animes['year'] = pd.to_datetime(animes['aired'], errors='coerce').dt.year
    yearly = (animes.dropna(subset=['year'])
                    .groupby('year')['score']
                    .median())
    plt.figure()
    yearly.plot()
    plt.title("Median Anime Score Over Years")
    plt.xlabel("Year Aired")
    plt.ylabel("Median Score")
    plt.tight_layout()
    plt.show()

def identify_cult_and_hype(animes):
    members_p10 = animes['members'].quantile(0.10)
    members_p90 = animes['members'].quantile(0.90)
    score_p90   = animes['score'].quantile(0.90)
    score_med   = animes['score'].median()

    cult_hits = (animes[(animes['members'] < members_p10) & 
                        (animes['score'] > score_p90)]
                 .sort_values('score', ascending=False)
                 .loc[:, ['title','score','members']].head(20))

    over_hyped = (animes[(animes['members'] > members_p90) & 
                         (animes['score'] < score_med)]
                  .sort_values('members', ascending=False)
                  .loc[:, ['title','score','members']].head(20))
    return cult_hits, over_hyped

def plot_fav_length_distribution(profiles):
    lengths = profiles['favorites_anime'].apply(len)
    plt.figure()
    lengths.hist(bins=20)
    plt.title("Distribution of Favourite List Lengths")
    plt.xlabel("Number of Favourites")
    plt.ylabel("Users")
    plt.tight_layout()
    plt.show()
    return lengths

def topic_modelling(reviews, n_topics=5, sample_size=10000, top_words=10):
    sample_texts = reviews['text'].dropna().sample(min(sample_size, len(reviews)), random_state=42)
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    X = tfidf.fit_transform(sample_texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = np.array(tfidf.get_feature_names_out())
    topics = {}
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-top_words:][::-1]
        topics[f"Topic {idx+1}"] = words[top_idx]
    topics_df = pd.DataFrame(topics)
    return topics_df

# ---- Main workflow ----
animes, profiles, reviews = load_data()

# 1. Which genres dominate?
genre_series = prepare_genre_series(animes)
plot_genre_distribution(genre_series)

# 2. Rating distribution
plot_rating_distribution(reviews)

# 3. Episode count vs popularity
plot_episode_vs_popularity(animes)

# 4. Score drift over years
plot_score_drift(animes)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 6 & 7  – cult hits / over-hyped
cult_hits_df, over_hyped_df = identify_cult_and_hype(animes)
cult_hits_df.to_csv(os.path.join(SCRIPT_DIR, "cult_hits_top20.csv"), index=False)
over_hyped_df.to_csv(os.path.join(SCRIPT_DIR, "overhyped_top20.csv"), index=False)
print("Saved cult-hits and over-hyped tables next to the script.")

# 8 – favourite-list length distribution plot (unchanged)
fav_lengths = plot_fav_length_distribution(profiles)

# 11 – topic-model keywords
topics_df = topic_modelling(reviews)
topics_df.to_csv(os.path.join(SCRIPT_DIR, "top_words_per_topic.csv"), index=False)
print("Saved topic-keywords CSV next to the script.")