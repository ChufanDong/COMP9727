import os
from IO.load_csv import load_anime_data, load_profile_data, load_review_data
from preprocessing.clean import clean_animes, clean_profiles, clean_reviews


# 获取项目根目录（例如：animeRecommender/）
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 构建默认数据路径
def get_default_path(*subdirs):
    return os.path.join(BASE_DIR, "data", "archive", *subdirs)


def get_clean_animes(path=None):
    if path is None:
        path = get_default_path("animes.csv")
    return clean_animes(load_anime_data(path))


def get_clean_profiles(path=None):
    if path is None:
        path = get_default_path("profiles.csv")
    return clean_profiles(load_profile_data(path))


def get_clean_reviews(path=None):
    if path is None:
        path = get_default_path("reviews.csv")
    return clean_reviews(load_review_data(path))


def get_all_cleaned_data():
    return (
        get_clean_animes(),
        get_clean_profiles(),
        get_clean_reviews()
    )
