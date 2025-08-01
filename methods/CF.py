import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Any, List
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import json
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

import pandas as pd

def build_svd_training(
    reviews: pd.DataFrame,
    train_profiles: pd.DataFrame,
    test_profiles: pd.DataFrame,
    score_for_fav: int = 7
) -> pd.DataFrame:
    """
    构建适合SVD训练的增强评分数据集：
    1. 删除不在训练集用户列表中的评分
    2. 给训练集用户的喜欢动漫列表赋固定分
    3. 删除测试集喜欢动漫可能泄露的评分

    Parameters
    ----------
    reviews : pd.DataFrame
        原始评分数据，至少包含 ['uid','profile','anime_uid','score']
    train_profiles : pd.DataFrame
        训练集用户喜欢列表 DataFrame，至少包含 ['profile','favorites_anime']
    test_profiles : pd.DataFrame
        测试集用户喜欢列表 DataFrame
    score_for_fav : int
        给训练集喜欢动漫的固定评分（默认 7）

    Returns
    -------
    pd.DataFrame
        增强后的干净评分数据，可直接用于 SVD 训练
    """

    # 1️⃣ 只保留训练集用户
    train_users = set(train_profiles['profile'])
    reviews_filtered = reviews[reviews['profile'].isin(train_users)].copy()

    # 2️⃣ 构建训练集喜欢列表固定评分
    extra_rows = []
    for _, row in train_profiles.iterrows():
        user = row['profile']
        for anime in row['favorites_anime']:
            extra_rows.append({
                'uid': None,  # 可选：SVD不依赖uid
                'profile': user,
                'anime_uid': int(anime),
                'score': score_for_fav
            })
    train_extra_df = pd.DataFrame(extra_rows)

    # 3️⃣ 合并增强评分
    augmented_reviews = pd.concat([reviews_filtered, train_extra_df], ignore_index=True)

    # 同一用户-动漫重复评分：保留原始评分，丢弃7分填充
    augmented_reviews = augmented_reviews.drop_duplicates(
        subset=['profile', 'anime_uid'],
        keep='first'
    )

    # 4️⃣ 删除测试集冲突评分（避免数据泄露）
    test_conflicts = set()
    for _, row in test_profiles.iterrows():
        user = row['profile']
        for anime in row['favorites_anime']:
            test_conflicts.add((user, int(anime)))

    mask_conflict = augmented_reviews.apply(
        lambda x: (x['profile'], x['anime_uid']) in test_conflicts, axis=1
    )
    cleaned_reviews = augmented_reviews[~mask_conflict].reset_index(drop=True)

    cleaned_reviews = cleaned_reviews.drop_duplicates().reset_index(drop=True)

    print("原始评分数量:", len(reviews))
    print("训练用户评分数量:", len(reviews_filtered))
    print("增强后评分数量:", len(augmented_reviews))
    print("清理冲突后评分数量:", len(cleaned_reviews))

    return cleaned_reviews

""" profiles = get_clean_profiles()
reviews = get_clean_reviews()

# 二次预处理（如果你要做模型训练）
profiles = profile_preprocess(profiles)
reviews = review_preprocess(reviews)

profiles = profiles[['profile', 'favorites_anime']]  # 只保留必要列
reviews = reviews[['uid', 'profile', 'anime_uid', 'score']]  # 只保留SVD需要的列

from preprocessing.split_dataset import split_profile

train_profiles, test_profiles = split_profile(profiles, train_size=0.8, test_size=0.2)

training_data = build_svd_training(
    reviews=reviews,
    train_profiles=train_profiles,
    test_profiles=test_profiles,
    score_for_fav=7
)

print("最终SVD训练集行数:", len(training_data))
print(training_data.head()) """

from surprise import SVD, Dataset, Reader

def train_svd_model(training_data, n_factors=50, n_epochs=20, random_state=42):
    """
    用Surprise的SVD模型训练评分矩阵

    Parameters
    ----------
    training_data : pd.DataFrame
        至少包含 ['profile','anime_uid','score'] 三列
    n_factors : int
        潜在因子数量
    n_epochs : int
        训练迭代次数
    random_state : int
        随机种子，保证可复现

    Returns
    -------
    model : surprise.SVD
        训练好的SVD模型
    trainset : surprise.Trainset
        Surprise内部训练集，可用于预测
    """
    # 构建Surprise数据集
    reader = Reader(rating_scale=(training_data['score'].min(), training_data['score'].max()))
    data = Dataset.load_from_df(training_data[['profile','anime_uid','score']], reader)
    trainset = data.build_full_trainset()

    # 训练SVD模型
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
    model.fit(trainset)

    return model, trainset

def recommend_for_user(model, trainset, user_id, top_k=10):
    """
    给指定用户生成Top-K推荐列表
    """
    # 获取训练集中所有物品（原始ID）
    all_items = set(trainset._raw2inner_id_items.keys())

    # 获取该用户已评分物品
    try:
        rated_items = {
            iid
            for (uid, iid, _) in trainset.all_ratings()
            if trainset.to_raw_uid(uid) == user_id
        }
    except ValueError:
        # 用户不存在于训练集（冷启动）
        return []

    # 未评分物品集合
    candidates = all_items - rated_items

    # 预测未评分物品的分数
    predictions = [(iid, model.predict(user_id, iid).est) for iid in candidates]

    # 按预测评分降序排序，取Top-K
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]

def recommend_for_all_users(model, trainset, top_k=10):
    """
    为训练集中的所有用户生成Top-K推荐字典
    
    Returns
    -------
    dict[str, list[tuple[int,float]]]
        {profile: [(anime_uid, predicted_score), ...]}
    """
    user_recommendations = {}

    # 遍历训练集的所有用户原始ID
    all_users = [trainset.to_raw_uid(inner_id) for inner_id in trainset.all_users()]

    for user_id in tqdm(all_users, desc="Generating Recommendations"):
        recs = recommend_for_user(model, trainset, user_id, top_k=top_k)
        user_recommendations[user_id] = recs

    return user_recommendations

def main(top_k=10):
    # 1️⃣ 加载基础清洗后的数据
    profiles = get_clean_profiles()
    reviews = get_clean_reviews()

    # 2️⃣ 二次预处理
    profiles = profile_preprocess(profiles)
    reviews = review_preprocess(reviews)

    # 3️⃣ 选择必要列
    profiles = profiles[['profile', 'favorites_anime']]
    reviews = reviews[['uid', 'profile', 'anime_uid', 'score']]

    # 4️⃣ 划分训练集和测试集
    from preprocessing.split_dataset import split_profile
    train_profiles, test_profiles = split_profile(profiles, train_size=0.8, test_size=0.2)

    # 5️⃣ 构建SVD训练数据
    training_data = build_svd_training(
        reviews=reviews,
        train_profiles=train_profiles,
        test_profiles=test_profiles,
        score_for_fav=7
    )
    print("最终SVD训练集行数:", len(training_data))

    # 6️⃣ 训练SVD模型
    model, trainset = train_svd_model(training_data, n_factors=50, n_epochs=20)

    # 7️⃣ 为所有用户生成推荐字典
    recommendations = recommend_for_all_users(model, trainset, top_k=top_k)
    print(f"总共生成了 {len(recommendations)} 个用户的Top-{top_k}推荐")

    # 8️⃣ 可选：打印示例
    for user, recs in list(recommendations.items())[:3]:
        print(f"{user}: {recs}")

    return recommendations

if __name__ == "__main__":
    rec_dict = main(top_k=10)

    with open("svd_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(rec_dict, f, ensure_ascii=False, indent=2)
