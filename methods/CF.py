import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from surprise import accuracy
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

def train_svd_model(
    training_data,
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    biased=True,
    random_state=42
):
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
    model = SVD(n_factors=n_factors,n_epochs=n_epochs,lr_all=lr_all,reg_all=reg_all,biased=biased,random_state=random_state)
    model.fit(trainset)

    return model, trainset

from methods.cold_start import recommend_for_cold_start_profiles

def recommend_for_user(model, trainset, user_id, top_k=10, cold_start_recs=None):
    """
    给指定用户生成Top-K推荐列表
    """
    if cold_start_recs and user_id in cold_start_recs:
        return cold_start_recs[user_id][:top_k]
    # 获取训练集中所有物品（原始ID）
    all_items = set(trainset._raw2inner_id_items.keys())

    # 获取该用户已评分物品
    try:
        rated_items = {
            trainset.to_raw_iid(inner_iid)
            for (inner_uid, inner_iid, _) in trainset.all_ratings()
            if trainset.to_raw_uid(inner_uid) == user_id
        }

    except ValueError:
        # 用户不存在于训练集（冷启动）
        return cold_start_recs.get(user_id, []) if cold_start_recs else []

    # 未评分物品集合
    candidates = all_items - rated_items

    # 预测未评分物品的分数
    predictions = [(iid, model.predict(user_id, iid).est) for iid in candidates]

    # 按预测评分降序排序，取Top-K
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]

def recommend_for_all_users(model, trainset, train_profiles, top_k=10):
    """
    为训练集中的所有用户生成Top-K推荐字典
    
    Returns
    -------
    dict[str, list[tuple[int,float]]]
        {profile: [(anime_uid, predicted_score), ...]}
    """
    cold_start_recs = recommend_for_cold_start_profiles(train_profiles, n=top_k)
    user_recommendations = {}

    # 遍历训练集的所有用户原始ID
    all_users = [trainset.to_raw_uid(inner_id) for inner_id in trainset.all_users()]

    for user_id in tqdm(all_users, desc="Generating Recommendations"):
        recs = recommend_for_user(model, trainset, user_id, top_k=top_k, cold_start_recs=cold_start_recs)
        user_recommendations[user_id] = recs

    return user_recommendations

def recommend_for_all_users_fast(model, trainset, train_profiles, top_k=10):
    """
    批量生成所有用户Top-K推荐
    冷启动用户直接使用热门Top-K推荐
    """
    import numpy as np
    from methods.cold_start import recommend_for_cold_start_profiles

    num_users, num_items = trainset.n_users, trainset.n_items

    # 1️⃣ 先生成冷启动用户推荐字典
    cold_start_recs = recommend_for_cold_start_profiles(train_profiles, n=top_k)
    cold_start_users = set(cold_start_recs.keys())

    # 2️⃣ 计算完整预测矩阵（只用于非冷启动用户）
    pred_matrix = np.dot(model.pu, model.qi.T)
    if model.biased:  # 只有 biased=True 时才加偏置
        pred_matrix += model.trainset.global_mean
        pred_matrix += model.bu[:, np.newaxis]
        pred_matrix += model.bi[np.newaxis, :]

    # 3️⃣ 裁剪到评分区间，确保与 model.predict 一致
    min_rating, max_rating = trainset.rating_scale
    pred_matrix = np.clip(pred_matrix, min_rating, max_rating)

    # 3️⃣ 屏蔽已评分物品
    rated_mask = np.zeros_like(pred_matrix, dtype=bool)
    for inner_uid, inner_iid, _ in trainset.all_ratings():
        rated_mask[inner_uid, inner_iid] = True
    pred_matrix[rated_mask] = -np.inf

    # 4️⃣ 批量Top-K推荐
    top_k_indices = np.argpartition(-pred_matrix, top_k, axis=1)[:, :top_k]
    row_indices = np.arange(num_users)[:, None]
    top_k_sorted_idx = top_k_indices[
        row_indices, np.argsort(-pred_matrix[row_indices, top_k_indices])
    ]

    # 5️⃣ 构建推荐字典
    recommendations = {}
    for inner_uid in range(num_users):
        user_id = trainset.to_raw_uid(inner_uid)

        # 如果是冷启动用户 → 直接使用热门推荐
        if user_id in cold_start_users:
            recommendations[user_id] = cold_start_recs[user_id][:top_k]
            continue

        # 非冷启动用户 → SVD矩阵推荐
        top_items = top_k_sorted_idx[inner_uid]
        recs = [
            (int(trainset.to_raw_iid(inner_iid)), float(pred_matrix[inner_uid, inner_iid]))
            for inner_iid in top_items
        ]
        recommendations[user_id] = recs

    return recommendations


def auto_recommend_dump(top_k=10):
    # 1️⃣ 加载基础清洗后的数据
    profiles = get_clean_profiles()
    reviews = get_clean_reviews()

    # 2️⃣ 二次预处理
    profiles = profile_preprocess(profiles)
    reviews = review_preprocess(reviews)

    # 3️⃣ 选择必要列
    #profiles = profiles[['profile', 'favorites_anime']]
    #reviews = reviews[['uid', 'profile', 'anime_uid', 'score']]

    # 4️⃣ 划分训练集和测试集
    from preprocessing.split_dataset import split_profile
    train_profiles, test_profiles = split_profile(profiles, train_size=0.5, test_size=0.5)

    # 5️⃣ 构建SVD训练数据
    training_data = build_svd_training(
        reviews=reviews,
        train_profiles=train_profiles,
        test_profiles=test_profiles,
        score_for_fav=8
    )
    print("最终SVD训练集行数:", len(training_data))

    # 6️⃣ 训练SVD模型
    model, trainset = train_svd_model(training_data,n_factors=150,n_epochs=30,lr_all=0.005,reg_all=0.05,biased=False,random_state=42)

    testset = []
    for _, row in test_profiles.iterrows():
        user_id = str(row['profile'])
        for anime_id in row['favorites_anime']:
            testset.append((user_id, str(anime_id), 8))  # 假设收藏等于满分

    # 预测并计算 RMSE
    from surprise import accuracy
    pred = model.test(testset)
    rmse = accuracy.rmse(pred, verbose=True)
    print(f"RMSE on split test set favorites: {rmse:.4f}")

    # 7️⃣ 为所有用户生成推荐字典
    #recommendations = recommend_for_all_users(model, trainset, train_profiles, top_k=top_k)
    recommendations = recommend_for_all_users_fast(model, trainset, train_profiles, top_k=top_k)
    print(f"总共生成了 {len(recommendations)} 个用户的Top-{top_k}推荐")

    # 8️⃣ 可选：打印示例
    for user, recs in list(recommendations.items())[:3]:
        print(f"{user}: {recs}")

    with open("methods/svd_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False)

    return recommendations

def read_json_recommendations(file_path: str) -> Dict[str, List]:
    """
    从JSON文件读取推荐结果
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    rec_dict = auto_recommend_dump(top_k=10)

    with open("methods/svd_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(rec_dict, f, ensure_ascii=False)
