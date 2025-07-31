import pandas as pd
from typing import List, Tuple, Dict

from cold_start import cold_start_top_n
from collaborative_basic import collaborative_recommend
from content_based     import content_recommend     

def recommend(
    profiles: pd.DataFrame,
    method: str = "collaborative",
    top_k: int = 20
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Returns {profile_id: [(anime_uid, score), …]} for ALL users.
    """
    cold_mask  = profiles["is_cold_start"] | (profiles["favorites_anime"].apply(len) <= 3)
    cold_users = profiles[cold_mask]
    warm_users = profiles[~cold_mask]

    if method == "collaborative":
        warm_recs = collaborative_recommend(warm_users, top_k)
    elif method == "content":
        warm_recs = content_recommend(warm_users, top_k)
    else:
        raise ValueError("method must be 'collaborative' or 'content'")

    if not cold_users.empty:
        cs_df   = cold_start_top_n(top_k)
        cs_list = list(zip(cs_df["uid"], cs_df["pop_score"]))
        cold_recs = {pid: cs_list for pid in cold_users["profile"]}
    else:
        cold_recs = {}

    return {**warm_recs, **cold_recs}