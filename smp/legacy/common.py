"""Shared feature handling for the SMP baselines and PolicyInduction runner."""

from __future__ import annotations

import pandas as pd

# Never model input: join keys, a path that embeds user + video ids, and a
# column that is "mp4" on every row.
DROP = ["pid", "uid", "vid", "video_path", "video_format", "popularity", "label"]
USER_STATS = [
    "user_follower_count",
    "user_following_count",
    "user_likes_count",
    "user_video_count",
    "user_digg_count",
    "user_heart_count",
    "user_friend_count",
]


def model_inputs(df: pd.DataFrame, user_stats: bool = True) -> pd.DataFrame:
    """The columns both methods see, as plain strings/numbers."""
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    if not user_stats:
        X = X.drop(columns=USER_STATS)
    X["post_suggested_words"] = X["post_suggested_words"].apply(
        lambda v: ", ".join(v) if v is not None and len(v) else "none"
    )
    X["post_time"] = X["post_time"].dt.strftime("%Y-%m-%d %H:%M")
    return X.reset_index(drop=True)
