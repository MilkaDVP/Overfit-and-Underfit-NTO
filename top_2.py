import os
import random
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker, Pool
from implicit.als import AlternatingLeastSquares
from rectools.dataset import Dataset
from rectools.models import ImplicitALSWrapperModel


BASE_SEED = 42
VAL_SHARE = 0.15
ALS_FACTORS = 50
ALS_ITERS = 15
ALS_REG = 0.01
ALS_TOP_K = 15
CATBOOST_ITERS = 1000
CATBOOST_LR = 0.05
CATBOOST_DEPTH = 6
CATBOOST_L2 = 3.0
DATA_ROOT = Path(".")  # <-- укажи путь к корневой папке с CSV


def arm_rng(seed: int = BASE_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_sources(data_root: Path) -> Dict[str, pd.DataFrame]:
    def grab_csv(name: str) -> pd.DataFrame:
        return pd.read_csv(data_root / name)

    stash = {
        "train": grab_csv("train.csv"),
        "candidates": grab_csv("candidates.csv"),
        "books": grab_csv("books.csv"),
        "users": grab_csv("users.csv"),
        "book_genres": grab_csv("book_genres.csv"),
        "genres": grab_csv("genres.csv"),
        "book_descriptions": grab_csv("book_descriptions.csv"),
    }
    return stash


def build_als_candidates(interactions: pd.DataFrame) -> pd.DataFrame:
    work_df = interactions.copy()
    work_df["target"] = (work_df["has_read"] > 0).astype(int)
    work_df["weight"] = work_df["has_read"]

    als_ready = work_df[["user_id", "book_id", "weight", "timestamp"]].rename(
        columns={"book_id": "item_id", "timestamp": "datetime"}
    )
    als_ready["datetime"] = pd.to_datetime(als_ready["datetime"])

    ds = Dataset.construct(interactions_df=als_ready)

    als_core = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERS,
        regularization=ALS_REG,
        random_state=BASE_SEED,
        dtype=np.float32,
    )
    als_model = ImplicitALSWrapperModel(model=als_core, verbose=0)
    als_model.fit(ds)

    all_users = ds.user_id_map.external_ids.tolist()

    def recommend_for_user(uid: int, top_n: int = ALS_TOP_K) -> Optional[pd.DataFrame]:
        user_list = [int(u) for u in ds.user_id_map.external_ids.tolist()]
        if uid not in user_list:
            print(f"[skip] uid={uid} absent in mapping")
            return None

        recs = als_model.recommend(
            users=[uid],
            dataset=ds,
            k=top_n,
            filter_viewed=True,
            add_rank_col=True,
        )
        recs["user_id"] = uid
        recs = recs.rename(
            columns={"item_id": "book_id", "score": "als_score", "rank": "candidate_rank"}
        )
        return recs

    print("\n>>> ALS candidate sweep starting...")
    collector: List[pd.DataFrame] = []
    for user_ext in tqdm(all_users):
        user_int = int(user_ext)
        user_recs = recommend_for_user(user_int, top_n=ALS_TOP_K)
        if user_recs is not None and not user_recs.empty:
            collector.append(user_recs)

    if not collector:
        return pd.DataFrame()

    bag = pd.concat(collector, ignore_index=True)
    print(f"\n>>> ALS rows collected: {len(bag)}")
    print("\n>>> Peek rows:")
    print(bag.head(20))

    bag.to_csv("als_candidates.csv", index=False)
    bag = bag.drop(columns=["als_score", "candidate_rank"])
    return bag


def split_train_val(train_frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    copy_df = train_frame.copy()
    if copy_df["timestamp"].dtype == object:
        copy_df["timestamp"] = pd.to_datetime(copy_df["timestamp"])

    chron = copy_df.sort_values("timestamp")
    val_size = int(len(chron) * VAL_SHARE)
    train_size = len(chron) - val_size

    train_part = chron.iloc[:train_size].copy()
    val_part = chron.iloc[train_size:].copy()

    for part in (train_part, val_part):
        part.drop(columns=["timestamp"], inplace=True)
        part.reset_index(drop=True, inplace=True)

    return {"train": train_part, "val": val_part}


def prepare_training_tables(
    pos_train: pd.DataFrame,
    pos_val: pd.DataFrame,
    neg_pool: pd.DataFrame,
    books_meta: pd.DataFrame,
    users_meta: pd.DataFrame,
):
    neg_pool = neg_pool.copy()
    neg_pool["has_read"] = 0

    neg_train, neg_val = train_test_split(
        neg_pool, test_size=VAL_SHARE, random_state=BASE_SEED, shuffle=True
    )

    train_mix = pd.concat([neg_train, pos_train], axis=0)
    val_mix = pd.concat([neg_val, pos_val], axis=0)

    train_mix = train_mix.merge(books_meta, on="book_id").merge(users_meta, on="user_id")
    val_mix = val_mix.merge(books_meta, on="book_id").merge(users_meta, on="user_id")

    return train_mix, val_mix


def fit_ranker(train_df: pd.DataFrame, val_df: pd.DataFrame) -> CatBoostRanker:
    cat_feats = ["book_id", "author_id", "author_name", "language", "publisher", "genre_id", "gender"]
    num_feats = ["publication_year", "avg_rating", "age"]

    feature_cols = cat_feats + num_feats
    target_col = "has_read"
    group_col = "user_id"

    train_sorted = train_df.sort_values(group_col)
    val_sorted = val_df.sort_values(group_col)

    X_train = train_sorted[feature_cols]
    y_train = train_sorted[target_col]
    g_train = train_sorted[group_col]

    X_val = val_sorted[feature_cols]
    y_val = val_sorted[target_col]
    g_val = val_sorted[group_col]

    pool_train = Pool(data=X_train, label=y_train, group_id=g_train, cat_features=cat_feats)
    pool_val = Pool(data=X_val, label=y_val, group_id=g_val, cat_features=cat_feats)

    params = {
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=20",
        "iterations": CATBOOST_ITERS,
        "learning_rate": CATBOOST_LR,
        "depth": CATBOOST_DEPTH,
        "l2_leaf_reg": CATBOOST_L2,
        "random_seed": BASE_SEED,
        "thread_count": -1,
        "early_stopping_rounds": 50,
        "use_best_model": True,
        "verbose": 5,
    }

    print(">>> CatBoost ranker fit in progress...")
    model = CatBoostRanker(**params)
    model.fit(pool_train, eval_set=pool_val)
    model.save_model("catboost_ranker_model.cbm")
    return model


def explode_candidates(raw_candidates: pd.DataFrame) -> pd.DataFrame:
    exploded = (
        raw_candidates.copy()
        .assign(book_id_list=lambda x: x["book_id_list"].str.split(","))
        .explode("book_id_list")
        .rename(columns={"book_id_list": "book_id"})
        .reset_index(drop=True)
    )
    exploded["book_id"] = exploded["book_id"].str.strip().astype(int)
    exploded = exploded.drop_duplicates(["user_id", "book_id"])
    return exploded


def rank_and_save_submission(
    model: CatBoostRanker,
    candidates_df: pd.DataFrame,
    books_df: pd.DataFrame,
    users_df: pd.DataFrame,
    output_name: str = "faygo2.csv",
) -> pd.DataFrame:
    test_expanded = explode_candidates(candidates_df)
    enriched = test_expanded.merge(books_df, on="book_id", how="left").merge(
        users_df, on="user_id", how="left"
    )

    feature_names = model.feature_names_
    payload = pd.DataFrame()

    for feat in feature_names:
        if feat not in enriched.columns:
            continue
        if feat in {"author_name", "language", "publisher", "genre_id", "gender", "book_id", "author_id"}:
            payload[feat] = enriched[feat].fillna("unknown").astype(str)
        else:
            payload[feat] = pd.to_numeric(enriched[feat], errors="coerce").fillna(0)

    cat_cols = ["book_id", "author_id", "author_name", "language", "publisher", "genre_id", "gender"]
    predict_pool = Pool(data=payload, cat_features=cat_cols)

    enriched["prediction_score"] = model.predict(predict_pool)

    original_counts = (
        candidates_df.assign(book_count=lambda x: x["book_id_list"].str.split(",").apply(len))
        .set_index("user_id")["book_count"]
        .to_dict()
    )

    submit_rows = []
    for uid, grp in enriched.groupby("user_id"):
        ordered = grp.sort_values("prediction_score", ascending=False).drop_duplicates("book_id")
        top_n = original_counts.get(uid, len(ordered))
        top_ids = ordered.head(top_n)["book_id"].astype(str).tolist()
        top_ids = top_ids[:20]
        submit_rows.append({"user_id": uid, "book_id_list": ",".join(top_ids)})

    submit_df = pd.DataFrame(submit_rows).reset_index(drop=True)
    submit_df.to_csv(output_name, index=False)
    print(f">>> Submission dumped: {output_name}")
    return submit_df


def main():
    arm_rng()
    data_root = DATA_ROOT

    frames = read_sources(data_root)
    interactions = frames["train"].copy()
    interactions["has_read"] += 1

    raw_candidates = frames["candidates"]
    books_df = frames["books"]
    users_df = frames["users"]
    book_genres_df = frames["book_genres"]

    als_candidates = build_als_candidates(interactions)
    books_enriched = books_df.merge(book_genres_df, on="book_id")

    interactions = interactions.drop(columns=["rating"])
    splits = split_train_val(interactions)

    train_table, val_table = prepare_training_tables(
        splits["train"], splits["val"], als_candidates, books_enriched, users_df
    )

    ranker = fit_ranker(train_table, val_table)
    rank_and_save_submission(ranker, raw_candidates, books_enriched, users_df)


if __name__ == "__main__":
    main()
