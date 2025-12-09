import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

try:
    from catboost import CatBoostRanker, Pool
except Exception:
    CatBoostRanker = None
    Pool = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None


# ===========================
# CONFIG
# ===========================
DATA_DIR = Path("")
RANDOM_STATE = 42
SPLIT_DAYS = 0  # set 0 to use full train for fitting
N_COLD = 10

# BERT
BERT_MODEL_NAME = "cointegrated/rubert-tiny2"
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 64
BERT_SVD_DIM = 64

# CatBoost - OPTIMIZED
CATBOOST_ITER = 1500
CATBOOST_LR = 0.05
CATBOOST_DEPTH = 6
EARLY_STOPPING_ROUNDS = 150

# ALS
ALS_FACTORS = 64


# ===========================
# HELPERS
# ===========================
def load_csv(name: str, prefer_merged: bool = True) -> pd.DataFrame:
    """Load CSV, preferring *_merged.csv when it exists."""
    base_path = DATA_DIR / name
    if prefer_merged and base_path.suffix == ".csv":
        merged_path = base_path.with_name(f"{base_path.stem}_merged{base_path.suffix}")
        if merged_path.exists():
            print(f">>> Loading merged file: {merged_path.name}")
            return pd.read_csv(merged_path)

    print(f">>> Loading file: {base_path.name}")
    return pd.read_csv(base_path)


def read_data():
    print(">>> Loading raw data...")
    train = load_csv("train.csv")
    books = load_csv("books.csv")
    users = load_csv("users.csv")
    genres = load_csv("genres.csv")
    book_genres = load_csv("book_genres.csv")
    candidates = load_csv("candidates.csv", prefer_merged=False)
    targets = load_csv("targets.csv", prefer_merged=False)
    book_descriptions = load_csv("book_descriptions.csv")

    train["timestamp"] = pd.to_datetime(train["timestamp"])
    return train, books, users, genres, book_genres, candidates, targets, book_descriptions


# ===========================
# COLLABORATIVE FILTERING (ALS)
# ===========================
def build_als_features(train_df: pd.DataFrame, n_factors: int = ALS_FACTORS) -> Tuple[Dict, Dict]:
    """Implicit ALS для получения user и book embeddings"""
    print(">>> Building ALS embeddings...")

    user_ids = train_df["user_id"].unique()
    book_ids = train_df["book_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    book_to_idx = {b: i for i, b in enumerate(book_ids)}

    rows, cols, data = [], [], []
    for _, row in train_df.iterrows():
        u_idx = user_to_idx[row["user_id"]]
        b_idx = book_to_idx[row["book_id"]]
        weight = 3.0 if row["has_read"] == 1 else 1.0
        rows.append(u_idx)
        cols.append(b_idx)
        data.append(weight)

    interaction_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(book_ids)),
    )

    print(f">>> Running ALS (SVD) with {n_factors} factors...")
    svd = TruncatedSVD(n_components=n_factors, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(interaction_matrix)
    book_factors = svd.components_.T

    user_embeddings = {uid: user_factors[idx] for uid, idx in user_to_idx.items()}
    book_embeddings = {bid: book_factors[idx] for bid, idx in book_to_idx.items()}

    global_user_emb = user_factors.mean(axis=0)
    global_book_emb = book_factors.mean(axis=0)

    return (user_embeddings, global_user_emb), (book_embeddings, global_book_emb)


# ===========================
# TEMPORAL FEATURES
# ===========================
def build_temporal_features(train_df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Временные признаки для пользователей"""
    print(">>> Building temporal features...")

    if reference_date is None:
        reference_date = train_df["timestamp"].max()

    temporal_dict = defaultdict(dict)

    for user_id, group in train_df.groupby("user_id"):
        group = group.sort_values("timestamp")

        last_interaction = (reference_date - group["timestamp"].max()).days
        first_interaction = (reference_date - group["timestamp"].min()).days
        activity_span = max(first_interaction - last_interaction, 1)

        n_interactions = len(group)
        interactions_per_day = n_interactions / activity_span

        temporal_dict[user_id]["days_since_last"] = last_interaction
        temporal_dict[user_id]["days_since_first"] = first_interaction
        temporal_dict[user_id]["activity_span"] = activity_span
        temporal_dict[user_id]["interactions_per_day"] = interactions_per_day
        temporal_dict[user_id]["total_interactions"] = n_interactions

        for days in [30, 90, 180]:
            cutoff = reference_date - pd.Timedelta(days=days)
            recent = group[group["timestamp"] >= cutoff]
            temporal_dict[user_id][f"interactions_last_{days}d"] = len(recent)
            temporal_dict[user_id][f"read_last_{days}d"] = recent["has_read"].sum()
            temporal_dict[user_id][
                f"read_ratio_last_{days}d"
            ] = recent["has_read"].mean() if len(recent) > 0 else 0

    temporal_df = pd.DataFrame.from_dict(temporal_dict, orient="index").reset_index()
    temporal_df.columns = ["user_id"] + list(temporal_df.columns[1:])

    return temporal_df.fillna(0)


# ===========================
# USER PREFERENCES
# ===========================
def build_user_preferences(
    train_df: pd.DataFrame,
    books_df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Предпочтения пользователей по жанрам и авторам"""
    print(">>> Building user preferences...")

    train_with_meta = train_df.merge(
        books_df[["book_id", "author_id"]],
        on="book_id",
        how="left",
    )
    train_with_genres = train_with_meta.merge(book_genres_df, on="book_id", how="left")

    # User-Genre preferences
    user_genre_stats = []
    for (user_id, genre_id), group in train_with_genres.groupby(["user_id", "genre_id"]):
        if pd.isna(genre_id):
            continue
        user_genre_stats.append(
            {
                "user_id": user_id,
                "genre_id": int(genre_id),
                "ug_interactions": len(group),
                "ug_read_count": group["has_read"].sum(),
                "ug_read_ratio": group["has_read"].mean(),
                "ug_avg_rating": group["rating"].mean(),
            }
        )

    user_genre_df = pd.DataFrame(user_genre_stats)

    # User-Author preferences
    user_author_stats = []
    for (user_id, author_id), group in train_with_meta.groupby(["user_id", "author_id"]):
        if pd.isna(author_id):
            continue
        user_author_stats.append(
            {
                "user_id": user_id,
                "author_id": int(author_id),
                "ua_interactions": len(group),
                "ua_read_count": group["has_read"].sum(),
                "ua_read_ratio": group["has_read"].mean(),
                "ua_avg_rating": group["rating"].mean(),
            }
        )

    user_author_df = pd.DataFrame(user_author_stats)

    return user_genre_df, user_author_df


# ===========================
# BERT FEATURES
# ===========================
def build_bert_book_features(
    book_descriptions: pd.DataFrame,
    used_book_ids: np.ndarray,
) -> pd.DataFrame:
    cache_path = Path("bert_book_svd.npz")

    if cache_path.exists():
        print(f">>> Loading cached BERT embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        book_ids = data["book_ids"]
        emb = data["emb"]
        df = pd.DataFrame(emb, columns=[f"bert_{i}" for i in range(emb.shape[1])])
        df["book_id"] = book_ids
        return df[df["book_id"].isin(used_book_ids)].reset_index(drop=True)

    if torch is None or AutoTokenizer is None or AutoModel is None:
        print("!!! torch / transformers not available, skipping BERT features")
        return pd.DataFrame({"book_id": list(used_book_ids)})

    print(">>> Computing BERT embeddings for books ...")

    descriptions_df = book_descriptions[
        book_descriptions["book_id"].isin(used_book_ids)
    ].copy()
    descriptions_df["description"] = descriptions_df["description"].fillna("")
    descriptions_df = descriptions_df.sort_values("book_id").reset_index(drop=True)

    book_ids = descriptions_df["book_id"].values
    texts = descriptions_df["description"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()

    all_embs = []

    for start in range(0, len(texts), BERT_BATCH_SIZE):
        end = min(start + BERT_BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LEN,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            last_hidden = outputs.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

        all_embs.append(pooled.cpu().numpy().astype("float32"))

        if (start // BERT_BATCH_SIZE) % 10 == 0:
            print(f"    processed {end}/{len(texts)} descriptions")

    emb = np.concatenate(all_embs, axis=0)
    print(f">>> Raw BERT embedding shape: {emb.shape}")

    if emb.shape[1] <= BERT_SVD_DIM:
        emb_svd = emb
    else:
        print(f">>> Running SVD to {BERT_SVD_DIM} dims ...")
        svd = TruncatedSVD(n_components=BERT_SVD_DIM, random_state=RANDOM_STATE)
        emb_svd = svd.fit_transform(emb)

    np.savez_compressed(cache_path, book_ids=book_ids, emb=emb_svd)

    df = pd.DataFrame(emb_svd, columns=[f"bert_{i}" for i in range(emb_svd.shape[1])])
    df["book_id"] = book_ids
    return df


# ===========================
# ENHANCED FEATURE ENGINEERING
# ===========================
def build_enhanced_features(
    cand_df: pd.DataFrame,
    train_df: pd.DataFrame,
    books_df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    user_embeddings: Tuple,
    book_embeddings: Tuple,
    user_genre_prefs: pd.DataFrame,
    user_author_prefs: pd.DataFrame,
    temporal_feats: pd.DataFrame,
    book_bert_df: pd.DataFrame,
) -> pd.DataFrame:
    print(">>> Building enhanced features...")

    # ВАЖНО: копия + reset_index, чтобы не таскать старые индексы
    df = cand_df.copy().reset_index(drop=True)

    # === 1. Basic metadata ===
    df = df.merge(books_df, on="book_id", how="left")
    df = df.reset_index(drop=True)

    # === 2. ALS Embeddings + Similarity ===
    user_emb_dict, global_user_emb = user_embeddings
    book_emb_dict, global_book_emb = book_embeddings

    print("   - Computing ALS similarity...")
    user_embs = np.array([user_emb_dict.get(uid, global_user_emb) for uid in df["user_id"]])
    book_embs = np.array([book_emb_dict.get(bid, global_book_emb) for bid in df["book_id"]])

    dot_products = np.sum(user_embs * book_embs, axis=1)
    user_norms = np.linalg.norm(user_embs, axis=1)
    book_norms = np.linalg.norm(book_embs, axis=1)
    df["als_similarity"] = dot_products / (user_norms * book_norms + 1e-9)
    df["als_dot"] = dot_products
    df["als_user_norm"] = user_norms
    df["als_book_norm"] = book_norms

    # === 3. User-Genre affinity ===
    print("   - Computing genre affinity...")

    book_to_genres = book_genres_df.groupby("book_id")["genre_id"].apply(list).to_dict()
    df["book_genres"] = df["book_id"].map(book_to_genres)

    user_genre_dict = {}
    for user_id, group in user_genre_prefs.groupby("user_id"):
        user_genre_dict[user_id] = {
            row["genre_id"]: (row["ug_read_ratio"], row["ug_interactions"])
            for _, row in group.iterrows()
        }

    def calculate_genre_affinity_safe(user_id, book_genres):
        if book_genres is None or len(book_genres) == 0:
            return 0.0

        user_prefs = user_genre_dict.get(user_id, {})
        if not user_prefs:
            return 0.0

        affinity_scores = []
        for genre_id in book_genres:
            if genre_id in user_prefs:
                read_ratio, interactions = user_prefs[genre_id]
                score = read_ratio * np.log1p(interactions)
                affinity_scores.append(score)

        return np.mean(affinity_scores) if affinity_scores else 0.0

    df["user_genre_affinity"] = df.apply(
        lambda row: calculate_genre_affinity_safe(row["user_id"], row["book_genres"]),
        axis=1,
    )

    df["num_book_genres"] = df["book_genres"].apply(lambda x: len(x) if x is not None else 0)
    df["has_user_preferred_genre"] = df.apply(
        lambda row: 1
        if (
            row["book_genres"] is not None
            and any(g in user_genre_dict.get(row["user_id"], {}) for g in row["book_genres"])
        )
        else 0,
        axis=1,
    )

    # === 4. User-Author affinity ===
    print("   - Computing author affinity...")

    user_author_dict = {}
    for user_id, group in user_author_prefs.groupby("user_id"):
        user_author_dict[user_id] = {
            row["author_id"]: (row["ua_read_ratio"], row["ua_interactions"])
            for _, row in group.iterrows()
        }

    def calculate_author_affinity_safe(user_id, author_id):
        if pd.isna(author_id):
            return 0.0

        author_id = int(author_id)
        user_prefs = user_author_dict.get(user_id, {})

        if author_id in user_prefs:
            read_ratio, interactions = user_prefs[author_id]
            return read_ratio * np.log1p(interactions)
        return 0.0

    df["user_author_affinity"] = df.apply(
        lambda row: calculate_author_affinity_safe(row["user_id"], row["author_id"]),
        axis=1,
    )

    df["has_read_author_before"] = df.apply(
        lambda row: 1
        if (not pd.isna(row["author_id"]) and int(row["author_id"]) in user_author_dict.get(row["user_id"], {}))
        else 0,
        axis=1,
    )

    # === 5. Temporal features ===
    print("   - Adding temporal features...")
    df = df.merge(temporal_feats, on="user_id", how="left")
    df = df.reset_index(drop=True)

    # === 6. Book popularity features ===
    print("   - Computing book popularity...")
    book_pop = train_df.groupby("book_id").agg(
        {
            "user_id": "count",
            "has_read": ["sum", "mean"],
            "rating": "mean",
        }
    )
    book_pop.columns = [
        "book_total_interactions",
        "book_read_count",
        "book_read_ratio",
        "book_avg_rating_train",
    ]
    book_pop = book_pop.reset_index()

    df = df.merge(book_pop, on="book_id", how="left")
    df = df.reset_index(drop=True)

    if "publication_year" in df.columns:
        current_year = 2021
        df["book_age"] = current_year - df["publication_year"]
        df["is_new_book"] = (df["book_age"] <= 2).astype(int)

    # === 7. User statistics ===
    print("   - Computing user statistics...")
    user_stats = train_df.groupby("user_id").agg(
        {
            "book_id": "count",
            "has_read": ["sum", "mean"],
            "rating": "mean",
        }
    )
    user_stats.columns = [
        "user_total_interactions",
        "user_read_count",
        "user_read_ratio",
        "user_avg_rating",
    ]
    user_stats = user_stats.reset_index()

    df = df.merge(user_stats, on="user_id", how="left")
    df = df.reset_index(drop=True)

    # === 8. Previous user-book interaction ===
    print("   - Checking previous interactions...")
    prev_interactions = set(zip(train_df["user_id"], train_df["book_id"]))
    df["had_prev_interaction"] = [
        1 if (uid, bid) in prev_interactions else 0 for uid, bid in zip(df["user_id"], df["book_id"])
    ]

    prev_details = (
        train_df.groupby(["user_id", "book_id"])
        .agg(
            {
                "has_read": "max",
                "rating": "mean",
            }
        )
        .reset_index()
    )
    prev_details.columns = ["user_id", "book_id", "prev_has_read", "prev_rating"]

    df = df.merge(prev_details, on=["user_id", "book_id"], how="left")
    df = df.reset_index(drop=True)
    df["prev_has_read"] = df["prev_has_read"].fillna(0).astype("int8")
    df["prev_rating"] = df["prev_rating"].fillna(0)

    # === 9. BERT similarity ===
    if not book_bert_df.empty and len(book_bert_df) > 0:
        print("   - Computing BERT similarity...")

        bert_cols = [c for c in book_bert_df.columns if c.startswith("bert_")]

        if bert_cols:
            user_book_read = (
                train_df[train_df["has_read"] == 1]
                .groupby("user_id")["book_id"]
                .apply(list)
                .to_dict()
            )

            book_bert_dict = {}
            for _, row in book_bert_df.iterrows():
                book_bert_dict[row["book_id"]] = row[bert_cols].values

            user_bert_profiles = {}
            for user_id, read_books in user_book_read.items():
                book_vecs = [book_bert_dict[bid] for bid in read_books if bid in book_bert_dict]
                if book_vecs:
                    user_bert_profiles[user_id] = np.mean(book_vecs, axis=0)

            bert_sims = []
            for uid, bid in zip(df["user_id"], df["book_id"]):
                if uid not in user_bert_profiles or bid not in book_bert_dict:
                    bert_sims.append(0.0)
                else:
                    user_vec = user_bert_profiles[uid]
                    book_vec = book_bert_dict[bid]
                    sim = np.dot(user_vec, book_vec) / (
                        np.linalg.norm(user_vec) * np.linalg.norm(book_vec) + 1e-9
                    )
                    bert_sims.append(sim)

            df["bert_similarity"] = bert_sims
        else:
            df["bert_similarity"] = 0.0
    else:
        df["bert_similarity"] = 0.0

    # === 10. Cross features ===
    print("   - Creating cross features...")
    df["user_book_rating_diff"] = df["user_avg_rating"] - df["book_avg_rating_train"]
    df["user_book_read_ratio_diff"] = df["user_read_ratio"] - df["book_read_ratio"]
    df["user_popularity_preference"] = df["user_total_interactions"] / (
        df["book_total_interactions"] + 1
    )

    if "language" in df.columns:
        user_lang_pref = (
            train_df.merge(books_df[["book_id", "language"]], on="book_id", how="left")
            .groupby("user_id")["language"]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else -1)
            .to_dict()
        )

        df["user_preferred_language"] = df["user_id"].map(user_lang_pref)
        df["is_preferred_language"] = (df["language"] == df["user_preferred_language"]).astype(int)
        df = df.drop("user_preferred_language", axis=1)

    # === 11. Fill missing values ===
    print("   - Filling missing values...")
    df = df.drop(["book_genres"], axis=1, errors="ignore")

    # !!! КРИТИЧЕСКИЙ ФИКС !!!
    # На всякий случай схлопываем дубликаты по (user_id, book_id),
    # аггрегируя числовые признаки средним, категориальные — первым значением.
    if "user_id" in df.columns and "book_id" in df.columns:
        print("   - Collapsing duplicates by (user_id, book_id) if any...")
        group_cols = ["user_id", "book_id"]
        agg_dict = {}
        for col in df.columns:
            if col in group_cols:
                continue
            if np.issubdtype(df[col].dtype, np.number):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"
        df = df.groupby(group_cols, as_index=False).agg(agg_dict)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("-1")

    print(f">>> Feature matrix shape: ({len(df)}, {df.shape[1]})")
    return df


# ===========================
# SMART CANDIDATES
# ===========================
def build_smart_candidates(
    train_df: pd.DataFrame,
    train_hist: pd.DataFrame,
    future_df: pd.DataFrame,
    user_genre_prefs: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    user_embeddings: Tuple,
    book_embeddings: Tuple,
) -> pd.DataFrame:
    """Build candidate pools with a mix of ALS hard negatives and genre/popular cold items."""
    print(">>> Building smart candidate pools...")

    rng = np.random.RandomState(RANDOM_STATE)

    user_emb_dict, global_user_emb = user_embeddings
    book_emb_dict, global_book_emb = book_embeddings

    book_pop = train_df.groupby("book_id").size().sort_values(ascending=False)
    top_book_ids = list(book_pop.index[:5000].tolist())
    top_book_set = set(top_book_ids)

    # Precompute ALS vectors for the most popular books to speed up per-user scoring
    top_book_embs = np.array([book_emb_dict.get(bid, global_book_emb) for bid in top_book_ids])

    user2books = train_df.groupby("user_id")["book_id"].apply(set).to_dict()

    genre_to_books = defaultdict(set)
    for _, row in book_genres_df.iterrows():
        genre_to_books[row["genre_id"]].add(row["book_id"])

    future_users = future_df["user_id"].unique()
    rows = []

    for u in future_users:
        user_future = future_df[future_df["user_id"] == u]
        books_read = user_future.loc[user_future["has_read"] == 1, "book_id"].tolist()
        books_planned = user_future.loc[user_future["has_read"] == 0, "book_id"].tolist()

        for b in books_read:
            rows.append((int(u), int(b), 2))
        for b in books_planned:
            rows.append((int(u), int(b), 1))

        user_books_all = user2books.get(u, set())
        forbid = set(books_read) | set(books_planned) | set(user_books_all)

        # Hard negatives: top ALS recommendations the user has not interacted with
        user_vec = user_emb_dict.get(u, global_user_emb)
        als_scores = top_book_embs.dot(user_vec)
        sorted_idx = np.argsort(-als_scores)

        cold_items = []
        als_target = max(5, N_COLD // 2)
        for idx in sorted_idx:
            bid = int(top_book_ids[idx])
            if bid in forbid:
                continue
            cold_items.append(bid)
            if len(cold_items) >= als_target:
                break

        # Genre/popular sampling to diversify negatives
        user_prefs = user_genre_prefs[user_genre_prefs["user_id"] == u]
        remaining_slots = N_COLD - len(cold_items)

        if remaining_slots > 0 and len(user_prefs) > 0:
            top_genres = user_prefs.nlargest(3, "ug_interactions")["genre_id"].tolist()
            genre_books = set()
            for genre_id in top_genres:
                genre_books.update(genre_to_books.get(genre_id, set()))

            genre_pool = list((genre_books & top_book_set) - forbid - set(cold_items))
            if genre_pool:
                take = min(remaining_slots, len(genre_pool))
                genre_sample = rng.choice(genre_pool, size=take, replace=False).tolist()
                cold_items.extend(genre_sample)
                remaining_slots = N_COLD - len(cold_items)

        if remaining_slots > 0:
            pop_pool = list(top_book_set - forbid - set(cold_items))
            if pop_pool:
                take = min(remaining_slots, len(pop_pool))
                pop_sample = rng.choice(pop_pool, size=take, replace=False).tolist()
                cold_items.extend(pop_sample)

        for bid in cold_items[:N_COLD]:
            rows.append((int(u), int(bid), 0))

    cand_df = pd.DataFrame(rows, columns=["user_id", "book_id", "relevance"])
    print(f">>> Generated {len(cand_df)} candidate pairs")
    return cand_df


# ===========================
# MAIN PIPELINE
# ===========================
def main():
    train, books, users, genres, book_genres, candidates, targets, book_descriptions = read_data()

    # Use full train for fitting; no hold-out by time to maximize training signal
    print(">>> Using full train data for fitting (no time split)")
    train_hist = train.copy()
    future = train.copy()

    used_book_ids = np.union1d(
        train["book_id"].unique(),
        candidates["book_id_list"].astype(str).str.split(",").explode().astype(int).unique(),
    )

    # Build feature components
    book_bert_df = build_bert_book_features(book_descriptions, used_book_ids)
    user_embeddings, book_embeddings = build_als_features(train)
    temporal_feats = build_temporal_features(train)
    user_genre_prefs, user_author_prefs = build_user_preferences(train, books, book_genres)

    cand_offline = build_smart_candidates(
        train,
        train_hist,
        future,
        user_genre_prefs,
        book_genres,
        user_embeddings,
        book_embeddings,
    )

    X_offline = build_enhanced_features(
        cand_offline,
        train_hist,
        books,
        book_genres,
        user_embeddings,
        book_embeddings,
        user_genre_prefs,
        user_author_prefs,
        temporal_feats,
        book_bert_df,
    )

    y_offline = cand_offline["relevance"].values
    user_ids_offline = cand_offline["user_id"].values

    # Prepare for CatBoost
    drop_cols = ["relevance", "user_id", "book_id", "title", "author_name"]
    drop_cols = [c for c in drop_cols if c in X_offline.columns]

    X_train = X_offline.drop(columns=drop_cols).reset_index(drop=True)

    cat_cols = []
    for col in ["gender", "language", "publisher", "author_id"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str).fillna("-1")
            cat_cols.append(col)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = X_train[numeric_cols].fillna(0)

    print(f">>> Training matrix: {X_train.shape}, Features: {list(X_train.columns[:10])}...")

    # Train CatBoost with early stopping
    if CatBoostRanker is None or Pool is None:
        raise ImportError("catboost is not installed")

    print(">>> Training CatBoostRanker with early stopping...")

    unique_users = np.unique(user_ids_offline)
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(unique_users)
    n_val_users = max(1, int(len(unique_users) * 0.2))
    val_users = set(unique_users[:n_val_users])

    is_val = np.isin(user_ids_offline, list(val_users))
    is_train = ~is_val

    train_idx = np.where(is_train)[0]
    val_idx = np.where(is_val)[0]

    X_tr = X_train.iloc[train_idx].reset_index(drop=True)
    y_tr = y_offline[train_idx]
    train_group = user_ids_offline[train_idx]

    X_val = X_train.iloc[val_idx].reset_index(drop=True)
    y_val = y_offline[val_idx]
    val_group = user_ids_offline[val_idx]

    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

    train_pool = Pool(
        data=X_tr,
        label=y_tr,
        group_id=train_group,
        cat_features=cat_feature_indices,
    )
    val_pool = Pool(
        data=X_val,
        label=y_val,
        group_id=val_group,
        cat_features=cat_feature_indices,
    )

    use_gpu = (
        torch is not None
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
    )

    model = CatBoostRanker(
        loss_function="YetiRankPairwise",
        eval_metric="NDCG:top=20",
        iterations=CATBOOST_ITER,
        learning_rate=CATBOOST_LR,
        depth=CATBOOST_DEPTH,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        rsm=0.8,
        bootstrap_type="Bayesian",
        random_seed=RANDOM_STATE,
        verbose=50,
        od_type="Iter",
        od_wait=EARLY_STOPPING_ROUNDS,
        task_type="GPU" if use_gpu else "CPU",
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    print(f">>> Best iteration: {model.get_best_iteration()}")
    best_scores = model.get_best_score()
    print(">>> Raw best_score dict:", best_scores)

    val_score = None
    if isinstance(best_scores, dict) and "validation" in best_scores:
        val_metrics = best_scores["validation"]
        if isinstance(val_metrics, dict):
            if "NDCG:top=20;type=Base" in val_metrics:
                val_score = val_metrics["NDCG:top=20;type=Base"]
            elif "NDCG:top=20" in val_metrics:
                val_score = val_metrics["NDCG:top=20"]

    if val_score is not None:
        print(f">>> Best NDCG@20: {val_score:.6f}")
    else:
        print(">>> Best NDCG@20 not found in get_best_score(), see dict above.")

    # Predict on test
    print(">>> Preparing real test candidates...")

    cand_test_rows = []
    for _, row in candidates.iterrows():
        user_id = row["user_id"]
        book_ids = [int(b.strip()) for b in str(row["book_id_list"]).split(",") if b.strip()]
        for book_id in book_ids:
            cand_test_rows.append({"user_id": user_id, "book_id": book_id})

    cand_test = pd.DataFrame(cand_test_rows).reset_index(drop=True)

    X_test = build_enhanced_features(
        cand_test,
        train,
        books,
        book_genres,
        user_embeddings,
        book_embeddings,
        user_genre_prefs,
        user_author_prefs,
        temporal_feats,
        book_bert_df,
    )

    X_test_model = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns]).reset_index(
        drop=True
    )

    for col in cat_cols:
        if col in X_test_model.columns:
            X_test_model[col] = X_test_model[col].astype(str).fillna("-1")

    numeric_cols = X_test_model.select_dtypes(include=[np.number]).columns
    X_test_model[numeric_cols] = X_test_model[numeric_cols].fillna(0)

    # sanity-check: размеры должны совпасть
    print(f">>> cand_test rows: {len(cand_test)}, X_test_model rows: {len(X_test_model)}")

    if len(X_test_model) != len(cand_test):
        raise RuntimeError(
            f"Row mismatch after feature building: cand_test={len(cand_test)}, X_test_model={len(X_test_model)}"
        )

    print(">>> Predicting...")
    scores = model.predict(X_test_model)

    cand_test["score"] = scores
    cand_test = cand_test.sort_values(["user_id", "score"], ascending=[True, False])

    submission_rows = []
    for user_id in targets["user_id"]:
        user_preds = cand_test[cand_test["user_id"] == user_id]
        top_20 = user_preds.head(20)
        book_list = ",".join(map(str, top_20["book_id"].tolist()))
        submission_rows.append({"user_id": user_id, "book_id_list": book_list})

    submission = pd.DataFrame(submission_rows)
    submission.to_csv("submission.csv", index=False)
    print(">>> Submission saved to submission.csv")
    print(f">>> Submission shape: {submission.shape}")


if __name__ == "__main__":
    main()
