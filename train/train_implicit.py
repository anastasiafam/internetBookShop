import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import implicit

import warnings
warnings.filterwarnings("ignore")

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CONFIG = {
    "data_paths": {
        "users": "../data/Users.csv",
        "books": "../data/Book.csv",
        "ratings": "../data/Ratings.csv",
        "books_add": "../data/Book_additional_info2.csv"
    },
    "filtering": {
        "min_user_interactions": 5,
        "min_book_interactions": 3,
    },
    "model_params": {
        "content_n_components": 100,

        # implicit ALS
        "implicit_factors": 32,
        "implicit_regularization": 0.05,
        "implicit_iterations": 30,
        "implicit_alpha": 20.0,

        "random_state": 42,
        "eval_k": 100,
    },
}

# ======================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ======================
def preprocess_genres(genre_str):
    try:
        # Преобразуем строку в список, если это строковое представление списка
        genres_list = ast.literal_eval(genre_str)
    except (ValueError, SyntaxError):
        # Если не получилось — считаем, что это уже строка
        genres_list = [genre_str] if pd.notna(genre_str) else []

    # Объединяем в строку через запятую, в нижнем регистре, без кавычек
    return ', '.join(str(g).strip().lower() for g in genres_list if pd.notna(g) and str(g).strip())

def prepare_implicit_data():
    ratings = pd.read_csv(CONFIG["data_paths"]["ratings"], low_memory=False)
    books = pd.read_csv(CONFIG["data_paths"]["books"], low_memory=False)
    books_additional = pd.read_csv(CONFIG["data_paths"]["books_add"], low_memory=False)
    books_additional = books_additional[['isbn','language','subjects']]
    books_additional.fillna('no_info', inplace=True)
    books_additional['subjects'] = books_additional['subjects'].apply(preprocess_genres)
    books = books.merge(books_additional, left_on ='ISBN', right_on ='isbn')
    print(books.columns)
    print(f"Исходные данные: {len(ratings)} записей, {len(books)} книг")

    # очистка текста по книгам
    for col in ["Book-Title", "Book-Author", "Publisher"]:
        if col in books.columns:
            books[col] = books[col].astype(str).str.lower().str.strip()
        else:
            books[col] = ""
      # ЯВНЫЕ оценки (>0) — для avg_rating и rating_count
    explicit_ratings = ratings[ratings["Book-Rating"] > 0].copy()

    book_stats = (
        explicit_ratings.groupby("ISBN")
        .agg(
            avg_rating=("Book-Rating", "mean"),
            rating_count=("Book-Rating", "count"),
        )
        .reset_index()
    )

    # сохраняем для отладки при необходимости
    book_stats.to_csv("../data/mean_rating_explicit.csv", index=False)

    # мёрджим статистику в таблицу книг
    books = books.merge(book_stats, on="ISBN", how="left")

    books["avg_rating"] = books["avg_rating"].fillna(0.0)
    books["rating_count"] = books["rating_count"].fillna(0).astype(int)

    # год публикации и декада
    books["Year-Of-Publication"] = pd.to_numeric(
        books["Year-Of-Publication"], errors="coerce"
    )
    books["publication_decade"] = (
        (books["Year-Of-Publication"] // 10 * 10)
        .fillna(0)
        .astype(int)
    )

    books.to_csv("../data/books_prepared.csv", index=False)

    # минимум 4 явные оценки > 0
    min_explicit_per_user = 30

    explicit = ratings[ratings["Book-Rating"] > 0]
    exp_counts = explicit.groupby("User-ID")["Book-Rating"].count()

    users_with_enough_explicit = exp_counts[exp_counts >= min_explicit_per_user].index

    ratings = ratings[ratings["User-ID"].isin(users_with_enough_explicit)]

    print(
        f"После фильтра по явным оценкам (>= {min_explicit_per_user}): "
        f"{len(ratings)} записей, {ratings['User-ID'].nunique()} пользователей"
    )

    # удаляем дубликаты по ISBN
    books = books.drop_duplicates(subset=["ISBN"], keep="first")

    # фильтрация пользователей/книг по количеству взаимодействий (0 и >0)
    user_counts = ratings["User-ID"].value_counts()
    item_counts = ratings["ISBN"].value_counts()

    ratings = ratings[
        ratings["User-ID"].isin(
            user_counts[user_counts >= CONFIG["filtering"]["min_user_interactions"]].index
        )
        & ratings["ISBN"].isin(
            item_counts[item_counts >= CONFIG["filtering"]["min_book_interactions"]].index
        )
    ]

    print(f"После фильтрации: {len(ratings)} записей")

  
    print(f"Пользователей: {ratings['User-ID'].nunique()}")
    print(f"Книг: {ratings['ISBN'].nunique()}")
    print(f"Средний # взаимодействий/пользователь: {len(ratings) / ratings['User-ID'].nunique():.1f}")
    print(f"Медиана взаимодействий: {ratings.groupby('User-ID').size().median():.1f}")
    return books, ratings


# ======================
# CONTENT-BASED МОДЕЛЬ
# ======================


def build_content_model(books: pd.DataFrame):
    books = books.copy()

    # Текстовое поле без числовых признаков в качестве токенов
    books["content"] = (
        books["Book-Author"].fillna("").astype(str)
        + " "
        + books["Book-Title"].fillna("").astype(str)
        + " "
        + books["Publisher"].fillna("").astype(str)
        + " "
        + books["publication_decade"].astype(str)
        + " "
        + books["subjects"].astype(str)
        + " "
        + books["language"].astype(str)
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
    )
    tfidf_matrix = tfidf.fit_transform(books["content"])
    svd = TruncatedSVD(
        n_components=CONFIG["model_params"]["content_n_components"],
        random_state=CONFIG["model_params"]["random_state"],
    )
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # безопасная нормализация
    norms = np.linalg.norm(tfidf_reduced, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    tfidf_reduced = tfidf_reduced / norms
    tfidf_reduced = np.nan_to_num(tfidf_reduced, nan=0.0, posinf=0.0, neginf=0.0)

    assert np.isfinite(tfidf_reduced).all(), "tfidf_reduced содержит нечисловые значения"

    isbn_to_idx = {isbn: idx for idx, isbn in enumerate(books["ISBN"])}
    idx_to_isbn = {idx: isbn for isbn, idx in isbn_to_idx.items()}

    return {
        "tfidf_reduced": tfidf_reduced,
        "isbn_to_idx": isbn_to_idx,
        "idx_to_isbn": idx_to_isbn,
        "books": books,
        "tfidf": tfidf,
        "svd": svd,
    }


# ======================
# БИНАРНАЯ МАТРИЦА (для оценки)
# ======================


def create_binary_interactions(ratings, user_to_idx, item_to_idx):
    rows = ratings["User-ID"].map(user_to_idx).values
    cols = ratings["ISBN"].map(item_to_idx).values

    valid_mask = (~pd.isna(rows)) & (~pd.isna(cols))
    rows = rows[valid_mask].astype(np.int32)
    cols = cols[valid_mask].astype(np.int32)

    data = np.ones(len(rows), dtype=np.float32)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_to_idx), len(item_to_idx)),
        dtype=np.float32,
    )


# ======================
# IMPLICIT ALS МОДЕЛЬ
# ======================
def train_implicit_model_with_params(ratings: pd.DataFrame, books: pd.DataFrame,
                                     factors: int, alpha: float, reg: float,
                                     iterations: int):
    print(f"=== Обучение ALS: factors={factors}, alpha={alpha}, reg={reg}, iters={iterations} ===")

    implicit_events = ratings.copy()
    implicit_events["weight"] = 1.0
    implicit_events.loc[implicit_events["Book-Rating"] >= 6, "weight"] = 2.0
    implicit_events.loc[implicit_events["Book-Rating"] >= 9, "weight"] = 3.0
    # implicit_events["weight"] = 0.5
    # implicit_events.loc[implicit_events["Book-Rating"] > 0, "weight"] = 1.0
    # implicit_events.loc[implicit_events["Book-Rating"] >= 6, "weight"] = 2.0
    # implicit_events.loc[implicit_events["Book-Rating"] >= 9, "weight"] = 3.0

    user_ids = pd.Categorical(implicit_events["User-ID"])
    item_ids = pd.Categorical(implicit_events["ISBN"])

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids.categories)}
    item_to_idx = {isbn: idx for idx, isbn in enumerate(item_ids.categories)}
    idx_to_item = {idx: isbn for isbn, idx in item_to_idx.items()}

    interaction_matrix = csr_matrix(
        (
            implicit_events["weight"].values.astype(np.float32),
            (user_ids.codes, item_ids.codes),
        ),
        dtype=np.float32,
    )
    print("n_items matrix:", interaction_matrix.shape[1])
    print("n_items mapping:", len(item_to_idx))
    assert interaction_matrix.shape[1] == len(item_to_idx)
    binary_matrix = create_binary_interactions(implicit_events, user_to_idx, item_to_idx)

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        random_state=CONFIG["model_params"]["random_state"],
        alpha=alpha,
        use_gpu=False,
    )
    model.fit(interaction_matrix.T)

    model_data = {
        "model": model,
        "books": books,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "interaction_matrix": interaction_matrix,
        "train_interactions_binary": binary_matrix,
        "test_interactions_binary": None,
        "user_categories": user_ids.categories,
        "item_categories": item_ids.categories,
    }

    return model_data
def hyperparameter_search_implicit(ratings, books, k_eval=20):
    factors_grid = [32, 64]
    alpha_grid = [5.0, 10.0, 20.0]
    reg_grid = [0.01, 0.05]
    iterations = 30

    best_score = -1.0
    best_params = None
    best_model_data = None
    results = []

    for f in factors_grid:
        for a in alpha_grid:
            for r in reg_grid:
                model_data = train_implicit_model_with_params(
                    ratings, books, factors=f, alpha=a, reg=r, iterations=iterations
                )
                metrics = evaluate_implicit_model(model_data, k=k_eval)
                score = metrics["ndcg_at_k"]  # целевая метрика

                print(
                    f"f={f}, alpha={a}, reg={r} -> "
                    f"Prec@{k_eval}={metrics['precision_at_k']:.4f}, "
                    f"Rec@{k_eval}={metrics['recall_at_k']:.4f}, "
                    f"NDCG@{k_eval}={metrics['ndcg_at_k']:.4f}"
                )

                results.append(
                    {
                        "factors": f,
                        "alpha": a,
                        "reg": r,
                        "precision": metrics["precision_at_k"],
                        "recall": metrics["recall_at_k"],
                        "ndcg": metrics["ndcg_at_k"],
                    }
                )

                if score > best_score:
                    best_score = score
                    best_params = (f, a, r)
                    best_model_data = model_data

    print("=== Лучшие гиперпараметры ALS ===")
    print(
        f"factors={best_params[0]}, alpha={best_params[1]}, reg={best_params[2]}, "
        f"best NDCG@{k_eval}={best_score:.4f}"
    )

    return best_model_data, results
def train_implicit_model(ratings: pd.DataFrame, books: pd.DataFrame):
    print("Подготовка данных для Implicit ALS...")

    # Используем ВСЕ взаимодействия (0 = прочитал, >0 = оценил),
    # но даём разный вес
    implicit_events = ratings.copy()

    implicit_events["weight"] = 1.0  # базовый вес для "прочитал"

    # усиливаем положительные явные оценки
    implicit_events.loc[implicit_events["Book-Rating"] >= 7, "weight"] = 2.0
    implicit_events.loc[implicit_events["Book-Rating"] >= 9, "weight"] = 3.0

    # категориальные индексы
    user_ids = pd.Categorical(implicit_events["User-ID"])
    item_ids = pd.Categorical(implicit_events["ISBN"])

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids.categories)}
    item_to_idx = {isbn: idx for idx, isbn in enumerate(item_ids.categories)}
    idx_to_item = {idx: isbn for isbn, idx in item_to_idx.items()}

    interaction_matrix = csr_matrix(
        (
            implicit_events["weight"].values.astype(np.float32),
            (user_ids.codes, item_ids.codes),
        ),
        dtype=np.float32,
    )

    print(
        f"Implicit матрица взаимодействий: {interaction_matrix.shape}, "
        f"ненулевых: {interaction_matrix.nnz}"
    )

    # бинарная матрица для метрик (пользователь ⇔ факт взаимодействия)
    binary_matrix = create_binary_interactions(implicit_events, user_to_idx, item_to_idx)

    model = implicit.als.AlternatingLeastSquares(
        factors=CONFIG["model_params"]["implicit_factors"],
        regularization=CONFIG["model_params"]["implicit_regularization"],
        iterations=CONFIG["model_params"]["implicit_iterations"],
        random_state=CONFIG["model_params"]["random_state"],
        alpha=CONFIG["model_params"]["implicit_alpha"],
        use_gpu=False,
    )

    print("Обучение Implicit ALS...")
    model.fit(interaction_matrix.T)

    return {
        "model": model,
        "books": books,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "interaction_matrix": interaction_matrix,
        "train_interactions_binary": binary_matrix,
        "test_interactions_binary": None,  # можно добавить разбиение позже
        "user_categories": user_ids.categories,
        "item_categories": item_ids.categories,
    }


# ======================
# ОЦЕНКА IMPLICIT
# ======================


def manual_ranking_metrics(model, model_data, train_interactions, test_interactions, k=10):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    for user_idx in range(train_interactions.shape[0]):
        test_items = set(test_interactions[user_idx].indices)
        if not test_items:
            continue

        try:
            rec_ids, _ = model.recommend(
                user_idx,
                train_interactions[user_idx],
                N=k,
                filter_already_liked_items=False,
            )
            recommended_items = list(rec_ids)
            hits = len(set(recommended_items) & test_items)

            precision = hits / min(k, len(recommended_items)) if recommended_items else 0.0
            recall = hits / len(test_items) if test_items else 0.0

            if hits > 0:
                relevance = np.array(
                    [1 if item in test_items else 0 for item in recommended_items],
                    dtype=np.float32,
                )
                discounts = np.log2(np.arange(2, len(relevance) + 2))
                dcg = np.sum(relevance / discounts)
                ideal_relevance = np.sort(relevance)[::-1]
                idcg = np.sum(ideal_relevance / discounts)
                ndcg = dcg / idcg if idcg > 0 else 0.0
            else:
                ndcg = 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
        except Exception:
            continue

    return {
        "precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
    }


def evaluate_implicit_model(model_data, k=10):
    print(f"Оценка модели (K={k})...")
    model = model_data["model"]
    train_bin = model_data["train_interactions_binary"]
    test_bin = model_data["test_interactions_binary"] or train_bin

    metrics = manual_ranking_metrics(model, model_data, train_bin, test_bin, k)
    return {
        "precision_at_k": metrics["precision"],
        "recall_at_k": metrics["recall"],
        "ndcg_at_k": metrics["ndcg"],
    }


# ======================
# СОХРАНЕНИЕ
# ======================


def save_models(content_model, implicit_model_data, implicit_metrics):
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../metrics", exist_ok=True)

    # content model
    with open("../models/content_model.pkl", "wb") as f:
        pickle.dump(content_model, f)

    # implicit model в формате, который ждёт app
    implicit_to_save = {
        "model": implicit_model_data["model"],
        "books": implicit_model_data["books"],
        "user_id_to_idx": implicit_model_data["user_to_idx"],
        "idx_to_item_id": implicit_model_data["idx_to_item"],
        "interaction_matrix": implicit_model_data["interaction_matrix"],
    }
    with open("../models/implicit_als_model.pkl", "wb") as f:
        pickle.dump(implicit_to_save, f)

    metrics_full = {
        "timestamp": datetime.now().isoformat(),
        "implicit": implicit_metrics,
    }
    with open("../metrics/model_metrics.json", "w") as f:
        json.dump(metrics_full, f, indent=2)

    print("✅ Модели и метрики сохранены")


# ======================
# MAIN
# ======================


print("=== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===")
books, ratings = prepare_implicit_data()

print("=== ОБУЧЕНИЕ CONTENT-BASED МОДЕЛИ ===")
content_model = build_content_model(books)

# print("=== ОБУЧЕНИЕ IMPLICIT ALS ===")
# implicit_model_data = train_implicit_model(ratings, books)
print("=== ОБУЧЕНИЕ IMPLICIT ALS ===")
# implicit_model_data = train_implicit_model(ratings, books)
implicit_model_data, search_results = hyperparameter_search_implicit(
    ratings, books, k_eval=20
)
print("=== ОЦЕНКА IMPLICIT ALS ===")
k = CONFIG["model_params"]["eval_k"]
implicit_metrics = evaluate_implicit_model(implicit_model_data, k=k)

print("=== СОХРАНЕНИЕ ===")
save_models(content_model, implicit_model_data, implicit_metrics)

m = implicit_metrics
print("✅ Готово!")
print(
    f"Implicit → Precision@{k}: {m['precision_at_k']:.4f}, "
    f"Recall@{k}: {m['recall_at_k']:.4f}, "
    f"NDCG@{k}: {m['ndcg_at_k']:.4f}"
)

