# train_surprise.py
import pandas as pd
import numpy as np
import pickle
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# ======================
# Конфигурация
# ======================
DATA_PATH = "../data/Ratings.csv"
BOOKS_PATH = "../data/Book.csv"
MODEL_SAVE_PATH = "../models/surprise_svd_model.pkl"
METRICS_SAVE_PATH = "../metrics/surprise_metrics.json"

# Пороги фильтрации (опционально, как в твоих скриптах)
MIN_USER_RATINGS = 5
MIN_BOOK_RATINGS = 5

# ======================
# Загрузка и фильтрация данных
# ======================
def load_and_filter_data():
    ratings = pd.read_csv(DATA_PATH)
    books = pd.read_csv(BOOKS_PATH)

    # Удаляем implicit (рейтинги = 0)
    ratings = ratings[ratings["Book-Rating"] > 0]
    print(f"После удаления 0: {len(ratings)} оценок")

    # Фильтрация пользователей и книг
    user_counts = ratings['User-ID'].value_counts()
    item_counts = ratings['ISBN'].value_counts()

    ratings = ratings[
        ratings['User-ID'].isin(user_counts[user_counts >= MIN_USER_RATINGS].index) &
        ratings['ISBN'].isin(item_counts[item_counts >= MIN_BOOK_RATINGS].index)
    ]
    print(f"После фильтрации: {len(ratings)} оценок")

    # Оставляем только книги, которые остались после фильтрации
    books = books[books['ISBN'].isin(ratings['ISBN'])]

    return ratings, books

# ======================
# Обучение модели Surprise (SVD)
# ======================
def train_surprise_model(ratings):
    # Reader для шкалы 1–10
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    # Разделение
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Модель SVD (Matrix Factorization)
    model = SVD(n_factors=150, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
    print("Обучение Surprise SVD...")
    model.fit(trainset)

    # Оценка
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    return model, trainset, testset, books, {'rmse': rmse, 'mae': mae}

# ======================
# Сохранение модели и данных
# ======================
def save_model_and_books(model, books, metrics):
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../metrics", exist_ok=True)

    # Сохраняем модель и книги
    with open("../models/surprise_svd_model.pkl", "wb") as f:
        pickle.dump({
            'model': model,
            'books': books
        }, f)

    # Сохраняем метрики
    import json
    with open("../metrics/surprise_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Модель и метрики сохранены")

# ======================
# Основной запуск
# ======================
if __name__ == "__main__":
    ratings, books = load_and_filter_data()
    model, trainset, testset, books, metrics = train_surprise_model(ratings)
    save_model_and_books(model, books, metrics)