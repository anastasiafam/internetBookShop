import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ======================
# КОНФИГ
# ======================

TOP_N_POPULAR = 8
TOP_N_SIMILAR = 10
TOP_N_PERSONAL = 6

# ======================
# ЗАГРУЗКА МОДЕЛЕЙ / ДАННЫХ
# ======================

@st.cache_resource
def load_models():
    try:
        with open("../models/content_model.pkl", "rb") as f:
            content_data = pickle.load(f)
        with open("../models/implicit_als_model.pkl", "rb") as f:
            implicit_data = pickle.load(f)
        with open("../models/surprise_svd_model.pkl", "rb") as f:
            surprise_data = pickle.load(f)
        return content_data, implicit_data, surprise_data
    except FileNotFoundError as e:
        st.error(f"Файлы моделей не найдены: {e}")
        st.stop()

@st.cache_resource
def load_ratings():
    usecols = ["User-ID", "ISBN", "Book-Rating"]
    return pd.read_csv("../data/Ratings.csv", low_memory=False, usecols=usecols)

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

# def safe_image(url: str) -> bool:
#     return isinstance(url, str)
from PIL import Image
from io import BytesIO
import requests
def display_book_image(image_url, width=110):
    """
    Отображает изображение книги или placeholder
    """
    img = validate_and_load_image(image_url)
    
    if img:
        st.image(img)
    else:
        # Красивый placeholder вместо битого изображения
        st.markdown(f"""
            <div style='width: {300}px; height: {int(400)}px; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 8px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <div style='text-align: center; color: white;'>
                    <div style='font-size: 2.5rem; margin-bottom: 0.3rem;'>📚</div>
                    <div style='font-size: 0.7rem; opacity: 0.8;'>No Image</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
def validate_and_load_image(url, target_size=(300, 400)):
    """
    Проверяет URL и загружает изображение, валидируя его содержимое и изменяя размер.
    
    Args:
        url (str): URL изображения.
        target_size (tuple): Целевой размер (ширина, высота). По умолчанию (150, 200).
    
    Returns:
        PIL.Image.Image или None: изображение нужного размера или None при ошибке.
    """
    if not url or not isinstance(url, str):
        return None
    
    if not url.startswith(('http://', 'https://')):
        return None
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.amazon.com/'
        }
        
        # Загружаем изображение
        response = requests.get(url, timeout=5, headers=headers, allow_redirects=True)
        
        if response.status_code != 200:
            return None
        
        if len(response.content) < 100:  # Слишком маленький файл
            return None
        
        # Открываем изображение
        img = Image.open(BytesIO(response.content))
        
        # Проверки на валидность
        width, height = img.size
        if width < 10 or height < 10:
            return None
        if width == 1 and height == 1:
            return None
        
        # Конвертация в RGB (если нужно)
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        # ✅ Изменение размера до target_size с сохранением качества
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        return img
        
    except (requests.RequestException, Image.UnidentifiedImageError, Exception) as e:
        # print(f"Ошибка загрузки {url}: {e}")  # для отладки
        return None

def get_popular_books(books_df: pd.DataFrame, top_n: int = TOP_N_POPULAR):
    df = books_df.copy()
    # чуть ограничим по рейтингу, чтобы убрать мусор
    df = df[df["avg_rating"] >= 6]
    return df.nlargest(top_n, "rating_count").to_dict("records")

from rapidfuzz import fuzz
import string
def titles_are_similar(title1: str, title2: str, threshold: int = 90) -> bool:
    if not isinstance(title1, str) or not isinstance(title2, str):
        return False
   
    # Используем ratio — простое, но надёжное сравнение
    score = fuzz.token_set_ratio(title1.lower().strip(), title2.lower().strip())
    print(f'{title1}, {title2}, {score}')
    return score >= threshold
def normalize_title_for_dedup(title: str) -> str:
    if not isinstance(title, str):
        return ""
    return title.strip().lower().rstrip(string.punctuation)
def is_similar_to_any(title: str, seen_titles: list, threshold: int = 90) -> bool:
    if not isinstance(title, str):
        return False
    title_clean = title.lower().strip()
    for seen in seen_titles:
        if not isinstance(seen, str):
            continue
        if fuzz.partial_ratio(title_clean, seen.lower().strip()) >= threshold:
            return True
    return False
def recommend_content_based_hybrid(
    tfidf_reduced,
    isbn_to_idx,
    idx_to_isbn,
    books,
    isbn,
    top_n: int = TOP_N_SIMILAR,
):
    if isbn not in isbn_to_idx:
        return []

    # Получаем название целевой книги
    target_book_row = books[books["ISBN"] == isbn]
    if target_book_row.empty:
        return []
    target_title = target_book_row.iloc[0]["Book-Title"]

    idx = isbn_to_idx[isbn]
    target_vec = tfidf_reduced[idx].reshape(1, -1)
    sim_scores = cosine_similarity(target_vec, tfidf_reduced).flatten()
    sim_scores[idx] = -1  # Исключаем саму книгу по ISBN

    top_indices = sim_scores.argsort()[-top_n * 10 :][::-1]

    recs = []
    for i in top_indices:
        rec_isbn = idx_to_isbn.get(i)
        if not rec_isbn:
            continue
        row = books[books["ISBN"] == rec_isbn]
        if row.empty:
            continue
        book = row.iloc[0].to_dict()

        # Исключаем книги с таким же названием
        if titles_are_similar(target_title, book.get("Book-Title"), threshold=90):
            continue
        if sim_scores[i]>=0.96:
            continue
        book["similarity_score"] = float(sim_scores[i])
        recs.append(book)

    seen_titles = set()
    unique_recs = []
    for book in recs:
        norm_title = normalize_title_for_dedup(book.get("Book-Title", ""))
        if is_similar_to_any(norm_title, seen_titles, threshold=90):
            continue
        seen_titles.add(norm_title)
        unique_recs.append(book)

    # Сортировка по similarity_score + avg_rating
    unique_recs.sort(
        key=lambda x: (x.get("similarity_score", 0.0), x.get("avg_rating", 0.0)),
        reverse=True,
    )

    return unique_recs[:top_n]

def recommend_implicit(
    model,
    user_id: int,
    books_df: pd.DataFrame,
    user_id_to_idx: dict,
    idx_to_item_id: dict,
    interaction_matrix,
    all_ratings: pd.DataFrame,
    top_n: int = TOP_N_PERSONAL,
):
    # новый пользователь
    if user_id not in user_id_to_idx:
        print('New user')
        return get_popular_books(books_df, top_n)

    user_idx = user_id_to_idx[user_id]
    
    # Явно преобразовать в CSR формат и транспонировать
    if not isinstance(interaction_matrix, csr_matrix):
        interaction_matrix = csr_matrix(interaction_matrix)
    
    user_items = interaction_matrix.T[user_idx].tocsr()  # ← КЛЮЧЕВОЕ: .tocsr()
    
    print(f"[DEBUG] user_items type: {type(user_items)}")
    print(f"[DEBUG] user_items shape: {user_items.shape}")

    try:
        # Возвращает ДВА массива: (ids, scores)
        ids, scores = model.recommend(
            user_idx, 
            user_items,
            N=top_n * 3,
            filter_already_liked_items=True
        )
    except Exception as e:
        print(f"Error: {e}")
        return get_popular_books(books_df, top_n)
    
    seen_isbns = set(all_ratings[all_ratings['User-ID'] == user_id]['ISBN'])
    
    filtered = []
    # Используем zip() для итерации по двум массивам
    for item_idx, score in zip(ids, scores):
        isbn = idx_to_item_id.get(item_idx)
        if not isbn or isbn in seen_isbns:
            continue
        filtered.append((isbn, float(score)))
        if len(filtered) >= top_n:
            break
    
    if not filtered:
        return get_popular_books(books_df, top_n)
    
    isbns = [isbn for isbn, _ in filtered]
    df = books_df[books_df['ISBN'].isin(isbns)].copy()
    score_map = {isbn: s for isbn, s in filtered}
    df['implicit_score'] = df['ISBN'].map(score_map)
    df = df.sort_values('implicit_score', ascending=False)
    
    return df.to_dict('records')

def recommend_surprise(model, user_id, books_df, all_isbns, user_ratings_df, top_n=6):
    rated_isbns = set(user_ratings_df[user_ratings_df["User-ID"] == user_id]["ISBN"])
    if not rated_isbns:
        return get_popular_books(books_df, top_n)

    predictions = []
    for isbn in all_isbns:
        if isbn not in rated_isbns:
            try:
                pred = model.predict(str(user_id), str(isbn))
                predictions.append((isbn, pred.est))
            except Exception:
                continue

    if not predictions:
        return get_popular_books(books_df, top_n)

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_isbns = [isbn for isbn, _ in predictions[:top_n]]
    isbn_to_pred = dict(predictions[:top_n])

    recs = books_df[books_df["ISBN"].isin(isbn_to_pred.keys())].copy()
    recs["predicted_rating"] = recs["ISBN"].map(isbn_to_pred)
    recs = recs.sort_values("predicted_rating", ascending=False)
    return recs.head(top_n).to_dict("records")

@st.cache_data(show_spinner=False)
def cached_content_recs(isbn: str, top_n: int):
    return recommend_content_based_hybrid(
        tfidf_reduced, isbn_to_idx, idx_to_isbn, books, isbn, top_n
    )

@st.cache_data(show_spinner=False)
def cached_implicit_recs(user_id: int, top_n: int):
    return recommend_implicit(
        implicit_model,
        user_id,
        books,
        user_id_to_idx,
        idx_to_item_id,
        interaction_matrix,
        ratings,
        top_n,
    )


st.set_page_config(page_title="📚 Книжный Магазин", layout="wide")

if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_isbn" not in st.session_state:
    st.session_state.selected_isbn = None

content_data, implicit_data, surprise_data = load_models()
ratings = load_ratings()

# content-based
tfidf_reduced = content_data["tfidf_reduced"]
isbn_to_idx = content_data["isbn_to_idx"]
idx_to_isbn = content_data["idx_to_isbn"]
books = content_data["books"]

# implicit
implicit_model = implicit_data["model"]
implicit_books = implicit_data["books"]  # на всякий
user_id_to_idx = implicit_data["user_id_to_idx"]
idx_to_item_id = implicit_data["idx_to_item_id"]
interaction_matrix = implicit_data["interaction_matrix"]

surprise_model = surprise_data["model"]
surprise_books = surprise_data["books"]
all_isbns = surprise_books["ISBN"].tolist()

MIN_EXPLICIT_FOR_ACTIVE = 30  # или другое число

def is_active_user(user_id: int, ratings_df: pd.DataFrame) -> bool:
    user_explicit = ratings_df[
        (ratings_df["User-ID"] == user_id) & (ratings_df["Book-Rating"] > 0)
    ]
    return len(user_explicit) >= MIN_EXPLICIT_FOR_ACTIVE

# ============= НАСТРОЙКА СТИЛЕЙ =============
def load_custom_css():
    st.markdown("""
        <style>
        /* Основной фон */
        .main {
            background-color: #f8f9fa;
        }
        /* Центрирование контейнера с колонками */
       
        /* Стилизация ВСЕЙ колонки как карточки */
        [data-testid="column"] > div:first-child {
            background: white;
            padding: 5rem;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        /* Изображения внутри колонок с фиксированной высотой */
        [data-testid="column"] img {
            height: 160px !important;
            width: 110px !important;
            object-fit: cover;
            margin: 0.5rem auto;
            display: block;
            border-radius: 8px;
        }
        
        /* Стиль для placeholder книг */
        .book-placeholder {
            width: 110px;
            height: 160px;
            min-height: 160px;
            max-height: 160px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0.5rem auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .book-placeholder-content {
            text-align: center;
            color: white;
        }
        
        .book-placeholder-icon {
            font-size: 2.5rem;
            margin-bottom: 0.3rem;
        }
        
        .book-placeholder-text {
            font-size: 0.7rem;
            opacity: 0.8;
        }
        
        /* Заголовки книг */
        .book-title {
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0.5rem 0 0.3rem 0;
            line-height: 1.3;
            min-height: 2.4rem;
            color: #1a1a1a;
            text-align: left;
        }
        
        /* Авторы */
        .book-author {
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 0.5rem;
            min-height: 1.2rem;
            text-align: left;
        }
        
        /* Рейтинг */
        .rating-badge {
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            color: #1a1a1a;
            padding: 0.25rem 0.7rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.75rem;
            display: inline-block;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(255, 215, 0, 0.3);
        }
        
        /* Заголовки секций */
        .section-header {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1a1a1a;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #ff6b6b;
        }
        
        /* Поисковая панель */
        .search-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
        }
        
        /* Кнопки */
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            transform: scale(1.02);
        }
        
        /* Отступы между колонками */
        [data-testid="column"] {
            padding: 0 0.5rem !important;
        }
        
        /* Убрать лишние отступы у элементов */
        [data-testid="column"] .element-container {
            margin: 0;
        }
        
        </style>
    """, unsafe_allow_html=True)
load_custom_css()

# ============= HOME VIEW =============
if st.session_state.view == "home":
    # Хедер
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
            <h1 style='color: white; margin: 0; font-size: 2rem;'>📚 Книжный Магазин</h1>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1rem;'>
                Откройте мир книг с персональными рекомендациями
            </p>
        </div>
    """, unsafe_allow_html=True)
    # Поиск
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.markdown("### 🔍 Поиск книги")
        
        query = st.text_input(
            "Поиск",
            placeholder="Введите название книги...",
            label_visibility="collapsed"
        )
        
        matched_titles = []
        if query:
            mask = books['Book-Title'].str.contains(query.lower(), na=False)
            matched = books[mask]['Book-Title'].dropna().unique()[:50]
            matched_titles = sorted(matched)

        if matched_titles:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_title = st.selectbox(
                    "Результаты",
                    options=["Выберите книгу..."] + matched_titles,
                    key="search_results",
                    label_visibility="collapsed"
                )
            with col2:
                if selected_title and selected_title != "Выберите книгу...":
                    if st.button("Открыть", type="primary", use_container_width=True):
                        row = books[books["Book-Title"] == selected_title].iloc[0]
                        st.session_state.selected_isbn = row["ISBN"]
                        st.session_state.view = "book_detail"
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Популярные книги
    st.markdown('<h2 class="section-header">🔥 Популярные книги</h2>', unsafe_allow_html=True)
    
    popular_books = get_popular_books(books, top_n=TOP_N_POPULAR)
    
    # 4 колонки
    for row_start in range(0, len(popular_books), 4):
        cols = st.columns(4, gap="medium")
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(popular_books):
                book = popular_books[idx]
                with col:
                    # Изображение
                    # if safe_image(book.get("Image-URL-L")):
                    #     st.image(book["Image-URL-L"], width=300)
                    display_book_image(book["Image-URL-L"])
                    # Название
                    title = book['Book-Title'].title()
                    st.markdown(
                        f'<div class="book-title">{title[:50]}{"..." if len(title) > 50 else ""}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Автор
                    author = book["Book-Author"].title() if isinstance(book["Book-Author"], str) else "Неизвестный автор"
                    st.markdown(
                        f'<div class="book-author">{author[:30]}{"..." if len(author) > 30 else ""}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Рейтинг
                    st.markdown(
                        f'<div style="text-align: left;"><span class="rating-badge">⭐ {book["avg_rating"]:.1f} ({int(book["rating_count"])})</span></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Кнопка
                    if st.button("Подробнее", key=f"home_{idx}", use_container_width=True, type="secondary"):
                        st.session_state.selected_isbn = book["ISBN"]
                        st.session_state.view = "book_detail"
                        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # CTA для персональных рекомендаций
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 1.5rem; border-radius: 10px; text-align: left;'>
                <h4 style='color: white; margin-bottom: 0.5rem;'>🎯 Персональные рекомендации</h4>
                <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>
                    Подберем книги специально для вас
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Начать подбор", use_container_width=True, type="primary"):
            st.session_state.view = "personal"
            st.rerun()

# ============= BOOK DETAIL VIEW =============
elif st.session_state.view == "book_detail":
    isbn = st.session_state.selected_isbn
    if isbn is None:
        st.session_state.view = "home"
        st.rerun()
    
    book_row = books[books["ISBN"] == isbn]
    if book_row.empty:
        st.warning("📚 Книга не найдена.")
    else:
        book_row = book_row.iloc[0]
        
        # Кнопка "Назад"
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button("← Назад"):
                st.session_state.view = "home"
                st.rerun()
        
        st.markdown("")
        
        # Информация о книге
        st.markdown("""
            <div style='background: white; padding: 2rem; border-radius: 12px; 
                        box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 2rem;'>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            display_book_image(book_row.get("Image-URL-M"))
        
        with col2:
            st.title(book_row["Book-Title"].title())
            author = book_row["Book-Author"].title() if isinstance(book_row["Book-Author"], str) else "Неизвестный автор"
            st.markdown(f"### ✍️ {author}")
            
            st.markdown("")
            
            # Метрики
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Рейтинг", f"{book_row['avg_rating']:.1f} ⭐")
            with metric_col2:
                st.metric("Оценок", int(book_row["rating_count"]))
            with metric_col3:
                if not pd.isna(book_row["Year-Of-Publication"]):
                    st.metric("Год", int(book_row["Year-Of-Publication"]))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Похожие книги
        st.markdown('<h2 class="section-header">🎯 Похожие книги</h2>', unsafe_allow_html=True)
        
        with st.spinner("Подбираем похожие книги..."):
            similar_books = cached_content_recs(isbn, TOP_N_SIMILAR)
            
            if similar_books:
                cols = st.columns(4, gap="medium")
                for i, rec in enumerate(similar_books):
                    col = cols[i % 4]
                    with col:
                        display_book_image(rec.get("Image-URL-L"))
                        title = rec['Book-Title'].title()
                        st.markdown(
                            f'<div class="book-title">{title[:45]}{"..." if len(title) > 45 else ""}</div>',
                            unsafe_allow_html=True
                        )
                        
                        author = rec["Book-Author"].title() if isinstance(rec["Book-Author"], str) else "—"
                        st.markdown(
                            f'<div class="book-author">{author[:30]}</div>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(
                            f'<div style="text-align: left;"><span class="rating-badge">⭐ {rec["avg_rating"]:.1f}</span></div>',
                            unsafe_allow_html=True
                        )
                        if st.button("Подробнее", key=f"rec_{i}", use_container_width=True, type="secondary"):
                            st.session_state.selected_isbn = rec["ISBN"]
                            st.rerun()
            else:
                st.info("📚 Похожие книги не найдены.")

# ============= PERSONAL RECOMMENDATIONS VIEW =============
elif st.session_state.view == "personal":
    # Хедер
    st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 2rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
            <h1 style='color: white; margin: 0; font-size: 2rem;'>🎯 Персональные рекомендации</h1>
            <p style='color: rgba(255,255,255,0.95); margin-top: 0.5rem;'>
                Получите подборку книг на основе ваших предпочтений
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Ввод User ID
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.number_input("Введите ваш User-ID:", min_value=1, max_value=1_000_000, value=11676, step=1)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Сгенерировать", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    user_ratings = ratings[ratings["User-ID"] == user_id]
    
    if not user_ratings.empty:
        st.success(f"✅ Найдено {len(user_ratings)} ваших оценок")
        
        st.markdown('<h3 class="section-header">📖 Ваши недавние книги</h3>', unsafe_allow_html=True)
        
        cols = st.columns(5, gap="medium")
        for i, (_, row) in enumerate(user_ratings.head(5).iterrows()):
            br = books[books["ISBN"] == row["ISBN"]].iloc[0]
            if br.empty:
                continue
            book = br.iloc[0]
            col = cols[i % 5]
            with col:
                rating_display = "📖 Прочитано" if row["Book-Rating"] == 0 else f"⭐ {int(row['Book-Rating'])}/10"
                st.markdown(f"{rating_display}")
                print(br.get("Image-URL-L"))
                display_book_image(br.get("Image-URL-L"))
             
    
    if generate_btn:
        with st.spinner("🔮 Анализируем ваши предпочтения..."):
            if is_active_user(user_id, ratings):
                recs = recommend_implicit(
                    implicit_model,
                    user_id,
                    books,
                    user_id_to_idx,
                    idx_to_item_id,
                    interaction_matrix,
                    ratings,
                    top_n=6,
                )
                source = "на основе похожих пользователей"
            else:
                user_ratings_explicit = ratings[
                    (ratings["User-ID"] == user_id) & (ratings["Book-Rating"] > 0)
                ]
                recs = recommend_surprise(
                    surprise_model,
                    user_id,
                    books,
                    all_isbns,
                    user_ratings_explicit,
                    top_n=6,
                )
                source = "на основе ваших оценок"
            
            if recs:
                st.markdown(f'<h2 class="section-header">📚 Специально для вас ({source})</h2>', unsafe_allow_html=True)
                
                # 4 колонки
                for row_start in range(0, len(recs), 4):
                    cols = st.columns(4, gap="medium")
                    for i, col in enumerate(cols):
                        idx = row_start + i
                        if idx < len(recs):
                            rec = recs[idx]
                            with col:
                                if pd.notna(rec.get("Image-URL-L")):
                                    display_book_image(rec["Image-URL-L"],width=300 )
                                
                                title = rec['Book-Title'].title()
                                st.markdown(
                                    f'<div class="book-title">{title[:45]}{"..." if len(title) > 45 else ""}</div>',
                                    unsafe_allow_html=True
                                )
                                
                                author = rec["Book-Author"].title() if pd.notna(rec["Book-Author"]) else "—"
                                st.markdown(
                                    f'<div class="book-author">{author[:30]}</div>',
                                    unsafe_allow_html=True
                                )
                                
                                if "predicted_rating" in rec:
                                    st.markdown(f"<div style='text-align: left; margin: 0.5rem 0;'><strong>🎯 Прогноз: {rec['predicted_rating']:.1f}/10</strong></div>", unsafe_allow_html=True)
                                
                                st.markdown(
                                    f'<div style="text-align: left;"><span class="rating-badge">⭐ {rec.get("avg_rating", 0):.1f}</span></div>',
                                    unsafe_allow_html=True
                                )
                                
                                if st.button("Подробнее", key=f"pers_{idx}", use_container_width=True, type="secondary"):
                                    st.session_state.selected_isbn = rec["ISBN"]
                                    st.session_state.view = "book_detail"
                                    st.rerun()
            else:
                st.warning("⚠️ Не удалось сгенерировать рекомендации. Попробуйте другой ID.")
    
    st.markdown("")
    if st.button("← Назад на главную", key="back_from_personal"):
        st.session_state.view = "home"
        st.rerun()