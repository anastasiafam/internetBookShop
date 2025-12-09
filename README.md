README — Система рекомендаций книг
==================================

> Веб‑приложение на Streamlit для персональных рекомендаций книг на основе нескольких моделей: content‑based, implicit ALS и Surprise SVD.app.py+2​

1\. Описание проекта
--------------------

Проект представляет собой рекомендательную систему, которая использует данные о книгах и пользовательских оценках для формирования следующих типов рекомендаций:

*   Популярные книги (по статистике рейтингов).app.py​
    
*   Похожие книги (content‑based модель на основе описательных признаков книги)
    
*   Персональные рекомендации для пользователя (implicit ALS и Surprise SVD)

Веб‑интерфейс реализован с помощью Streamlit и позволяет интерактивно работать с моделью: выбирать пользователя, искать книги и просматривать списки рекомендаций.app.py​

2\. Структура проекта
---------------------
project/  
├─ app.py                   # Веб-приложение Streamlit
├─ Dockerfile             # Описание контейнера
├─ requirements.txt        # Зависимости Python
├─ scripts/  
│  ├─ train_implicit.py    # Обучение content + implicit ALS
│  └─ train_surprise.py          # Обучение Surprise SVD
├─ data/  
│  ├─ Users.csv  
│  ├─ Book.csv  
│  └─ Ratings.csv  
├─ models/  
│  ├─ content_model.pkl  
│  ├─ implicit_als_model.pkl  
│  └─ surprise_svd_model.pkl  
└─ metrics/  
   ├─ model_metrics.json          # Метрики ALS
   └─ surprise_metrics.json       # Метрики SVD


3\. Функциональные возможности
------------------------------

*   Интерфейс для просмотра:
    
    *   списка популярных книг;app.py​
        
    *   рекомендаций книг, похожих на выбранную книгу (по содержанию);
        
    *   персональных рекомендаций для конкретного пользователя.
        
*   Загрузка предобученных моделей:
    
    *   content\_model.pkl — TF‑IDF + SVD + признаки книг для поиска похожих произведений;
        
    *   implicit\_als\_model.pkl — модель implicit ALS и матрица взаимодействий пользователь–книга;
        
    *   surprise\_svd\_model.pkl — модель SVD (Surprise) для коллаборативных рекомендаций.
        

4\. Требования к среде
----------------------

4.1. Программные требования
---------------------------

*   Python 3.10–3.11.
    
*   Установленные зависимости из requirements.txt, включая:train\_surprise.py+2​
    
    *   streamlit — веб‑интерфейс;
        
    *   pandas, numpy — работа с данными;
        
    *   scikit-learn — TF‑IDF, SVD, метрики;
        
    *   scipy — разреженные матрицы (csr\_matrix);
        
    *   implicit — реализация ALS для неявных откликов;train\_implicit-kopiia.py​
        
    *   scikit-surprise — реализация SVD (Surprise);train\_surprise.py​
        
    *   Pillow, requests — загрузка и отображение обложек книг.app.py​
        

4.2. Аппаратные требования
--------------------------

*   RAM: от 2 ГБ (рекомендуется больше при обучении моделей).
    
*   CPU: 2+ ядра (ALS и TF‑IDF активно используют CPU).train\_implicit-kopiia.py​
    
*   Свободное место на диске для CSV‑файлов, моделей и временных файлов (десятки–сотни МБ в зависимости от размера датасета).
    

5\. Установка и запуск без Docker
---------------------------------

5.1. Получение исходного кода
-----------------------------

git clone https://github.com/anastasiafam/internetBookShop.git 
cd book-recs-app

5.2. Установка зависимостей
---------------------------
```
python -m venv venv  
source venv/bin/activate        
pip install -r requirements.txt   `
```

5.3. Подготовка данных и моделей
--------------------------------

*   Скопировать исходные файлы в папку data/:
    
    *   Users.csv
        
    *   Book.csv
        
    *   Ratings.csv
        
*   Обучить модели:

    *   content\_model.pkl
        
    *   implicit\_als\_model.pkl
        
    *   surprise\_svd\_model.pkl
        

5.4. Запуск приложения
----------------------

```
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

После запуска интерфейс будет доступен по адресу http://localhost:8501

6\. Установка и запуск через Docker
-----------------------------------

6.1. Сборка образа
------------------
```
docker build -t book-recs-app .
``` 

6.2. Запуск контейнера
----------------------
```
docker run -d \
  -p 8501:8501 \
  --name book-recs \
  book-recs-app
```
Приложение станет доступным по адресу http://localhost:8501

7\. Обучение и переобучение моделей
-----------------------------------

7.1. Обучение content‑based + implicit ALS
------------------------------------------

1.  **Подготовка данных** (prepare\_implicit\_data):
    
    *   загрузка Ratings.csv и Book.csv;
        
    *   очистка текстовых полей книг (Book-Title, Book-Author, Publisher), приведение к нижнему регистру, удаление пробелов;
        
    *   расчёт среднего рейтинга и числа оценок по каждой книге (явные оценки > 0);
        
    *   фильтрация пользователей по минимальному числу явных оценок (например, ≥ 30) и по общему числу взаимодействий (min\_user\_interactions);
        
    *   фильтрация книг по min\_book\_interactions;
        
    *   формирование признака десятилетия публикации (publication\_decade).
        
2.  **Content‑based модель** (build\_content\_model):
    
    *   формирование текстового поля content (автор, название, издатель, десятилетие и статистика рейтингов);
        
    *   вычисление TF‑IDF (до 20 000 признаков, n‑граммы 1–2, английский стоп‑лист);
        
    *   понижение размерности с помощью TruncatedSVD до content\_n\_components компонентов;
        
    *   нормализация эмбеддингов и построение словарей isbn\_to\_idx / idx\_to\_isbn;
        
    *   сохранение всего в структуру content\_model.train\_implicit-kopiia.py​
        
3.  **Implicit ALS** (train\_implicit\_model и/или hyperparameter\_search\_implicit):
    
    *   построение матрицы взаимодействий пользователь–книга (csr\_matrix) с весами, зависящими от оценки (Book-Rating);
        
    *   обучение AlternatingLeastSquares с параметрами factors, regularization, iterations, alpha;train\_implicit-kopiia.py​
        
    *   вычисление метрик ранжирования (Precision@K, Recall@K, NDCG@K) через функции manual\_ranking\_metrics и evaluate\_implicit\_model.train\_implicit-kopiia.py​
        
4.  **Сохранение модели** (save\_models):
    
    *   ../models/content\_model.pkl — контентная модель;
        
    *   ../models/implicit\_als\_model.pkl — словарь с ALS‑моделью и маппингами (user\_id\_to\_idx, idx\_to\_item\_id, interaction\_matrix);
        
    *   ../metrics/model\_metrics.json — метрики ALS.train\_implicit-kopiia.py​
        

Запуск обучения ALS
-------------------
```
cd scripts  python train_implicit-kopiia.py
```

7.2. Обучение Surprise SVD
--------------------------

1.  **Загрузка и фильтрация данных** (load\_and\_filter\_data):
    
    *   чтение Ratings.csv и Book.csv;
        
    *   удаление неявных оценок (Book-Rating == 0);
        
    *   фильтрация пользователей и книг по порогам MIN\_USER\_RATINGS и MIN\_BOOK\_RATINGS;
        
    *   фильтрация таблицы книг в соответствии с отфильтрованными ISBN.train\_surprise.py​
        
2.  **Обучение модели** (train\_surprise\_model):
    
    *   создание объекта Reader со шкалой оценок (1–10);
        
    *   построение Dataset из рейтингов;
        
    *   разбиение на обучающую и тестовую выборку (train/test split 80/20);
        
    *   обучение модели SVD с заданными гиперпараметрами;
        
    *   оценка качества по RMSE и MAE.train\_surprise.py​
        
3.  **Сохранение результатов** (save\_model\_and\_books):
    
    *   ../models/surprise\_svd\_model.pkl — словарь { 'model': model, 'books': books };
        
    *   ../metrics/surprise\_metrics.json — JSON с RMSE и MAE.train\_surprise.py​
        

Запуск обучения SVD
-------------------
```
cd scripts  python train_surprise.py   `
``` 

После выполнения будут обновлены файлы в ../models и ../metrics, которые сразу можно использовать в веб‑приложении.
