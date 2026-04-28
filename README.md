# Music RAG Assistant

Інтерактивний AI-помічник про музику з власною базою знань.
Проєкт поєднує RAG-підхід, векторний пошук та генерацію відповідей, а також підтримує завантаження PDF (включно з OCR для сканованих документів).

## Чому цей проєкт

Класичний чат-бот може вигадувати факти. У цьому проєкті відповідь формується на основі реального контексту з локальної бази знань:

- спочатку релевантні фрагменти знаходяться у ChromaDB
- далі LLM генерує відповідь тільки з цього контексту
- якщо даних недостатньо, бот чесно повідомляє про це

Результат: більш контрольована, пояснювана та практична AI-система.

## Ключові можливості

- Чат українською мовою з RAG-пошуком
- Індексація PDF у векторну базу
- Hybrid PDF extraction: native text + OCR fallback
- Підтримка артист-орієнтованих відповідей (MusicBrainz + Last.fm)
- Автоматичний воркер наповнення знань про артистів
- CLI для адміністрування та перевірки якості даних у ChromaDB
- React frontend з drag-and-drop завантаженням PDF

## Технології

Backend:

- Python
- Flask + Flask-CORS
- ChromaDB
- Ollama embeddings (модель nomic-embed-text)
- Groq API (LLM генерація)
- pypdf, pdf2image, pytesseract, pillow
- requests, python-dotenv

Frontend:

- React 19 + TypeScript
- Vite
- ESLint

Data sources:

- MusicBrainz API
- Last.fm API

## Архітектура

```text
User -> React UI -> Flask API
                -> /chat -> Embedding (Ollama) -> ChromaDB retrieval -> LLM (Groq)
                -> /upload -> PDF parsing/OCR -> chunking -> Embedding -> ChromaDB upsert

Worker -> MusicBrainz/Last.fm -> normalization/chunking -> Embedding -> ChromaDB
```

## Структура репозиторію

```text
backend/
  server.py          # Flask API: /chat, /upload
  music_worker.py    # batch-ingestion артистів з API
  db_admin.py        # CLI для аналізу/перевірки колекції
  init_database.py   # початкове seed-наповнення
  artists.txt        # список артистів для воркера/детекції

frontend-react/
  src/               # React UI
  vite.config.ts     # proxy /chat, /upload -> Flask
```

## Швидкий старт

### 1) Передумови

- Python 3.10+
- Node.js 20+
- Ollama з моделлю nomic-embed-text
- Встановлений Tesseract OCR (для OCR-фолбеку)
- Встановлений Poppler (для рендеру PDF-сторінок у OCR пайплайні)

### 2) Налаштування змінних

Створіть .env на основі .env.example і задайте:

- GROQ_API_KEY
- LASTFM_API_KEY (опційно, але бажано для richer контенту)
- MUSIC_WORKER_USER_AGENT

### 3) Встановлення залежностей

Backend:

```bash
pip install -r requirements.txt
```

Frontend:

```bash
cd frontend-react
npm install
```

### 4) Ініціалізація бази (перший запуск)

```bash
cd backend
py init_database.py
```

### 5) Запуск сервера

```bash
cd backend
py server.py
```

Сервер стартує на: http://127.0.0.1:5000

### 6) Запуск фронтенду

```bash
cd frontend-react
npm run dev
```

В dev-режимі фронт проксує /chat і /upload на Flask.

## Додаткові сценарії

### Наповнення бази артистами (worker)

```bash
cd backend
py music_worker.py
```

### Адмін-команди для ChromaDB

```bash
cd backend
py db_admin.py count
py db_admin.py list --limit 10
py db_admin.py query "Queen" --source-type api_worker --artist Queen
py db_admin.py quality --artist Queen
```

## API (основне)

- POST /chat
  - body: { "message": "..." }
  - result: { "response": "..." }

- POST /upload
  - form-data: file=<pdf>
  - result: status/message або error

## Що демонструє цей проєкт

- Практична реалізація RAG end-to-end
- Комбінація retrieval, reranking та controlled generation
- Робота з OCR-пайплайном і noisy даними
- Розділення на backend, worker, admin CLI та frontend
- Продуманий UX для завантаження документів і чат-взаємодії

## Поточні обмеження

- Якість відповіді залежить від наповненості бази знань
- Для OCR потрібні зовнішні системні залежності (Tesseract/Poppler)
- Локальний запуск Ollama обов'язковий для embeddings

## Автор

Ковальчук Владислав,
Навчальний/курсовий проєкт з фокусом на AI engineering, RAG та applied ML tooling.
