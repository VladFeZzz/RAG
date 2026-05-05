import io
import os
import re
import time
import threading
import subprocess
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chromadb
import ollama
from groq import Groq
from dotenv import load_dotenv
import pypdf
from werkzeug.exceptions import RequestEntityTooLarge

RELEVANCE_DISTANCE_THRESHOLD = 420.0
MAX_CHUNK_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200
N_RESULTS = 8
OCR_MIN_NATIVE_TEXT_CHARS = 800
OCR_MIN_ALPHA_RATIO = 0.45

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
DB_PATH = os.path.join(BASE_DIR, "my_base")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

load_dotenv(dotenv_path=ENV_PATH)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "45"))
MAX_UPLOAD_PAGES = int(os.getenv("MAX_UPLOAD_PAGES", "350"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

WORKER_AUTOSTART = os.getenv("WORKER_AUTOSTART", "false").strip().lower() in {"1", "true", "yes"}
WORKER_INTERVAL_MIN = int(os.getenv("WORKER_INTERVAL_MIN", "60"))
WORKER_ON_START = os.getenv("WORKER_ON_START", "true").strip().lower() in {"1", "true", "yes"}

ARTISTS_FILE = os.path.join(BASE_DIR, "artists.txt")
ARTISTS_RAW = os.getenv("WORKER_ARTISTS", "")
if os.path.exists(ARTISTS_FILE):
    with open(ARTISTS_FILE, "r", encoding="utf-8") as f:
        data = f.read().strip()
    if data:
        KNOWN_ARTISTS = [x.strip() for x in re.split(r"[\n,]+", data) if x.strip()]
    elif ARTISTS_RAW.strip():
        KNOWN_ARTISTS = [x.strip() for x in ARTISTS_RAW.split(",") if x.strip()]
    else:
        KNOWN_ARTISTS = ["Metallica", "Slayer", "Megadeth", "Anthrax", "Iron Maiden"]
else:
    if ARTISTS_RAW.strip():
        KNOWN_ARTISTS = [x.strip() for x in ARTISTS_RAW.split(",") if x.strip()]
    else:
        KNOWN_ARTISTS = ["Metallica", "Slayer", "Megadeth", "Anthrax", "Iron Maiden"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env file")
    exit()
else:
    print("Groq API key loaded successfully")

groq_client = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path=DB_PATH)

_worker_lock = threading.Lock()
_worker_running = False


def get_collection():
    # Re-fetch collection handle to immediately observe updates written by other processes.
    return chroma_client.get_or_create_collection(name="Curse_docs")


def run_worker_once():
    global _worker_running
    if _worker_lock.locked():
        return

    with _worker_lock:
        _worker_running = True
        try:
            worker_path = os.path.join(BASE_DIR, "music_worker.py")
            subprocess.run([sys.executable, worker_path], check=False)
        finally:
            _worker_running = False


def worker_loop():
    if WORKER_ON_START:
        run_worker_once()

    interval_sec = max(10, WORKER_INTERVAL_MIN * 60)
    while True:
        time.sleep(interval_sec)
        run_worker_once()


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(1, len(text))


def split_text_with_overlap(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + max_chars, text_len)

        if end < text_len:
            split_pos = cleaned.rfind(" ", start, end)
            if split_pos > start + int(max_chars * 0.6):
                end = split_pos

        chunk = cleaned[start:end].strip()
        if len(chunk) >= 30:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(0, end - overlap)

    return chunks


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception:
        return 0


def extract_pdf_text_native(pdf_bytes: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)
    except Exception:
        return ""


def extract_pdf_text_with_ocr(pdf_bytes: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        from pdf2image.exceptions import PDFInfoNotInstalledError
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError as e:
        raise RuntimeError(
            "OCR dependencies are missing. Install pdf2image, pytesseract, pillow and make sure Tesseract OCR is installed on OS."
        ) from e

    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=250)
    except PDFInfoNotInstalledError as e:
        raise RuntimeError(
            "Poppler is not installed or not available in PATH. Install Poppler and restart the server."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF pages for OCR: {e}") from e

    page_texts = []
    for page_num, image in enumerate(pages, start=1):
        try:
            text = pytesseract.image_to_string(image, lang="ukr+eng")
        except TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract OCR is not installed or not available in PATH. Install Tesseract and restart the server."
            ) from e
        except Exception as e:
            raise RuntimeError(f"OCR failed on page {page_num}: {e}") from e

        if text and text.strip():
            page_texts.append(text.strip())

    return "\n\n".join(page_texts)


def extract_pdf_text_hybrid(pdf_bytes: bytes):
    native_text = extract_pdf_text_native(pdf_bytes)
    if len(native_text) >= OCR_MIN_NATIVE_TEXT_CHARS and alpha_ratio(native_text) >= OCR_MIN_ALPHA_RATIO:
        return native_text, False

    ocr_text = extract_pdf_text_with_ocr(pdf_bytes)
    if len(ocr_text) > len(native_text):
        return ocr_text, True
    return native_text, False


def clean_bot_response(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []
    for line in text.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("context") or lowered.startswith("контекст"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def has_keyword_overlap(question: str, context: str) -> bool:
    tokens = [t.strip(".,!?()[]{}:;\"'`).").lower() for t in question.split()]
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return False

    context_lower = context.lower()
    return any(token in context_lower for token in tokens)


def extract_target_artist(question: str) -> str | None:
    q = (question or "").lower()
    for artist in KNOWN_ARTISTS:
        if artist.lower() in q:
            return artist
    return None


def build_artist_response_from_context(target_artist: str, context: str) -> str:
    if not target_artist or not context:
        return ""

    question_lower = ""
    if isinstance(context, dict):
        question_lower = str(context.get("question", "")).lower()
        context = str(context.get("context", ""))

    metadata_request = any(
        token in question_lower
        for token in ["musicbrainz", "країн", "country", "тег", "tag", "треки", "track", "playcount", "listeners", "id"]
    )

    system_prompt = f"""
You are a music historian and editor.
Write in Ukrainian only.
Use only the provided context about {target_artist}.
Do not output raw field labels like MusicBrainz ID, Теги, or Топ треки as a list.
Give a short, natural answer in 4 to 6 sentences.
For a general question, focus on the artist's history, origin, style, influence, and notable songs.
If the user explicitly asks for metadata, include only the relevant facts briefly and naturally.
Translate any English facts from the context into fluent Ukrainian.
Keep the answer concise and avoid repeating the same fact twice.
""".strip()

    user_prompt = f"Питання: {question_lower or target_artist}\n\nКонтекст:\n{context}\n\nСформулюй відповідь."

    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3 if metadata_request else 0.4,
        )
        answer = clean_bot_response(completion.choices[0].message.content)
        return answer
    except Exception as e:
        print(f"Artist summary generation warning: {e}")
        return ""


def rerank_by_keyword_overlap(question: str, docs: list[str], distances: list[float]):
    q_tokens = [t.strip(".,!?()[]{}:;\"'`").lower() for t in question.split()]
    q_tokens = [t for t in q_tokens if len(t) >= 3]

    scored = []
    for i, doc in enumerate(docs):
        d = doc or ""
        d_low = d.lower()
        overlap = sum(1 for t in q_tokens if t in d_low)
        dist = distances[i] if i < len(distances) else 10**9
        scored.append((overlap, -dist, d, dist))

    scored.sort(reverse=True)
    ranked_docs = [x[2] for x in scored]
    ranked_dists = [x[3] for x in scored]
    return ranked_docs, ranked_dists


@app.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


@app.errorhandler(RequestEntityTooLarge)
def handle_large_payload(_error):
    return jsonify({
        "error": (
            f"Файл занадто великий. Максимальний розмір: {MAX_UPLOAD_MB} MB. "
            "Спробуйте зменшити PDF або розбити його на кілька частин."
        )
    }), 413


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = (data.get('message') or '').strip()
    if not user_question:
        return jsonify({"response": "Будь ласка, введіть питання."}), 400

    print(f"\nUser question: {user_question}")

    try:
        collection = get_collection()
        query_embed = ollama.embeddings(model="nomic-embed-text", prompt=user_question)['embedding']
        results = collection.query(query_embeddings=[query_embed], n_results=N_RESULTS)

        context = ""
        best_distance = None
        if results['documents'] and results['documents'][0]:
            raw_docs = [doc for doc in results['documents'][0] if doc]
            raw_dists = results.get('distances', [[]])[0] if results.get('distances') else []

            target_artist = extract_target_artist(user_question)
            if target_artist:
                try:
                    artist_results = collection.query(
                        query_embeddings=[query_embed],
                        n_results=max(N_RESULTS, 8),
                        where={
                            "$and": [
                                {"source_type": "api_worker"},
                                {"artist": target_artist},
                            ]
                        },
                    )
                    artist_docs = artist_results.get('documents', [[]])[0] if artist_results.get('documents') else []
                    artist_dists = artist_results.get('distances', [[]])[0] if artist_results.get('distances') else []

                    # Prefer exact artist worker docs by prepending them before global matches.
                    merged_docs = []
                    merged_dists = []
                    for doc, dist in zip(artist_docs, artist_dists):
                        if doc:
                            merged_docs.append(doc)
                            merged_dists.append(dist)
                    for idx, doc in enumerate(raw_docs):
                        if doc and doc not in merged_docs:
                            merged_docs.append(doc)
                            merged_dists.append(raw_dists[idx] if idx < len(raw_dists) else 10**9)

                    raw_docs = merged_docs
                    raw_dists = merged_dists
                except Exception as e:
                    print(f"Artist-filter query warning: {e}")

            top_docs, top_dists = rerank_by_keyword_overlap(user_question, raw_docs, raw_dists)

            if target_artist:
                artist_docs = []
                artist_dists = []
                other_docs = []
                other_dists = []

                for idx, doc in enumerate(top_docs):
                    dist = top_dists[idx] if idx < len(top_dists) else 10**9
                    if target_artist.lower() in (doc or "").lower():
                        artist_docs.append(doc)
                        artist_dists.append(dist)
                    else:
                        other_docs.append(doc)
                        other_dists.append(dist)

                if artist_docs:
                    top_docs = artist_docs + other_docs
                    top_dists = artist_dists + other_dists

            if target_artist:
                artist_only_docs = [doc for doc in top_docs if target_artist.lower() in (doc or "").lower()]
                if artist_only_docs:
                    top_docs = artist_only_docs

            top_docs = top_docs[:3]
            top_dists = top_dists[:3] if top_dists else []

            context = "\n\n".join(top_docs)
            if top_dists:
                best_distance = min(top_dists)

            print(f"Found {len(top_docs)} context chunks after rerank")
            if best_distance is not None:
                print(f"Best distance: {best_distance:.4f}")
        else:
            print("No context found in database")

    except Exception as e:
        print(f"Search error: {e}")
        context = ""
        best_distance = None

    target_artist = extract_target_artist(user_question)
    artist_match_in_context = bool(target_artist and target_artist.lower() in context.lower())
    weak_retrieval = best_distance is not None and best_distance > RELEVANCE_DISTANCE_THRESHOLD
    if not context or (weak_retrieval and not has_keyword_overlap(user_question, context) and not artist_match_in_context):
        return jsonify({
            "response": "Не знайшов достатньо релевантної інформації у базі знань для цього питання. Спробуй перефразувати запит або додати PDF з потрібною темою."
        })

    if target_artist and artist_match_in_context:
        direct_artist_answer = build_artist_response_from_context(
            target_artist,
            {
                "question": user_question,
                "context": context,
            },
        )
        if direct_artist_answer:
            return jsonify({"response": direct_artist_answer})

    rag_hint = ""
    if "rag" in user_question.lower():
        rag_hint = (
            "In this project, the term RAG means Retrieval-Augmented Generation unless "
            "the user explicitly asks about a music genre."
        )

    system_prompt = f"""
    You are a helpful music expert assistant.
    Answer the user's question based ONLY on the provided snippets.
    If the answer is not in the context, politely say that you don't have this information in your database.
    Respond in Ukrainian.
    Never include technical labels like CONTEXT, snippets, source, or metadata in the final answer.
    {rag_hint}

    KNOWLEDGE SNIPPETS:
    {context}
    """

    print("Generating response...")
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        bot_response = clean_bot_response(completion.choices[0].message.content)
        print("Response generated successfully")

        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Groq generation error: {e}")
        return jsonify({"response": "Вибачте, сталася помилка при генерації відповіді."}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Файл не знайдено у запиті"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Файл не вибрано"}), 400

    print(f"\nProcessing file: {file.filename}")

    try:
        started_at = time.time()
        collection = get_collection()
        pdf_bytes = file.read()
        if not pdf_bytes:
            return jsonify({"error": "Файл порожній або не вдалося прочитати вміст."}), 400

        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_MB:
            return jsonify({
                "error": (
                    f"Файл має {file_size_mb:.1f} MB, що перевищує ліміт {MAX_UPLOAD_MB} MB. "
                    "Розбийте PDF на частини або стисніть файл."
                )
            }), 400

        page_count = get_pdf_page_count(pdf_bytes)
        if page_count > MAX_UPLOAD_PAGES:
            return jsonify({
                "error": (
                    f"PDF має {page_count} сторінок, а ліміт становить {MAX_UPLOAD_PAGES}. "
                    "Для стабільної роботи завантажте книгу частинами."
                )
            }), 400

        text_content, used_ocr = extract_pdf_text_hybrid(pdf_bytes)
        chunks = split_text_with_overlap(text_content)
        chunks = [c for c in chunks if len(c) >= 60 and alpha_ratio(c) >= 0.35]

        if not chunks:
            return jsonify({"error": "Не вдалося витягнути текст із PDF. Спробуйте інший файл або перевірте OCR налаштування."}), 400

        print(f"Text split into {len(chunks)} chunks. Starting indexing...")

        upload_stamp = int(time.time())
        for i, chunk in enumerate(chunks):
            embed = ollama.embeddings(model="nomic-embed-text", prompt=chunk)['embedding']
            chunk_id = f"{file.filename}_{upload_stamp}_chunk_{i}"

            collection.upsert(
                ids=[chunk_id],
                embeddings=[embed],
                documents=[chunk],
                metadatas=[{
                    "source_file": file.filename,
                    "chunk_index": i,
                    "ocr_used": used_ocr,
                    "upload_stamp": upload_stamp,
                }],
            )

        elapsed = time.time() - started_at
        print(f"File {file.filename} successfully added to knowledge base")
        return jsonify({
            "status": "success",
            "message": "Файл успішно оброблено та збережено!",
            "pages": page_count,
            "chunks": len(chunks),
            "ocr_used": used_ocr,
            "processing_seconds": round(elapsed, 2),
        })

    except Exception as e:
        print(f"PDF processing error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\nServer started at http://127.0.0.1:5000")
    print("Press CTRL+C to stop\n")
    if WORKER_AUTOSTART:
        thread = threading.Thread(target=worker_loop, daemon=True)
        thread.start()
        print(
            f"Worker autostart enabled. Interval: {WORKER_INTERVAL_MIN} min, "
            f"run_on_start={WORKER_ON_START}"
        )
    app.run(debug=True, use_reloader=False, port=5000)