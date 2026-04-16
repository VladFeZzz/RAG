import io
import os
import re
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chromadb
import ollama
from groq import Groq
from dotenv import load_dotenv
import pypdf

RELEVANCE_DISTANCE_THRESHOLD = 200.0
MAX_CHUNK_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200
N_RESULTS = 8
OCR_MIN_NATIVE_TEXT_CHARS = 800
OCR_MIN_ALPHA_RATIO = 0.45

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

load_dotenv(dotenv_path="../.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env file")
    exit()
else:
    print("Groq API key loaded successfully")

groq_client = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path="./my_base")
collection = chroma_client.get_or_create_collection(name="Curse_docs")


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


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = (data.get('message') or '').strip()
    if not user_question:
        return jsonify({"response": "Будь ласка, введіть питання."}), 400

    print(f"\nUser question: {user_question}")

    try:
        query_embed = ollama.embeddings(model="nomic-embed-text", prompt=user_question)['embedding']
        results = collection.query(query_embeddings=[query_embed], n_results=N_RESULTS)

        context = ""
        best_distance = None
        if results['documents'] and results['documents'][0]:
            raw_docs = [doc for doc in results['documents'][0] if doc]
            raw_dists = results.get('distances', [[]])[0] if results.get('distances') else []
            top_docs, top_dists = rerank_by_keyword_overlap(user_question, raw_docs, raw_dists)

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

    weak_retrieval = best_distance is not None and best_distance > RELEVANCE_DISTANCE_THRESHOLD
    if not context or (weak_retrieval and not has_keyword_overlap(user_question, context)):
        return jsonify({
            "response": "Не знайшов достатньо релевантної інформації у базі знань для цього питання. Спробуй перефразувати запит або додати PDF з потрібною темою."
        })

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
        pdf_bytes = file.read()
        if not pdf_bytes:
            return jsonify({"error": "Файл порожній або не вдалося прочитати вміст."}), 400

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

        print(f"File {file.filename} successfully added to knowledge base")
        return jsonify({"status": "success", "message": "Файл успішно оброблено та збережено!"})

    except Exception as e:
        print(f"PDF processing error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\nServer started at http://127.0.0.1:5000")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, use_reloader=False, port=5000)