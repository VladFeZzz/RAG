import os
import re
import time
import json
import hashlib
import requests
from datetime import datetime, timezone
from requests.exceptions import SSLError, RequestException

import chromadb
import ollama
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


# Load .env from project root
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

DB_PATH = os.path.join(BASE_DIR, "my_base")
COLLECTION_NAME = "Curse_docs"
EMBED_MODEL = "nomic-embed-text"

# MusicBrainz requires polite User-Agent and low request rate
USER_AGENT = os.getenv("MUSIC_WORKER_USER_AGENT", "music-rag-worker/1.0 (student project)")
MB_BASE = "https://musicbrainz.org/ws/2"
LASTFM_BASE = "http://ws.audioscrobbler.com/2.0/"
REQUEST_TIMEOUT = 25
MB_RETRIES = 3

# Optional API key (worker still runs with MusicBrainz only)
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "").strip()

# Comma separated list in .env, fallback defaults
ARTISTS_RAW = os.getenv("WORKER_ARTISTS", "Metallica,Slayer,Megadeth,Anthrax")
ARTISTS = [x.strip() for x in ARTISTS_RAW.split(",") if x.strip()]

# Keep chunks compact for embeddings
MAX_CHARS = 1200
OVERLAP = 150

# State file for run stats (optional)
STATE_PATH = os.path.join(BASE_DIR, "worker_state.json")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(1, len(text))


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            split_pos = text.rfind(" ", start, end)
            if split_pos > start + int(max_chars * 0.6):
                end = split_pos

        chunk = text[start:end].strip()
        if len(chunk) >= 60 and alpha_ratio(chunk) >= 0.35:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def mb_get(path: str, params: dict):
    time.sleep(1.1)  # respect MB rate limits
    headers = {"User-Agent": USER_AGENT}

    mb_urls = [MB_BASE]
    # Fallback for environments where TLS chain/handshake is problematic.
    if MB_BASE.startswith("https://"):
        mb_urls.append(MB_BASE.replace("https://", "http://", 1))

    last_err = None
    for base in mb_urls:
        for _ in range(MB_RETRIES):
            try:
                r = requests.get(f"{base}{path}", params=params, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except (SSLError, RequestException) as e:
                last_err = e
                time.sleep(0.8)

    raise RuntimeError(f"MusicBrainz request failed after retries: {last_err}")


def fetch_musicbrainz_artist(artist_name: str):
    try:
        data = mb_get("/artist", {"query": f'artist:"{artist_name}"', "fmt": "json", "limit": 1})
        artists = data.get("artists", [])
        if not artists:
            return None
        return artists[0]
    except Exception as e:
        print(f"[WARN] MusicBrainz artist fetch failed for {artist_name}: {e}")
        return None


def fetch_musicbrainz_recordings(artist_mbid: str, limit: int = 8):
    try:
        data = mb_get("/recording", {"artist": artist_mbid, "fmt": "json", "limit": limit})
        return data.get("recordings", []) or []
    except Exception as e:
        print(f"[WARN] MusicBrainz recordings fetch failed for {artist_mbid}: {e}")
        return []


def fetch_lastfm_toptracks(artist_name: str, limit: int = 8):
    if not LASTFM_API_KEY:
        return []

    params = {
        "method": "artist.gettoptracks",
        "artist": artist_name,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": str(limit),
    }
    r = requests.get(LASTFM_BASE, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    tracks = data.get("toptracks", {}).get("track", [])
    if isinstance(tracks, dict):
        tracks = [tracks]
    return tracks or []


def build_artist_doc(artist_name: str, mb_artist: dict, recordings: list, lastfm_tracks: list):
    mbid = mb_artist.get("id", "")
    country = mb_artist.get("country", "невідомо")
    disambiguation = mb_artist.get("disambiguation", "")
    tags = [t.get("name") for t in mb_artist.get("tags", []) if t.get("name")][:8]

    rec_titles = [r.get("title") for r in recordings if r.get("title")][:12]

    lastfm_titles = []
    for t in lastfm_tracks:
        name = t.get("name")
        playcount = t.get("playcount")
        if name:
            if playcount:
                lastfm_titles.append(f"{name} (playcount: {playcount})")
            else:
                lastfm_titles.append(name)

    text_parts = [
        f"Артист: {artist_name}",
        f"MusicBrainz ID: {mbid}",
        f"Країна: {country}",
    ]

    if disambiguation:
        text_parts.append(f"Опис/уточнення: {disambiguation}")

    if tags:
        text_parts.append("Теги (MusicBrainz): " + ", ".join(tags))

    if rec_titles:
        text_parts.append("Записи/треки (MusicBrainz): " + ", ".join(rec_titles))

    if lastfm_titles:
        text_parts.append("Топ треки (Last.fm): " + ", ".join(lastfm_titles))

    text_parts.append(
        "Джерела: MusicBrainz API, Last.fm API. Цей запис автоматично зібраний воркером."
    )

    return "\n".join(text_parts)


def build_lastfm_only_doc(artist_name: str, lastfm_tracks: list):
    lastfm_titles = []
    for t in lastfm_tracks:
        name = t.get("name")
        playcount = t.get("playcount")
        if name:
            if playcount:
                lastfm_titles.append(f"{name} (playcount: {playcount})")
            else:
                lastfm_titles.append(name)

    text_parts = [
        f"Артист: {artist_name}",
        "Опис зібрано в режимі Last.fm only, бо MusicBrainz був тимчасово недоступний.",
    ]

    if lastfm_titles:
        text_parts.append("Топ треки (Last.fm): " + ", ".join(lastfm_titles))

    text_parts.append("Джерело: Last.fm API. Цей запис автоматично зібраний воркером.")
    return "\n".join(text_parts)


def upsert_document(col, artist_name: str, full_text: str, source_key: str):
    doc_hash = stable_hash(full_text)
    chunks = split_text(full_text)

    if not chunks:
        return 0

    ids = []
    docs = []
    embs = []
    metas = []

    for i, chunk in enumerate(chunks):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        ids.append(f"api_{source_key}_{artist_name}_{doc_hash}_chunk_{i}")
        docs.append(chunk)
        embs.append(emb)
        metas.append(
            {
                "source_type": "api_worker",
                "source_key": source_key,
                "artist": artist_name,
                "hash": doc_hash,
                "chunk_index": i,
                "fetched_at": now_iso(),
            }
        )

    col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return len(chunks)


def run_smoke_queries(col, artists: list[str]):
    ok = True
    print("\n=== Smoke queries ===")
    for artist in artists:
        q = f"розкажи про гурт {artist}"
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=q)["embedding"]
        docs = []

        # Validate against worker-produced records only, not the entire mixed collection.
        try:
            r = col.query(
                query_embeddings=[emb],
                n_results=3,
                where={"source_type": "api_worker", "artist": artist},
            )
            docs = r.get("documents", [[]])[0]
        except Exception:
            # Fallback for Chroma versions with different where behavior.
            r = col.query(query_embeddings=[emb], n_results=10)
            candidate_docs = r.get("documents", [[]])[0]
            candidate_ids = r.get("ids", [[]])[0]
            docs = [d for d, doc_id in zip(candidate_docs, candidate_ids) if str(doc_id).startswith("api_")]

        joined = " ".join(docs).lower()
        hit = bool(docs) and artist.lower() in joined
        print(f"- {artist}: {'OK' if hit else 'MISS'} (docs={len(docs)})")
        if not hit:
            ok = False
    return ok


def count_worker_docs(col) -> int:
    try:
        rows = col.get(where={"source_type": "api_worker"}, include=[])
    except Exception:
        rows = col.get(include=[])
        ids = rows.get("ids", [])
        return len([x for x in ids if str(x).startswith("api_")])

    return len(rows.get("ids", []))


def save_state(payload: dict):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("State save warning:", e)


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    total_chunks = 0
    processed = 0
    failed = []

    print("Worker start:", now_iso())
    print("Artists:", ARTISTS)
    print("Last.fm key present:", bool(LASTFM_API_KEY))

    for artist in ARTISTS:
        try:
            mb_artist = fetch_musicbrainz_artist(artist)
            lastfm_tracks = fetch_lastfm_toptracks(artist, limit=10)

            if mb_artist:
                mbid = mb_artist.get("id", "")
                recordings = fetch_musicbrainz_recordings(mbid, limit=10) if mbid else []
                doc = build_artist_doc(artist, mb_artist, recordings, lastfm_tracks)
                source_key = "mb_lastfm"
            else:
                if not lastfm_tracks:
                    print(f"[WARN] Artist not available from APIs: {artist}")
                    failed.append({"artist": artist, "reason": "not_found_in_sources"})
                    continue
                doc = build_lastfm_only_doc(artist, lastfm_tracks)
                source_key = "lastfm_only"

            added = upsert_document(col, artist, doc, source_key=source_key)
            total_chunks += added
            processed += 1

            print(f"[OK] {artist}: chunks={added}, source={source_key}")

        except Exception as e:
            print(f"[ERR] {artist}: {e}")
            failed.append({"artist": artist, "reason": str(e)})

    smoke_ok = run_smoke_queries(col, ARTISTS[: min(4, len(ARTISTS))])
    worker_docs_total = count_worker_docs(col)

    state = {
        "ran_at": now_iso(),
        "processed_artists": processed,
        "total_artists": len(ARTISTS),
        "total_chunks_upserted": total_chunks,
        "worker_docs_total": worker_docs_total,
        "failed": failed,
        "smoke_ok": smoke_ok,
        "smoke_scope": "source_type=api_worker",
    }
    save_state(state)

    print("\n=== Worker summary ===")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()