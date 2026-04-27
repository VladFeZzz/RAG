import argparse
import sys
from collections import Counter

import chromadb


DB_PATH = "./my_base"
COLLECTION_NAME = "Curse_docs"
EMBED_MODEL = "nomic-embed-text"


def get_collection(db_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def cmd_count(args):
    _, col = get_collection(args.db_path, args.collection)
    print(f"Collection: {args.collection}")
    print(f"Count: {col.count()}")


def cmd_list(args):
    _, col = get_collection(args.db_path, args.collection)
    total = col.count()
    if total == 0:
        print("Collection is empty.")
        return

    rows = col.get(limit=min(args.limit, total), include=["documents", "metadatas"])
    ids = rows.get("ids", [])
    docs = rows.get("documents", [])
    metas = rows.get("metadatas", [])

    print(f"Collection: {args.collection}")
    print(f"Total: {total}")
    print(f"Showing: {len(ids)}")
    print("-" * 80)

    for i, doc_id in enumerate(ids, start=1):
        doc = (docs[i - 1] or "").replace("\n", " ").strip()
        meta = metas[i - 1] if i - 1 < len(metas) else None
        short = doc[: args.preview]
        print(f"{i}. id={doc_id}")
        print(f"   doc: {short}")
        print(f"   meta: {meta}")


def cmd_show(args):
    _, col = get_collection(args.db_path, args.collection)
    row = col.get(ids=[args.id], include=["documents", "metadatas"])

    ids = row.get("ids", [])
    if not ids:
        print(f"Document with id '{args.id}' not found.")
        return

    doc = (row.get("documents", [""])[0] or "").strip()
    meta = row.get("metadatas", [None])[0]

    print(f"id: {args.id}")
    print(f"meta: {meta}")
    print("-" * 80)
    print(doc)


def cmd_stats(args):
    _, col = get_collection(args.db_path, args.collection)
    total = col.count()
    if total == 0:
        print("Collection is empty.")
        return

    rows = col.get(limit=total, include=[])
    ids = rows.get("ids", [])

    kinds = []
    for doc_id in ids:
        if "_chunk_" in doc_id:
            kinds.append("uploaded_pdf_chunk")
        elif doc_id.startswith("doc_"):
            kinds.append("seed_doc")
        else:
            kinds.append("other")

    counts = Counter(kinds)

    print(f"Collection: {args.collection}")
    print(f"Total: {total}")
    print("Id types:")
    for k, v in counts.items():
        print(f"  - {k}: {v}")


def cmd_query(args):
    try:
        import ollama
    except Exception as e:
        print("Ollama package is required for query command.")
        print(f"Import error: {e}")
        sys.exit(1)

    _, col = get_collection(args.db_path, args.collection)

    emb = ollama.embeddings(model=args.embed_model, prompt=args.text)["embedding"]

    query_kwargs = {
        "query_embeddings": [emb],
        "n_results": args.top_k,
    }
    if args.source_type:
        query_kwargs["where"] = {"source_type": args.source_type}

    try:
        res = col.query(**query_kwargs)
    except Exception:
        # Fallback for older Chroma behavior without/with partial where support.
        res = col.query(query_embeddings=[emb], n_results=max(args.top_k * 3, args.top_k))

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    if args.source_type:
        filtered = []
        for doc_id, doc, dist, meta in zip(ids, docs, dists, metas):
            meta = meta or {}
            if meta.get("source_type") == args.source_type:
                filtered.append((doc_id, doc, dist, meta))
        ids = [x[0] for x in filtered][: args.top_k]
        docs = [x[1] for x in filtered][: args.top_k]
        dists = [x[2] for x in filtered][: args.top_k]

    if args.artist:
        filtered = []
        for doc_id, doc, dist in zip(ids, docs, dists):
            if args.artist.lower() in (doc or "").lower() or args.artist.lower() in str(doc_id).lower():
                filtered.append((doc_id, doc, dist))
        ids = [x[0] for x in filtered][: args.top_k]
        docs = [x[1] for x in filtered][: args.top_k]
        dists = [x[2] for x in filtered][: args.top_k]

    print(f"Query: {args.text}")
    print(f"Top K: {args.top_k}")
    if args.source_type:
        print(f"Filter source_type: {args.source_type}")
    if args.artist:
        print(f"Filter artist: {args.artist}")
    print("-" * 80)

    for i, (doc_id, doc, dist) in enumerate(zip(ids, docs, dists), start=1):
        short = (doc or "").replace("\n", " ").strip()[: args.preview]
        print(f"{i}. id={doc_id} dist={dist:.4f}")
        print(f"   doc: {short}")


def cmd_quality(args):
    _, col = get_collection(args.db_path, args.collection)

    if args.artist:
        where = {
            "$and": [
                {"source_type": "api_worker"},
                {"artist": args.artist},
            ]
        }
    else:
        where = {"source_type": "api_worker"}

    try:
        rows = col.get(where=where, include=["documents", "metadatas"])
    except Exception:
        rows = col.get(include=["documents", "metadatas"])

    ids = rows.get("ids", [])
    docs = rows.get("documents", [])
    metas = rows.get("metadatas", [])

    if args.artist:
        selected = []
        for doc_id, doc, meta in zip(ids, docs, metas):
            if args.artist.lower() in str(doc_id).lower() or (meta or {}).get("artist", "").lower() == args.artist.lower():
                selected.append((doc_id, doc, meta))
        ids = [x[0] for x in selected]
        docs = [x[1] for x in selected]
        metas = [x[2] for x in selected]

    if not ids:
        print("No worker documents found for quality check.")
        return

    joined = "\n".join((d or "") for d in docs)
    checks = {
        "has_musicbrainz_id": "musicbrainz id:" in joined.lower(),
        "has_mb_tags": "теги (musicbrainz):" in joined.lower(),
        "has_lastfm_tracks": "топ треки (last.fm):" in joined.lower(),
        "has_biography": "коротка історія/біографія:" in joined.lower(),
        "has_timeline": "початок активності:" in joined.lower() or "кінець активності:" in joined.lower(),
    }
    score = sum(1 for v in checks.values() if v)

    print("Worker content quality report")
    print(f"Artist: {args.artist or 'ALL'}")
    print(f"Chunks checked: {len(ids)}")
    print(f"Quality score: {score}/5")
    print("-" * 80)
    for name, ok in checks.items():
        print(f"{name}: {'OK' if ok else 'MISS'}")

    print("-" * 80)
    print("Sample IDs:")
    for doc_id in ids[:5]:
        print(f"  - {doc_id}")


def cmd_delete(args):
    _, col = get_collection(args.db_path, args.collection)
    col.delete(ids=[args.id])
    print(f"Deleted id: {args.id}")


def cmd_reset(args):
    if not args.yes:
        print("Reset is destructive. Re-run with --yes to confirm.")
        return

    client, _ = get_collection(args.db_path, args.collection)
    try:
        client.delete_collection(args.collection)
        print(f"Deleted collection: {args.collection}")
    except Exception as e:
        print(f"Delete skipped: {e}")

    client.get_or_create_collection(name=args.collection)
    print(f"Recreated empty collection: {args.collection}")


def build_parser():
    p = argparse.ArgumentParser(description="ChromaDB admin CLI")
    p.add_argument("--db-path", default=DB_PATH, help="Path to Chroma persistent directory")
    p.add_argument("--collection", default=COLLECTION_NAME, help="Collection name")

    sub = p.add_subparsers(dest="command", required=True)

    p_count = sub.add_parser("count", help="Show document count")
    p_count.set_defaults(func=cmd_count)

    p_list = sub.add_parser("list", help="List first N documents")
    p_list.add_argument("--limit", type=int, default=10, help="How many docs to show")
    p_list.add_argument("--preview", type=int, default=160, help="Preview chars for doc text")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show full document by id")
    p_show.add_argument("id", help="Document id")
    p_show.set_defaults(func=cmd_show)

    p_stats = sub.add_parser("stats", help="Show ID type distribution")
    p_stats.set_defaults(func=cmd_stats)

    p_query = sub.add_parser("query", help="Semantic search")
    p_query.add_argument("text", help="Query text")
    p_query.add_argument("--top-k", type=int, default=3, help="Top K results")
    p_query.add_argument("--embed-model", default=EMBED_MODEL, help="Ollama embedding model")
    p_query.add_argument("--preview", type=int, default=160, help="Preview chars for doc text")
    p_query.add_argument("--source-type", help="Optional metadata filter, e.g. api_worker")
    p_query.add_argument("--artist", help="Optional artist text filter")
    p_query.set_defaults(func=cmd_query)

    p_quality = sub.add_parser("quality", help="Check richness of worker artist content")
    p_quality.add_argument("--artist", help="Exact artist in metadata (optional)")
    p_quality.set_defaults(func=cmd_quality)

    p_delete = sub.add_parser("delete", help="Delete by id")
    p_delete.add_argument("id", help="Document id")
    p_delete.set_defaults(func=cmd_delete)

    p_reset = sub.add_parser("reset", help="Drop and recreate collection")
    p_reset.add_argument("--yes", action="store_true", help="Confirm destructive reset")
    p_reset.set_defaults(func=cmd_reset)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# Quick commands:
# py db_admin.py count                        -> count all records
# py db_admin.py list --limit 10             -> show first 10 records
# py db_admin.py show doc_9                  -> show one record by id
# py db_admin.py stats                        -> show id type stats
# py db_admin.py query "що таке spotify"      -> semantic search top-3
# py db_admin.py query "Slayer" --source-type api_worker --artist Slayer
# py db_admin.py quality --artist Slayer      -> quality check for one artist
# py db_admin.py delete doc_9                -> delete one record by id
# py db_admin.py reset --yes                 -> drop and recreate collection