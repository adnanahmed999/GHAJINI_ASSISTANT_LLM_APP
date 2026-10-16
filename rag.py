"""
RAG: search Sanjay's documents by meaning, using FAISS as the vector database.

Three separate jobs:
  1. Embedding model (OpenAI)   turns text into numbers (1536 per document)
  2. data/embeddings.json       keeps those numbers, so we only pay for them once
  3. FAISS (vector database)    searches the numbers to find the closest ones

  first run:   documents -> OpenAI -> embeddings.json -> FAISS
  later runs:  embeddings.json -> FAISS          (no embedding cost)
  each search: question -> OpenAI -> FAISS finds the nearest documents

Commands:
  python rag.py "am I allergic to anything?"    search
  python rag.py --rebuild                       re-create embeddings.json (e.g. after changing the documents)
"""

import json

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import EMBED_MODEL, EMBEDDINGS_FILE
from doc_store import doc_to_text, load_docs

_embedder = None
_index = None    # the FAISS index: holds only the numbers
_chunks = None   # the text and details of each row of the index, in the same order


def get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    return _embedder


# ---------- Job 1 + 2: make the embeddings and keep them in a JSON file ----------

def rebuild_embeddings() -> list[dict]:
    """Embed every document (one document = one chunk) with OpenAI and save to embeddings.json."""
    docs = load_docs()
    texts = [doc_to_text(d) for d in docs]
    vectors = get_embedder().embed_documents(texts)  # one OpenAI call for all documents

    chunks = [
        {"id": d["id"], "title": d["title"], "date": d["date"], "text": t,
         "embedding": [round(x, 6) for x in v]}
        for d, t, v in zip(docs, texts, vectors)
    ]
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_FILE.write_text(json.dumps({"model": EMBED_MODEL, "chunks": chunks}), encoding="utf-8")
    print(f"[rag] Embedded {len(chunks)} documents and saved them to {EMBEDDINGS_FILE.name}")
    return chunks


def load_embeddings() -> list[dict]:
    """Read embeddings.json; re-create it if it is missing, from another model, or the documents have changed."""
    if EMBEDDINGS_FILE.exists():
        data = json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
        current_texts = [doc_to_text(d) for d in load_docs()]
        saved_texts = [c["text"] for c in data.get("chunks", [])]
        if data.get("model") == EMBED_MODEL and saved_texts == current_texts:
            print(f"[rag] Loaded {len(saved_texts)} documents from {EMBEDDINGS_FILE.name} (no embedding cost)")
            return data["chunks"]
        print("[rag] Documents or model changed since embeddings.json was made, rebuilding it")
    return rebuild_embeddings()


# ---------- Job 3: put the numbers into FAISS and search ----------

def build_index(chunks: list[dict]):
    """Put all the vectors into a FAISS index (exact search, Euclidean/L2 distance)."""
    global _index, _chunks
    vectors = np.array([c["embedding"] for c in chunks], dtype="float32")
    _index = faiss.IndexFlatL2(vectors.shape[1])   # 1536 dimensions
    _index.add(vectors)
    _chunks = chunks


def search(query: str, k: int = 5) -> list[dict]:
    """Return the k documents closest in meaning to the query. Lower distance = better match."""
    if _index is None:
        build_index(load_embeddings())

    q = np.array([get_embedder().embed_query(query)], dtype="float32")
    distances, positions = _index.search(q, k)   # FAISS returns the k nearest rows

    return [
        {"id": _chunks[i]["id"], "title": _chunks[i]["title"], "date": _chunks[i]["date"],
         "distance": round(float(dist), 3), "text": _chunks[i]["text"]}
        for dist, i in zip(distances[0], positions[0])
    ]


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--rebuild"]:
        rebuild_embeddings()
        sys.exit()

    question = " ".join(sys.argv[1:]) or "am I allergic to anything?"
    for rank, hit in enumerate(search(question, k=3), 1):
        print(f"\n--- #{rank}  (distance {hit['distance']}, lower = closer)\n{hit['text']}")
