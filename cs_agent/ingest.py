"""Build the Redis knowledge-base index from kb/documents at startup.

Runs before the agent is served (main.py imports it), so the agent card only
becomes available once the index is ready. Vector embeddings are added when
model credentials are available; otherwise the index is BM25-only."""

import json
import os
import struct
import sys
from pathlib import Path

import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from rag_tools import DOC_PREFIX, EMBEDDING_DIM, KB_INDEX, REDIS_URL, _embed

KB_DOCUMENTS_DIR = Path(os.environ.get("KB_DOCUMENTS_DIR", "/app/kb/documents"))

EMBED_BATCH_SIZE = 25


def load_documents() -> list[dict]:
    """Load all KB documents ({id, title, content})."""
    docs = []
    for path in sorted(KB_DOCUMENTS_DIR.glob("*.json")):
        with open(path) as fp:
            docs.append(json.load(fp))
    return docs


def build_index() -> None:
    """(Re)create the KB index and load every document, embedding if possible."""
    client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No KB documents found in {KB_DOCUMENTS_DIR}")

    try:
        client.ft(KB_INDEX).dropindex(delete_documents=True)
    except redis.ResponseError:
        pass

    client.ft(KB_INDEX).create_index(
        fields=[
            TextField("title", weight=2.0),
            TextField("content"),
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
            ),
        ],
        definition=IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH),
    )

    # Embeddings are optional: without model credentials we fall back to
    # BM25-only so the local dev loop works out of the box.
    embeddings: list[list[float] | None] = [None] * len(documents)
    try:
        for start in range(0, len(documents), EMBED_BATCH_SIZE):
            batch = documents[start : start + EMBED_BATCH_SIZE]
            vectors = _embed([f"{d['title']}\n{d['content']}" for d in batch])
            embeddings[start : start + len(vectors)] = vectors
        print(f"[ingest] embedded {len(documents)} documents", file=sys.stderr)
    except Exception as e:
        embeddings = [None] * len(documents)
        print(
            f"[ingest] embeddings unavailable ({e}); kb_search_vector disabled, "
            "kb_search_bm25 still works",
            file=sys.stderr,
        )

    pipe = client.pipeline(transaction=False)
    for doc, vector in zip(documents, embeddings):
        mapping = {"title": doc["title"], "content": doc["content"]}
        if vector is not None:
            mapping["embedding"] = struct.pack(f"{EMBEDDING_DIM}f", *vector)
        pipe.hset(f"{DOC_PREFIX}{doc['id']}", mapping=mapping)
    pipe.execute()
    print(f"[ingest] indexed {len(documents)} documents into {KB_INDEX}", file=sys.stderr)


if __name__ == "__main__":
    build_index()
