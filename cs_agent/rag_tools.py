"""Knowledge-base search tools backed by Redis (RediSearch).

kb_search_bm25: full-text BM25 search (OR-semantics keyword query).
kb_search_vector: HNSW vector search over gemini-embedding-001 embeddings
(available only when the index was built with embeddings).

Replies are parsed via execute_command so both the classic array reply and
the Redis 8 map-style reply work regardless of redis-py version."""

import hashlib
import os
import re
import struct

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
KB_INDEX = "kb_idx"
DOC_PREFIX = "doc:"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Default retrieval width. The KB splits one account's terms across several
# docs and a single "best option" question can need ~20+ docs (e.g. task_001
# spans 24), so 5 is far too low. Wider recall is the cheapest accuracy win.
DEFAULT_TOP_K = 20

# Query-embedding cache: KB query vectors only (NOT user data), keyed by query
# hash with a TTL, so repeat semantic searches skip the embedding API call.
QUERY_EMBED_CACHE_TTL_S = 86400
_QEMB_PREFIX = "qemb:"

_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
_genai_client = None


def _get_genai_client():
    """Reused genai client (one connection pool, not a new one per search)."""
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client()
    return _genai_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with gemini-embedding-001 via google-genai."""
    from google.genai import types

    # Reduced-dim output is unnormalized; the index uses COSINE, so that's fine.
    result = _get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [e.values for e in result.embeddings]


def _embed_query(query: str) -> list[float]:
    """Embed one query, caching the vector in Redis (TTL) to cut repeat
    embedding cost + latency. Safe: only KB query text is cached, never any
    user/session data, so this respects per-conversation isolation."""
    cache_key = _QEMB_PREFIX + hashlib.sha1(query.encode()).hexdigest()
    try:
        cached = _client.get(cache_key)
        if cached is not None:
            return list(struct.unpack(f"{EMBEDDING_DIM}f", cached))
    except Exception:
        pass
    vector = _embed([query])[0]
    try:
        _client.setex(
            cache_key, QUERY_EMBED_CACHE_TTL_S, struct.pack(f"{EMBEDDING_DIM}f", *vector)
        )
    except Exception:
        pass
    return vector


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_search_reply(reply) -> list[dict]:
    """Normalize an FT.SEARCH reply (array or map shape) to result dicts."""
    if isinstance(reply, dict):
        results = reply.get(b"results", reply.get("results")) or []
        out = []
        for row in results:
            attrs = row.get(b"extra_attributes", row.get("extra_attributes")) or {}
            doc = {"doc_id": _decode(row.get(b"id", row.get("id", "")))}
            doc.update({_decode(k): _decode(v) for k, v in attrs.items()})
            out.append(doc)
        return out
    out = []
    for i in range(1, len(reply) - 1, 2):
        doc = {"doc_id": _decode(reply[i])}
        fields = reply[i + 1]
        for j in range(0, len(fields) - 1, 2):
            doc[_decode(fields[j])] = _decode(fields[j + 1])
        out.append(doc)
    return out


def _strip_score(docs: list[dict]) -> list[dict]:
    for doc in docs:
        doc.pop("score", None)
    return docs


def kb_search_bm25(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Full-text (BM25) search over the Rho-Bank knowledge base.

    Args:
        query: Keywords or a short phrase to search for. Matching is ranked,
            so extra keywords help rather than hurt.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content.
    """
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return []
    # OR-join: RediSearch defaults to AND, which zeroes out long queries.
    or_query = "|".join(dict.fromkeys(terms))
    reply = _client.execute_command(
        "FT.SEARCH", KB_INDEX, or_query,
        "LIMIT", "0", str(top_k),
        "RETURN", "2", "title", "content",
    )
    return _parse_search_reply(reply)


def kb_search_vector(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Semantic (vector) search over the Rho-Bank knowledge base.

    Better than kb_search_bm25 when the query is a natural-language question
    rather than exact keywords.

    Args:
        query: A natural-language question or description.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content; or an error
        entry telling you to fall back to kb_search_bm25.
    """
    try:
        vector = struct.pack(f"{EMBEDDING_DIM}f", *_embed_query(query))
        reply = _client.execute_command(
            "FT.SEARCH", KB_INDEX, f"*=>[KNN {top_k} @embedding $vec AS score]",
            "PARAMS", "2", "vec", vector,
            "SORTBY", "score",
            "LIMIT", "0", str(top_k),
            "RETURN", "3", "title", "content", "score",
            "DIALECT", "2",
        )
        return _strip_score(_parse_search_reply(reply))
    except Exception as e:
        return [
            {
                "error": f"Vector search unavailable ({type(e).__name__}). "
                "Use kb_search_bm25 with keywords instead."
            }
        ]


def kb_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Hybrid knowledge-base search: PREFER THIS for most questions.

    Runs semantic (vector) and keyword (BM25) search and merges them,
    de-duplicated by doc_id, so you get both natural-language and exact-term
    matches in one call and rarely need to search the same topic twice.

    Args:
        query: A natural-language question or keywords. Extra terms help.
        top_k: Target number of documents to return.

    Returns:
        Merged, de-duplicated documents with doc_id, title, and full content.
    """
    merged: dict[str, dict] = {}
    vector_hits = kb_search_vector(query, top_k=top_k)
    if not (len(vector_hits) == 1 and vector_hits[0].get("error")):
        for doc in vector_hits:
            merged.setdefault(doc.get("doc_id") or f"_v{len(merged)}", doc)
    for doc in kb_search_bm25(query, top_k=top_k):
        merged.setdefault(doc.get("doc_id") or f"_b{len(merged)}", doc)
    return list(merged.values())[: max(top_k, 1)]
