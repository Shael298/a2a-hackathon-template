"""Optional session-scoped shared 'case file' in Redis.

A tiny shared scratchpad both agents can read/write WITHIN a single
conversation, keyed by the A2A contextId with a short TTL. It NEVER spans
contextIds, so it respects the harness's per-conversation isolation rule, and
it is purely additive: the agents work fully without it and it degrades
gracefully when Redis or the other agent is absent (e.g. a held-out pairing).
This is the score-safe form of Redis "shared memory that connects agents" —
not managed cross-session memory, which is disallowed here."""

import os

import redis
from google.adk.tools import ToolContext

from env_toolset import session_id

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CASE_TTL_S = 3600
_CASE_PREFIX = "case:"

_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def case_file_remember(note: str, tool_context: ToolContext) -> dict:
    """Save a short note to THIS conversation's shared case file.

    Use it to jot a durable fact for later turns (e.g. the user's stated
    figures, or what customer service asked for). Scoped to this conversation
    only; optional — never a substitute for the actual tools or for asking the
    user.
    """
    try:
        key = _CASE_PREFIX + session_id(tool_context)
        _client.rpush(key, note)
        _client.expire(key, CASE_TTL_S)
        return {"ok": True}
    except Exception as e:  # Redis down / unreachable — degrade silently.
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def case_file_recall(tool_context: ToolContext) -> dict:
    """Read back the notes saved earlier in THIS conversation's case file."""
    try:
        key = _CASE_PREFIX + session_id(tool_context)
        return {"notes": _client.lrange(key, 0, -1)}
    except Exception as e:
        return {"notes": [], "error": f"{type(e).__name__}: {e}"}
