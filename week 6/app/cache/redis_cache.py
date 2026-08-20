"""
Redis Caching Layer
--------------------
Central cache used across the platform. Every cacheable thing (documents,
embeddings, LLM responses, evaluations) is namespaced so that different
kinds of data never collide, and so a namespace can be invalidated on its
own if the underlying knowledge changes (avoids stale results).

Caching is a performance optimization, not a correctness requirement, so
every method here degrades gracefully if Redis is unreachable: a get()
returns None (cache miss) and a set() is a silent no-op, with a single
warning logged instead of raising. This means ingestion and Q&A keep
working even if Redis is down or not started yet.
"""
import json
import hashlib
import logging
import redis
from app.config import settings

logger = logging.getLogger("evaluator_generator_rag.cache")


class RedisCache:
    def __init__(self, url: str = None, ttl: int = None):
        self.client = redis.Redis.from_url(
            url or settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        self.default_ttl = ttl or settings.cache_ttl_seconds
        self._warned = False

    def _warn_once(self, exc: Exception):
        if not self._warned:
            logger.warning(
                "Redis unavailable (%s) — continuing without caching. "
                "Start Redis (e.g. `docker compose up -d`) to enable it.",
                exc,
            )
            self._warned = True

    @staticmethod
    def _hash(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _key(self, namespace: str, payload: str) -> str:
        return f"{namespace}:{self._hash(payload)}"

    def get(self, namespace: str, payload: str):
        try:
            raw = self.client.get(self._key(namespace, payload))
            return json.loads(raw) if raw is not None else None
        except redis.exceptions.RedisError as exc:
            self._warn_once(exc)
            return None

    def set(self, namespace: str, payload: str, value, ttl: int = None):
        try:
            self.client.set(self._key(namespace, payload), json.dumps(value), ex=ttl or self.default_ttl)
        except redis.exceptions.RedisError as exc:
            self._warn_once(exc)

    def invalidate_namespace(self, namespace: str):
        """Wipe a whole cache category, e.g. when a document is re-ingested."""
        try:
            for key in self.client.scan_iter(f"{namespace}:*"):
                self.client.delete(key)
        except redis.exceptions.RedisError as exc:
            self._warn_once(exc)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.exceptions.RedisError:
            return False


# Shared singleton used across the app
cache = RedisCache()

# Namespaces (keep these consistent everywhere they're used)
NS_DOCUMENT = "doc"
NS_EMBEDDING = "embedding"
NS_LLM_RESPONSE = "llm_response"
NS_EVALUATION = "evaluation"
