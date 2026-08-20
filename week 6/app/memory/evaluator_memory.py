"""
Evaluator Memory
-----------------
Stores ONLY what the Evaluator needs: evaluation history, feedback
history, and decisions. Completely separate class, separate Redis key
prefix ("eval_memory:*") — no shared state with GeneratorMemory.

Falls back to an empty in-process dict if Redis is unreachable, so
evaluation can still proceed without persisted history.
"""
import json
import logging
import redis
from app.cache.redis_cache import cache

logger = logging.getLogger("evaluator_generator_rag.memory.evaluator")


class EvaluatorMemory:
    PREFIX = "eval_memory"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.key = f"{self.PREFIX}:{session_id}"

    @staticmethod
    def _empty() -> dict:
        return {
            "evaluation_history": [],  # [{question, answer, decision, score, feedback}]
            "feedback_history": [],    # flat list of feedback strings
            "decisions": [],           # flat list of "accept" / "revise"
        }

    def _load(self) -> dict:
        try:
            raw = cache.client.get(self.key)
            return json.loads(raw) if raw else self._empty()
        except redis.exceptions.RedisError as exc:
            logger.warning("Evaluator memory unavailable (%s); using empty memory.", exc)
            return self._empty()

    def _save(self, data: dict):
        try:
            cache.client.set(self.key, json.dumps(data))
        except redis.exceptions.RedisError as exc:
            logger.warning("Could not persist evaluator memory (%s).", exc)

    def add_evaluation(self, question: str, answer: str, result: dict):
        data = self._load()
        entry = {
            "question": question,
            "answer": answer,
            "decision": result.get("decision"),
            "score": result.get("score"),
            "feedback": result.get("feedback"),
        }
        data["evaluation_history"].append(entry)
        data["decisions"].append(result.get("decision"))
        if result.get("feedback"):
            data["feedback_history"].append(result.get("feedback"))
        self._save(data)

    def get_history(self) -> dict:
        return self._load()

    def clear(self):
        try:
            cache.client.delete(self.key)
        except redis.exceptions.RedisError as exc:
            logger.warning("Could not clear evaluator memory (%s).", exc)
