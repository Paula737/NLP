"""
Generator Memory
-----------------
Stores ONLY what the Generator needs: conversation turns, retrieved
context, previous answers, and previous improvement attempts.

This class has no import of, or reference to, EvaluatorMemory, and it
writes to a distinct Redis key prefix ("gen_memory:*"). That is what
guarantees isolation between the two agents' memories.

If Redis is unreachable, memory falls back to an empty in-process dict
for that call (logged once) rather than crashing the request — memory
is for continuity across turns, not a hard requirement for answering.
"""
import json
import logging
import redis
from app.cache.redis_cache import cache

logger = logging.getLogger("evaluator_generator_rag.memory.generator")


class GeneratorMemory:
    PREFIX = "gen_memory"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.key = f"{self.PREFIX}:{session_id}"

    @staticmethod
    def _empty() -> dict:
        return {
            "conversation": [],          # [{question, answer}]
            "retrieved_context": [],     # context blob per turn
            "answers": [],               # flat list of generated answers
            "improvement_attempts": [],  # [{feedback, answer}]
        }

    def _load(self) -> dict:
        try:
            raw = cache.client.get(self.key)
            return json.loads(raw) if raw else self._empty()
        except redis.exceptions.RedisError as exc:
            logger.warning("Generator memory unavailable (%s); using empty memory.", exc)
            return self._empty()

    def _save(self, data: dict):
        try:
            cache.client.set(self.key, json.dumps(data))
        except redis.exceptions.RedisError as exc:
            logger.warning("Could not persist generator memory (%s).", exc)

    def add_turn(self, question: str, answer: str, context: str):
        data = self._load()
        data["conversation"].append({"question": question, "answer": answer})
        data["retrieved_context"].append(context)
        data["answers"].append(answer)
        self._save(data)

    def add_improvement_attempt(self, feedback: str, new_answer: str):
        data = self._load()
        data["improvement_attempts"].append({"feedback": feedback, "answer": new_answer})
        data["answers"].append(new_answer)
        self._save(data)

    def get_history(self) -> dict:
        return self._load()

    def clear(self):
        try:
            cache.client.delete(self.key)
        except redis.exceptions.RedisError as exc:
            logger.warning("Could not clear generator memory (%s).", exc)
