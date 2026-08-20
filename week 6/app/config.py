import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # "ollama" (free, local), "groq" (free tier, cloud, fast), or "openai" (paid)
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")

    # Embeddings are independent of the chat provider (Groq has no embeddings API):
    # "local" (free, sentence-transformers, no signup) or "openai"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")

    # OpenAI settings (used when llm_provider/embedding_provider == "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Groq settings (used when llm_provider == "groq")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Ollama settings (used when llm_provider == "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    # Local embeddings (used when embedding_provider == "local")
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    chroma_persist_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    max_loops: int = int(os.getenv("MAX_LOOPS", "4"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "3600"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))


settings = Settings()
