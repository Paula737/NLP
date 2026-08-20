"""
LLM Provider Factory
----------------------
Single place that decides which chat model / embedding model backs the
whole platform. Everything else (generator, evaluator, vector store)
calls these two functions instead of importing a specific provider
directly, so switching providers is a one-line .env change.

Chat provider:  settings.llm_provider -> "groq" | "ollama" | "openai"
Embedding provider: settings.embedding_provider -> "local" | "openai"
(kept independent because Groq has no embeddings endpoint)
"""
from app.config import settings


def get_chat_model(temperature: float = 0.2, json_mode: bool = False):
    provider = settings.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq
        kwargs = {
            "model": settings.groq_model,
            "api_key": settings.groq_api_key,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = {
            "model": settings.ollama_model,
            "base_url": settings.ollama_base_url,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["format"] = "json"  # Ollama's native structured-output mode
        return ChatOllama(**kwargs)

    from langchain_openai import ChatOpenAI
    kwargs = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


def get_embeddings():
    provider = settings.embedding_provider

    if provider == "local":
        # Free, runs on CPU, no API key or signup needed.
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=f"sentence-transformers/{settings.local_embedding_model}")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
