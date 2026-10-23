import os
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Exports .env values into os.environ so the LangChain, LangSmith, Tavily and
# Hugging Face SDKs can find their own keys (GOOGLE_API_KEY, LANGSMITH_*, ...).
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else find_dotenv())

# Synchronize LangChain and LangSmith env vars so all components trace to the same project
if os.environ.get("LANGSMITH_TRACING") == "true" and not os.environ.get("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.environ.get("LANGSMITH_PROJECT") and not os.environ.get("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
if os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    google_api_key: str
    tavily_api_key: str
    hf_token: str | None = None

    # LangSmith
    langsmith_tracing: bool = True
    langsmith_api_key: str | None = None
    langsmith_project: str = "rag_chatbot"

    # PostgreSQL
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ragdb"
    db_user: str = "postgres"
    db_password: str = "postgres"

    # Models
    chat_model: str = "gemma-4-26b-a4b-it"  # Fast MoE Gemma 4 on Google AI Studio
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_dim: int = 768

    # RAG tuning
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    similarity_threshold: float = 0.30  # cheap pre-filter before the CRAG grader
    max_history_messages: int = 10
    max_upload_mb: int = 100
    web_results: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()