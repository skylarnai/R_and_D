"""Chat model: Gemma 4 31B served through the Gemini API, used via LangChain."""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.chat_model,
        google_api_key=s.google_api_key,
        temperature=temperature,
        max_retries=5,
    )