"""EmbeddingGemma wrapped as a LangChain Embeddings class (runs locally)."""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingGemma(Embeddings):
    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # encode_document / encode_query apply EmbeddingGemma's own task prompts.
        vectors = self._model.encode_document(
            texts, normalize_embeddings=True, batch_size=16
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode_query(text, normalize_embeddings=True).tolist()


@lru_cache
def get_embeddings() -> EmbeddingGemma:
    return EmbeddingGemma(get_settings().embedding_model)