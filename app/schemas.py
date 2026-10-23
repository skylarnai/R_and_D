"""Pydantic models: API request/response bodies and internal data validation."""
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ---------- API ----------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = Field(
        default=None, description="Omit to start a new session; reuse to continue one."
    )
    stream: bool = Field(
        default=False,
        description="Set to true to stream tokens in real-time via Server-Sent Events (SSE).",
    )


class SourceInfo(BaseModel):
    type: Literal["document", "web"]
    name: str
    detail: str | None = None


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    answer_source: Literal["documents", "web", "none"]
    sources: list[SourceInfo] = []


class UploadResponse(BaseModel):
    session_id: UUID
    filename: str
    chunks_stored: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: str
    pgvector_installed: bool


# ---------- Internal ----------
class RetrievedChunk(BaseModel):
    filename: str
    chunk_index: int
    content: str
    score: float  # cosine similarity, higher = closer


class WebResult(BaseModel):
    title: str
    url: str
    content: str


class ChunkGrade(BaseModel):
    """CRAG grader verdict for a single retrieved chunk."""

    relevant: Literal["yes", "no"]

    @classmethod
    def from_llm_text(cls, text: str) -> "ChunkGrade":
        # Unclear output counts as "no" so doubtful chunks never reach the answer.
        verdict = "yes" if text.strip().lower().startswith("yes") else "no"
        return cls(relevant=verdict)