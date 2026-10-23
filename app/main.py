"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload"""
import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.db import check_database, ensure_session, init_db
from app.embeddings import get_embeddings
from app.ingestion import ingest_file
from app.schemas import ChatRequest, ChatResponse, HealthResponse, UploadResponse
from app.service import answer_question, stream_answer_question
from app.vectorstore import get_latest_session_with_docs

logger = logging.getLogger("rag_chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # fail fast if a required env variable is missing
    init_db()
    get_embeddings()  # load EmbeddingGemma once at startup
    yield


app = FastAPI(title="RAG Chatbot (Gemma + pgvector + LangGraph)", lifespan=lifespan)


@app.get("/health/db", response_model=HealthResponse)
def health_db():
    try:
        pgvector_ok = check_database()
    except Exception as exc:
        logger.exception("Database health check failed")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    return HealthResponse(
        status="ok", database="connected", pgvector_installed=pgvector_ok
    )


@app.post("/upload", response_model=UploadResponse)
def upload(
    file: UploadFile = File(...),
    session_id: UUID | None = Form(default=None),
):
    """Upload a PDF/TXT/MD/DOCX/CSV. Omit session_id to start a new session."""
    session_id = session_id or uuid4()
    ensure_session(session_id)
    data = file.file.read()

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {get_settings().max_upload_mb} MB limit.")

    try:
        count = ingest_file(session_id, file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Failed to process the file.")

    return UploadResponse(
        session_id=session_id, filename=file.filename or "upload", chunks_stored=count
    )


@app.post("/chat")
def chat(request: ChatRequest):
    """Ask a question. If request.stream is True, streams SSE tokens. Otherwise returns ChatResponse."""
    session_id = request.session_id or get_latest_session_with_docs() or uuid4()
    ensure_session(session_id)
    try:
        if request.stream:
            return StreamingResponse(
                stream_answer_question(session_id, request.question),
                media_type="text/event-stream",
            )
        return answer_question(session_id, request.question)
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="Failed to generate a response.")


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Dedicated endpoint to stream tokens live via Server-Sent Events (SSE)."""
    session_id = request.session_id or get_latest_session_with_docs() or uuid4()
    ensure_session(session_id)
    try:
        return StreamingResponse(
            stream_answer_question(session_id, request.question),
            media_type="text/event-stream",
        )
    except Exception:
        logger.exception("Chat stream request failed")
        raise HTTPException(status_code=500, detail="Failed to stream response.")


# Mount RAGAS Evaluation Endpoints
from Evaluation_Ragas.router import router as eval_router

app.include_router(eval_router)