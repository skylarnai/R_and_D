# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A medical/clinical RAG chatbot exposed as a FastAPI service. Users upload health documents (lab reports etc.). Questions are answered from those documents through a Corrective RAG (CRAG) pipeline, and the service falls back to Tavily web search when no uploaded chunk is relevant. Non-medical questions are declined.

Stack: FastAPI · LangChain (LCEL) · LangGraph · PostgreSQL + pgvector (raw psycopg2, no ORM) · Gemma chat model via Google AI Studio (`langchain-google-genai`) · local EmbeddingGemma (`sentence-transformers`) · LangSmith tracing · RAGAS evaluation.

## Commands

The populated virtualenv is `venv/` (`.venv/` is empty). Run everything from the repo root, because imports are absolute (`app.…`, `Evaluation_Ragas.…`).

```powershell
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload        # serves on http://127.0.0.1:8000, docs at /docs
```

There is no test suite, linter config or build step. To exercise the system, call the endpoints: `GET /health/db`, `POST /upload` (multipart `file`, optional `session_id`), `POST /chat`, `POST /chat/stream` (SSE), and `POST /evaluate/single` (RAGAS scoring of one question against a ground-truth `reference`; see `Evaluation_Ragas/README.md`).

## Required environment

`.env` at the repo root, loaded by `app/config.py` and exported into `os.environ` so the SDKs pick up their keys. Required: `GOOGLE_API_KEY`, `TAVILY_API_KEY`. Optional: `HF_TOKEN` (EmbeddingGemma is a gated HF model), `LANGSMITH_*`, `DB_*`. `config.py` mirrors `LANGSMITH_*` into `LANGCHAIN_*` env vars.

**The database schema is not created by the app.** On startup, `init_db()` only *verifies* that the `vector` extension and the tables `sessions`, `document_chunks` and `chat_messages` already exist, and it raises if they don't. Columns the code depends on:
- `sessions(session_id)`
- `document_chunks(id, session_id, filename, chunk_index, content, embedding vector(768))`
- `chat_messages(id, session_id, role, content)`

## Architecture

All tunables (model names, chunk size, `top_k`, `similarity_threshold`, history length, upload limit) live in `Settings` in `app/config.py`. Heavy objects (settings, LLM per temperature, embeddings model, chains, compiled graph, Tavily tool) are `@lru_cache` singletons.

**Ingestion** (`ingestion.py`): extract text (pdf/docx/txt/md/csv; CSV rows become `col: val; …` lines) → `RecursiveCharacterTextSplitter` → `EmbeddingGemma.embed_documents` → bulk insert into `document_chunks`. Everything is scoped by `session_id`.

**Chains** (`chains.py`): three LCEL chains (`prompt | llm | StrOutputParser`): the grader (yes/no → `ChunkGrade`; ambiguous output counts as "no"), the rewriter (turns the question into a web query, or returns the sentinel `NON_MEDICAL`), and the generator (the clinical-assistant system prompt, filled with `{context}` / `{context_origin}`).

**CRAG graph** (`graph.py`): `retrieve → grade_documents → (generate | rewrite_query → web_search → generate)`. Retrieval is cosine search in pgvector (`vectorstore.py`, `1 - (embedding <=> q)`), restricted to the session and pre-filtered by `similarity_threshold` before LLM grading. `is_conversational()` is a keyword heuristic that short-circuits greetings past grading and web search.

**Two parallel pipelines — keep them in sync.** `service.answer_question` (non-streaming `/chat`) runs the LangGraph graph. `service.stream_answer_question` (`/chat/stream`, `/chat` with `stream: true`) does **not** use the graph. It re-implements the same retrieve → grade → rewrite/web → generate logic inline so it can stream tokens from `get_generate_chain().stream(...)`. A change to routing, fallback messages, the NON_MEDICAL handling or the "not in your document, but from the web…" prefix has to be made in both `graph.py` and `service.py`.

SSE events emitted by the streaming path: `sources`, then a series of `token`, then `done`.

**Session handling**: when `session_id` is omitted, `/chat` uses the most recently uploaded session (`get_latest_session_with_docs`), or a new UUID if there is none. Chat history is the last `max_history_messages` rows in `chat_messages`, converted to `HumanMessage`/`AIMessage`.

**Evaluation** (`Evaluation_Ragas/`): the router is mounted into the main app at the bottom of `app/main.py`. `/evaluate/single` invokes the graph, falls back to raw similarity search for contexts if CRAG filtered everything, then scores with RAGAS (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall). The same Gemma LLM (temperature 0) is the judge and EmbeddingGemma provides the embeddings. Thresholds and a hard-coded default `session_id` are in `Evaluation_Ragas/config.py`.
