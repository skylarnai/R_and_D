"""Chat service: loads history, runs RAG pipeline, saves turns, and supports streaming."""
from __future__ import annotations

import json
from typing import Generator
from uuid import UUID

from langsmith import traceable

from app.chains import (
    format_chunks,
    format_web_results,
    get_generate_chain,
    get_grader_chain,
    get_rewrite_chain,
)
from app.config import get_settings
from app.embeddings import get_embeddings
from app.graph import get_graph, is_conversational
from app.memory import add_message, get_recent_messages
from app.schemas import ChatResponse, SourceInfo
from app.vectorstore import has_documents, similarity_search
from app.websearch import search_web


@traceable(name="rag_chat", run_type="chain")
def answer_question(session_id: UUID, question: str) -> ChatResponse:
    """Non-streaming execution: returns the full response object."""
    history = get_recent_messages(session_id)

    result = get_graph().invoke(
        {
            "session_id": str(session_id),
            "question": question,
            "chat_history": history,
            "documents": [],
            "use_web_search": False,
            "search_query": "",
            "web_results": [],
            "answer": "",
            "answer_source": "none",
            "has_uploaded_docs": False,
        }
    )

    add_message(session_id, "user", question)
    add_message(session_id, "assistant", result["answer"])

    if result["answer_source"] == "documents":
        sources = [
            SourceInfo(
                type="document",
                name=d.filename,
                detail=f"chunk {d.chunk_index}, similarity {d.score:.2f}",
            )
            for d in result["documents"]
        ]
    elif result["answer_source"] == "web":
        sources = [
            SourceInfo(type="web", name=r.title, detail=r.url)
            for r in result["web_results"]
        ]
    else:
        sources = []

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        answer_source=result["answer_source"],
        sources=sources,
    )


def stream_answer_question(session_id: UUID, question: str) -> Generator[str, None, None]:
    """Streaming execution: yields Server-Sent Events (SSE) token-by-token in real-time."""
    history = get_recent_messages(session_id)
    add_message(session_id, "user", question)

    s = get_settings()
    docs_exist = has_documents(session_id)

    # 1. Determine Context and Sources
    if is_conversational(question):
        context = "The user is engaging in conversational greeting. Acknowledge and offer medical assistance."
        origin, source = "conversational greeting", "none"
        sources: list[SourceInfo] = []
    else:
        chunks = []
        if docs_exist:
            query_vector = get_embeddings().embed_query(question)
            retrieved = similarity_search(session_id, query_vector, s.top_k)
            chunks = [c for c in retrieved if c.score >= s.similarity_threshold]

        relevant = []
        if chunks:
            inputs = [{"question": question, "chunk": d.content} for d in chunks]
            grades = get_grader_chain().batch(inputs, config={"max_concurrency": 2})
            relevant = [d for d, g in zip(chunks, grades) if g.relevant == "yes"]

        if relevant:
            context = format_chunks(relevant)
            origin, source = "the user's uploaded document", "documents"
            sources = [
                SourceInfo(
                    type="document",
                    name=d.filename,
                    detail=f"chunk {d.chunk_index}, similarity {d.score:.2f}",
                )
                for d in relevant
            ]
        else:
            # Fallback to web search
            query = get_rewrite_chain().invoke({"question": question, "chat_history": history})
            if "NON_MEDICAL" in query.upper():
                decline_msg = (
                    "I am a medical chatbot. I can only assist with medical, healthcare, "
                    "and health report-related questions. Please feel free to ask me questions "
                    "related to medicine or your uploaded health documents."
                )
                add_message(session_id, "assistant", decline_msg)
                yield f"data: {json.dumps({'event': 'token', 'token': decline_msg})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'session_id': str(session_id), 'answer_source': 'none', 'sources': []})}\n\n"
                return

            web_results = search_web(query)
            if web_results:
                context = format_web_results(web_results)
                origin, source = "reference medical sources from the internet", "web"
                sources = [SourceInfo(type="web", name=r.title, detail=r.url) for r in web_results]
            else:
                context = ""
                origin, source = "none", "none"
                sources = []

    # 2. Emit initial metadata with resolved sources
    yield f"data: {json.dumps({'event': 'sources', 'session_id': str(session_id), 'source': source, 'sources': [src.model_dump() for src in sources]})}\n\n"

    # 3. Handle no-info scenario
    if not context and source == "none" and not is_conversational(question):
        no_info_msg = (
            "The specific information you asked for is not available in the document you shared, and no relevant medical references were found on the web."
            if docs_exist
            else "I couldn't find relevant information on the web. Please let me know what specific health question you would like help with."
        )
        add_message(session_id, "assistant", no_info_msg)
        yield f"data: {json.dumps({'event': 'token', 'token': no_info_msg})}\n\n"
        yield f"data: {json.dumps({'event': 'done', 'session_id': str(session_id), 'answer_source': 'none', 'sources': []})}\n\n"
        return

    # 4. Optional web-search prefix for document sessions
    prefix = ""
    if source == "web" and docs_exist:
        prefix = (
            "The specific information you asked for is not available in the document you shared. "
            "However, I retrieved reference medical information from the internet for you:\n\n"
        )
        yield f"data: {json.dumps({'event': 'token', 'token': prefix})}\n\n"

    # 5. Stream LLM tokens live from Gemma 4 via get_generate_chain().stream(...)
    accumulated = [prefix]
    for token in get_generate_chain().stream(
        {
            "context": context,
            "context_origin": origin,
            "chat_history": history,
            "question": question,
        }
    ):
        accumulated.append(token)
        yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"

    # 6. Save final completed answer to persistent chat memory
    full_answer = "".join(accumulated)
    add_message(session_id, "assistant", full_answer)

    # 7. Final completion event
    yield f"data: {json.dumps({'event': 'done', 'session_id': str(session_id), 'answer_source': source, 'sources': [src.model_dump() for src in sources]})}\n\n"