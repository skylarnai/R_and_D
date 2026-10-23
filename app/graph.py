"""STEP 2 - the LangGraph workflow (CRAG).

    START -> retrieve -> grade_documents --(relevant chunks found)--> generate -> END
                                  |
                                  +--(none relevant)--> rewrite_query -> web_search -> generate -> END

Build order: state (TypedDict) -> nodes -> edges -> compile.
"""
from functools import lru_cache
from typing import Literal, TypedDict
from uuid import UUID

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from app.chains import (
    format_chunks,
    format_web_results,
    get_generate_chain,
    get_grader_chain,
    get_rewrite_chain,
)
from app.config import get_settings
from app.embeddings import get_embeddings
from app.schemas import RetrievedChunk, WebResult
from app.vectorstore import has_documents, similarity_search
from app.websearch import search_web


# ---------------- 1. State ----------------
class GraphState(TypedDict):
    session_id: str
    question: str
    chat_history: list[BaseMessage]
    documents: list[RetrievedChunk]  # retrieved, then filtered by the grader
    use_web_search: bool
    search_query: str
    web_results: list[WebResult]
    answer: str
    answer_source: Literal["documents", "web", "none"]
    has_uploaded_docs: bool


# ---------------- 2. Nodes ----------------
def retrieve_node(state: GraphState) -> dict:
    """Embed the question and run semantic search in pgvector (this session only)."""
    s = get_settings()
    sid = UUID(state["session_id"])
    docs_exist = has_documents(sid)
    if not docs_exist:
        return {"documents": [], "has_uploaded_docs": False}

    query_vector = get_embeddings().embed_query(state["question"])
    chunks = similarity_search(sid, query_vector, s.top_k)
    chunks = [c for c in chunks if c.score >= s.similarity_threshold]
    return {"documents": chunks, "has_uploaded_docs": True}


GREETING_WORDS = {
    "hello", "hi", "hey", "how are you", "good morning",
    "good afternoon", "good evening", "greetings", "help", "who are you"
}

GREETING_STARTERS = ("hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening")


def is_conversational(text: str) -> bool:
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()
    words = cleaned.split()
    if not words:
        return True
    if cleaned in GREETING_WORDS:
        return True
    if any(cleaned.startswith(prefix) for prefix in GREETING_STARTERS) and len(words) <= 8:
        return True
    return False


def grade_documents_node(state: GraphState) -> dict:
    """CRAG: keep only chunks the LLM grader marks relevant; flag web search if none."""
    docs = state["documents"]
    if not docs:
        return {"documents": [], "use_web_search": not is_conversational(state["question"])}

    if is_conversational(state["question"]):
        return {"documents": [], "use_web_search": False}

    inputs = [{"question": state["question"], "chunk": d.content} for d in docs]
    grades = get_grader_chain().batch(inputs, config={"max_concurrency": 2})
    relevant = [d for d, g in zip(docs, grades) if g.relevant == "yes"]
    return {"documents": relevant, "use_web_search": len(relevant) == 0}


def rewrite_query_node(state: GraphState) -> dict:
    query = get_rewrite_chain().invoke(
        {"question": state["question"], "chat_history": state["chat_history"]}
    )
    return {"search_query": query.strip() or state["question"]}


def web_search_node(state: GraphState) -> dict:
    query = state.get("search_query", "")
    if "NON_MEDICAL" in query.upper():
        return {"web_results": []}
    return {"web_results": search_web(query)}


def generate_node(state: GraphState) -> dict:
    """Send context + chat history + question through the ChatPromptTemplate chain."""
    query = state.get("search_query", "")
    if "NON_MEDICAL" in query.upper():
        return {
            "answer": "I am a medical chatbot. I can only assist with medical, healthcare, and health report-related questions. Please feel free to ask me questions related to medicine or your uploaded health documents.",
            "answer_source": "none",
        }

    if state["documents"]:
        context = format_chunks(state["documents"])
        origin, source = "the user's uploaded document", "documents"
        answer = get_generate_chain().invoke(
            {
                "context": context,
                "context_origin": origin,
                "chat_history": state["chat_history"],
                "question": state["question"],
            }
        )
        return {"answer": answer, "answer_source": source}

    elif state["web_results"]:
        context = format_web_results(state["web_results"])
        origin, source = "reference medical sources from the internet", "web"
        raw_answer = get_generate_chain().invoke(
            {
                "context": context,
                "context_origin": origin,
                "chat_history": state["chat_history"],
                "question": state["question"],
            }
        )

        if state.get("has_uploaded_docs", False):
            answer = (
                "The specific information you asked for is not available in the document you shared. "
                "However, I retrieved reference medical information from the internet for you:\n\n"
                f"{raw_answer}"
            )
        else:
            answer = raw_answer

        return {"answer": answer, "answer_source": source}

    elif is_conversational(state["question"]):
        context = "The user is engaging in conversational greeting. Acknowledge and offer medical assistance."
        origin, source = "conversational greeting", "none"
        answer = get_generate_chain().invoke(
            {
                "context": context,
                "context_origin": origin,
                "chat_history": state["chat_history"],
                "question": state["question"],
            }
        )
        return {"answer": answer, "answer_source": source}

    else:
        if state.get("has_uploaded_docs", False):
            msg = "The specific information you asked for is not available in the document you shared, and no relevant medical references were found on the web."
        else:
            msg = "I couldn't find relevant information on the web. Please let me know what specific health question you would like help with."
        return {
            "answer": msg,
            "answer_source": "none",
        }


# ---------------- 3. Conditional edge ----------------
def route_after_grading(state: GraphState) -> Literal["rewrite_query", "generate"]:
    if is_conversational(state["question"]):
        return "generate"
    return "rewrite_query" if state["use_web_search"] else "generate"


# ---------------- 4. Build + compile ----------------
@lru_cache
def get_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")  # first node to run
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()