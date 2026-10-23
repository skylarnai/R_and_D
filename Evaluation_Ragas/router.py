"""FastAPI Router for RAGAS Evaluation.

Provides a single on-the-fly endpoint to evaluate questions against ground-truth
references using Ragas metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall).
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.embeddings import get_embeddings
from app.graph import get_graph
from app.vectorstore import similarity_search
from Evaluation_Ragas.config import get_eval_config
from Evaluation_Ragas.dataset import build_ragas_dataset
from Evaluation_Ragas.evaluate_ragas import RagasEvaluator

router = APIRouter(prefix="/evaluate", tags=["Evaluation RAGAS"])

# Singleton evaluator cache to avoid re-initializing weights / models on every call
_cached_evaluator: RagasEvaluator | None = None


def get_cached_evaluator() -> RagasEvaluator:
    global _cached_evaluator
    if _cached_evaluator is None:
        _cached_evaluator = RagasEvaluator(config=get_eval_config())
    return _cached_evaluator


class SingleEvalRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Test question to query the RAG pipeline.",
        example="What is the patient's Hemoglobin level?",
    )
    reference: str = Field(
        ...,
        min_length=3,
        description="Ground-truth clinical reference answer for scoring.",
        example="The patient's Hemoglobin is 14.5 g/dL (normal reference interval: 13.0 - 16.5 g/dL).",
    )
    session_id: UUID | None = Field(
        default=None,
        description="Session ID with indexed documents in PostgreSQL.",
    )
    top_k: int = Field(
        default=5,
        description="Top-K chunks to retrieve from pgvector.",
    )


@router.post("/single")
def evaluate_single_query_endpoint(request: SingleEvalRequest):
    """Run an on-the-fly RAG execution and Ragas scoring on a single custom question.

    Returns generated answer, retrieved context count, and the 4 core RAGAS metrics:
    - Faithfulness
    - Answer Relevance
    - Context Precision
    - Context Recall
    """
    cfg = get_eval_config()
    sid = request.session_id or UUID(cfg.default_session_id)
    graph = get_graph()

    # Step 1: Execute LangGraph CRAG pipeline (Retrieval + Grading + Generation)
    try:
        graph_output = graph.invoke(
            {
                "session_id": str(sid),
                "question": request.question,
                "chat_history": [],
                "documents": [],
                "use_web_search": False,
                "search_query": "",
                "web_results": [],
                "answer": "",
                "answer_source": "none",
                "has_uploaded_docs": True,
            }
        )
        answer = graph_output.get("answer", "")
        answer_source = graph_output.get("answer_source", "none")
        raw_docs = graph_output.get("documents", [])
        contexts = [d.content if hasattr(d, "content") else str(d) for d in raw_docs]

        # If CRAG filtered all docs or returned empty, perform fallback similarity search
        if not contexts:
            query_vector = get_embeddings().embed_query(request.question)
            chunks = similarity_search(sid, query_vector, request.top_k)
            contexts = [c.content for c in chunks]

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"LangGraph execution failed: {exc}"
        )

    # Step 2: Evaluate with Ragas Judge
    try:
        sample_record = [
            {
                "user_input": request.question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": request.reference,
            }
        ]
        ragas_dataset = build_ragas_dataset(sample_record)
        evaluator = get_cached_evaluator()
        eval_result = evaluator.evaluate_dataset(ragas_dataset, show_progress=False)

        sample_detail = (
            eval_result.samples_table[0] if eval_result.samples_table else {}
        )

        return {
            "status": "success",
            "session_id": str(sid),
            "question": request.question,
            "reference": request.reference,
            "generated_answer": answer,
            "answer_source": answer_source,
            "retrieved_contexts_count": len(contexts),
            "scores": {
                "faithfulness": sample_detail.get("faithfulness", 0.0),
                "answer_relevancy": sample_detail.get("answer_relevancy", 0.0),
                "context_precision": sample_detail.get("context_precision", 0.0),
                "context_recall": sample_detail.get("context_recall", 0.0),
            },
            "threshold_status": eval_result.threshold_results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ragas evaluation failed: {exc}")
