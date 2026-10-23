"""pgvector storage and semantic (cosine) search, always scoped to one session."""
from uuid import UUID

import numpy as np
from langsmith import traceable
from psycopg2.extras import execute_values

from app.db import get_connection
from app.schemas import RetrievedChunk


def add_chunks(
    session_id: UUID, filename: str, chunks: list[str], vectors: list[list[float]]
) -> int:
    rows = [
        (str(session_id), filename, i, text, np.array(vec, dtype=np.float32))
        for i, (text, vec) in enumerate(zip(chunks, vectors))
    ]
    with get_connection() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO document_chunks (session_id, filename, chunk_index, content, embedding) VALUES %s",
            rows,
        )
    return len(rows)


@traceable(name="pgvector_semantic_search", run_type="retriever")
def similarity_search(
    session_id: UUID, query_vector: list[float], top_k: int
) -> list[RetrievedChunk]:
    qv = np.array(query_vector, dtype=np.float32)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT filename, chunk_index, content, 1 - (embedding <=> %s) AS score
            FROM document_chunks
            WHERE session_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qv, str(session_id), qv, top_k),
        )
        return [
            RetrievedChunk(filename=f, chunk_index=i, content=c, score=float(s))
            for f, i, c, s in cur.fetchall()
        ]


def has_documents(session_id: UUID) -> bool:
    """Check if the session has any uploaded document chunks."""
    with get_connection(with_vector=False) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM document_chunks WHERE session_id = %s LIMIT 1);",
            (str(session_id),),
        )
        return bool(cur.fetchone()[0])


def get_latest_session_with_docs() -> UUID | None:
    """Find the most recent session ID that has document chunks uploaded."""
    with get_connection(with_vector=False) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT session_id FROM document_chunks ORDER BY id DESC LIMIT 1;"
        )
        row = cur.fetchone()
        if row:
            return UUID(str(row[0]))
    return None