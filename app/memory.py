"""Per-session chat history stored in PostgreSQL."""
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import get_settings
from app.db import get_connection


def add_message(session_id: UUID, role: str, content: str) -> None:
    with get_connection(with_vector=False) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
            (str(session_id), role, content),
        )


def get_recent_messages(session_id: UUID) -> list[BaseMessage]:
    limit = get_settings().max_history_messages
    with get_connection(with_vector=False) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM chat_messages
                WHERE session_id = %s ORDER BY id DESC LIMIT %s
            ) recent ORDER BY id ASC
            """,
            (str(session_id), limit),
        )
        rows = cur.fetchall()
    return [
        HumanMessage(content=c) if r == "user" else AIMessage(content=c) for r, c in rows
    ]