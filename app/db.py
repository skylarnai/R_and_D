"""PostgreSQL access with psycopg2: every call opens a connection and closes it."""
from contextlib import contextmanager
from uuid import UUID

import psycopg2
from pgvector.psycopg2 import register_vector

from app.config import get_settings


@contextmanager
def get_connection(with_vector: bool = True):
    """Open a connection, commit on success, roll back on error, always close."""
    s = get_settings()
    conn = psycopg2.connect(
        host=s.db_host,
        port=s.db_port,
        dbname=s.db_name,
        user=s.db_user,
        password=s.db_password,
    )
    try:
        if with_vector:
            register_vector(conn)  # lets psycopg2 send/receive pgvector values
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


REQUIRED_TABLES = ("sessions", "document_chunks", "chat_messages")


def init_db() -> None:
    """Verify that required existing tables and pgvector extension are present."""
    with get_connection(with_vector=False) as conn, conn.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        if not cur.fetchone()[0]:
            raise RuntimeError("pgvector extension is not installed in the database.")

        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )
        existing = {row[0] for row in cur.fetchall()}
        missing = [t for t in REQUIRED_TABLES if t not in existing]
        if missing:
            raise RuntimeError(
                f"Missing required existing tables in database: {', '.join(missing)}"
            )


def ensure_session(session_id: UUID) -> None:
    """Create the session row if it doesn't exist (needed by the foreign keys)."""
    with get_connection(with_vector=False) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (str(session_id),),
        )


def check_database() -> bool:
    """Run a trivial query and report whether the pgvector extension is installed."""
    with get_connection(with_vector=False) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1;")
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        return bool(cur.fetchone()[0])