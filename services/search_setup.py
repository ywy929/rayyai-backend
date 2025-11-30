"""
FTS setup for chat messages (PostgreSQL tsvector + GIN index).
Creates column, trigger, and index if not present.
"""
from sqlalchemy import text
from sqlalchemy.engine import Engine


def ensure_chat_message_fts(engine: Engine) -> None:
    """Ensure tsvector column, trigger, and index exist for chat_message."""
    with engine.connect() as conn:
        # Add search_vector column if missing
        conn.execute(text(
            """
            ALTER TABLE chat_message
            ADD COLUMN IF NOT EXISTS search_vector tsvector;
            """
        ))

        # Create trigger to keep tsvector in sync with content
        conn.execute(text(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_message_fts
            ON chat_message USING GIN (search_vector);
            """
        ))

        # Use built-in tsvector_update_trigger
        conn.execute(text(
            """
            CREATE TRIGGER chat_message_tsv_update
            BEFORE INSERT OR UPDATE ON chat_message
            FOR EACH ROW EXECUTE FUNCTION tsvector_update_trigger(
                'search_vector', 'pg_catalog.english', 'content'
            );
            """
        ))

        # Backfill existing rows
        conn.execute(text(
            """
            UPDATE chat_message
            SET search_vector = to_tsvector('pg_catalog.english', coalesce(content,''))
            WHERE search_vector IS NULL;
            """
        ))

        conn.commit()


