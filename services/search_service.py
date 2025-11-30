"""
Search service: Full‑text search (FTS) for chat messages and conversations.
Hybrid vector search to be added after pgvector migration.
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text


class SearchService:
    def __init__(self, db: Session):
        self.db = db

    def search_messages(
        self,
        user_id: int,
        query: str,
        conversation_id: Optional[int] = None,
        role: Optional[str] = None,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """FTS over chat_message.content scoped to user and optional filters."""
        sql = [
            """
            SELECT m.message_id, m.conversation_id, m.role, m.content, m.created_at,
                   ts_rank_cd(m.search_vector, plainto_tsquery('english', :q)) AS rank
            FROM chat_message m
            JOIN chat_conversation c ON c.conversation_id = m.conversation_id
            WHERE c.user_id = :user_id
              AND m.search_vector @@ plainto_tsquery('english', :q)
            """
        ]
        params: Dict[str, Any] = {"user_id": user_id, "q": query}

        if conversation_id is not None:
            sql.append("AND m.conversation_id = :cid")
            params["cid"] = conversation_id

        if role in ("user", "assistant"):
            sql.append("AND m.role = :role")
            params["role"] = role

        if start_iso:
            sql.append("AND m.created_at >= :start_dt")
            params["start_dt"] = start_iso
        if end_iso:
            sql.append("AND m.created_at <= :end_dt")
            params["end_dt"] = end_iso

        sql.append("ORDER BY rank DESC, m.created_at DESC")
        sql.append("LIMIT :limit OFFSET :offset")
        params.update({"limit": limit, "offset": offset})

        rows = self.db.execute(text("\n".join(sql)), params).mappings().all()
        return [dict(r) for r in rows]

    def search_conversations(
        self,
        user_id: int,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Simple search over conversation title and first message content."""
        # Search in conversation titles first
        conv_rows = self.db.execute(
            text(
                """
                SELECT c.conversation_id, c.title, c.created_at, c.updated_at
                FROM chat_conversation c
                WHERE c.user_id = :user_id
                  AND (c.title ILIKE :like OR :like = '')
                ORDER BY c.updated_at DESC
                LIMIT :limit OFFSET :offset
                """
            ),
            {"user_id": user_id, "like": f"%{query}%", "limit": limit, "offset": offset},
        ).mappings().all()
        return [dict(r) for r in conv_rows]

    def export_conversation(self, user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
        rows = self.db.execute(
            text(
                """
                SELECT m.message_id, m.role, m.content, m.created_at
                FROM chat_message m
                JOIN chat_conversation c ON c.conversation_id = m.conversation_id
                WHERE c.user_id = :user_id AND m.conversation_id = :cid
                ORDER BY m.created_at ASC
                """
            ),
            {"user_id": user_id, "cid": conversation_id},
        ).mappings().all()
        return [dict(r) for r in rows]


