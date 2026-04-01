"""Chat persistence — stores and retrieves conversations in SQLite."""

import json
import os
import sqlite3
import uuid
from datetime import datetime

DB_PATH = os.getenv("CHAT_DB_PATH", "data/chats.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            trace TEXT,
            tokens INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def create_conversation(title: str | None = None) -> str:
    """Create a new conversation and return its ID."""
    conv_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    title = title or f"Chat {now[:16]}"
    conn = _connect()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return conv_id


def save_message(conversation_id: str, role: str, content: str,
                 trace: list[str] | None = None, tokens: int = 0):
    """Append a message to a conversation."""
    now = datetime.now().isoformat()
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, trace, tokens, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (conversation_id, role, content, json.dumps(trace) if trace else None, tokens, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    conn.close()


def load_messages(conversation_id: str) -> list[dict]:
    """Load all messages for a conversation."""
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, trace, tokens FROM messages "
        "WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    conn.close()
    messages = []
    for role, content, trace_json, tokens in rows:
        msg = {"role": role, "content": content}
        if trace_json:
            msg["trace"] = json.loads(trace_json)
        if tokens:
            msg["tokens"] = tokens
        messages.append(msg)
    return messages


def list_conversations() -> list[dict]:
    """List all conversations, most recent first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT c.id, c.title, c.updated_at, COUNT(m.id) as msg_count "
        "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
        "GROUP BY c.id ORDER BY c.updated_at DESC",
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "updated_at": r[2], "message_count": r[3]}
        for r in rows
    ]


def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def update_title(conversation_id: str, title: str):
    """Update a conversation's title."""
    conn = _connect()
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (title, conversation_id),
    )
    conn.commit()
    conn.close()
