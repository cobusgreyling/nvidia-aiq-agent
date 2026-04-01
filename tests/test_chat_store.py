"""Tests for chat persistence — SQLite conversation store."""

from unittest.mock import patch


class TestChatStore:
    """Test conversation CRUD operations."""

    def test_create_and_list_conversations(self, tmp_path):
        db_path = str(tmp_path / "test_chats.db")
        with patch("chat_store.DB_PATH", db_path):
            from chat_store import create_conversation, list_conversations

            conv_id = create_conversation("Test Chat")
            assert conv_id is not None
            assert len(conv_id) == 12

            convs = list_conversations()
            assert len(convs) == 1
            assert convs[0]["title"] == "Test Chat"
            assert convs[0]["id"] == conv_id

    def test_save_and_load_messages(self, tmp_path):
        db_path = str(tmp_path / "test_chats.db")
        with patch("chat_store.DB_PATH", db_path):
            from chat_store import create_conversation, save_message, load_messages

            conv_id = create_conversation("Test Chat")
            save_message(conv_id, "user", "Hello")
            save_message(conv_id, "assistant", "Hi there!", trace=["Step 1"], tokens=100)

            messages = load_messages(conv_id)
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["trace"] == ["Step 1"]
            assert messages[1]["tokens"] == 100

    def test_delete_conversation(self, tmp_path):
        db_path = str(tmp_path / "test_chats.db")
        with patch("chat_store.DB_PATH", db_path):
            from chat_store import (
                create_conversation, save_message,
                delete_conversation, list_conversations, load_messages,
            )

            conv_id = create_conversation("To Delete")
            save_message(conv_id, "user", "test msg")
            delete_conversation(conv_id)

            assert list_conversations() == []
            assert load_messages(conv_id) == []

    def test_update_title(self, tmp_path):
        db_path = str(tmp_path / "test_chats.db")
        with patch("chat_store.DB_PATH", db_path):
            from chat_store import create_conversation, update_title, list_conversations

            conv_id = create_conversation("Old Title")
            update_title(conv_id, "New Title")

            convs = list_conversations()
            assert convs[0]["title"] == "New Title"

    def test_multiple_conversations_ordered(self, tmp_path):
        db_path = str(tmp_path / "test_chats.db")
        with patch("chat_store.DB_PATH", db_path):
            from chat_store import create_conversation, save_message, list_conversations
            import time

            create_conversation("First")
            time.sleep(0.01)
            id2 = create_conversation("Second")
            save_message(id2, "user", "msg")

            convs = list_conversations()
            assert len(convs) == 2
            # Most recent first
            assert convs[0]["id"] == id2

    def test_message_count_in_listing(self, tmp_path):
        db_path = str(tmp_path / "test_chats.db")
        with patch("chat_store.DB_PATH", db_path):
            from chat_store import create_conversation, save_message, list_conversations

            conv_id = create_conversation("Count Test")
            save_message(conv_id, "user", "msg1")
            save_message(conv_id, "assistant", "msg2")
            save_message(conv_id, "user", "msg3")

            convs = list_conversations()
            assert convs[0]["message_count"] == 3
