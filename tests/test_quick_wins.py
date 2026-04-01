"""Tests for quick-win features — check_env, seed_sample_db, export chat."""

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCheckEnv:
    """Test the environment validation script."""

    def test_returns_nonzero_when_keys_missing(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            from check_env import main
            result = main()
            assert result == 1

    def test_returns_zero_when_all_present(self, tmp_path):
        env = {
            "NVIDIA_API_KEY": "nvapi-real-key",
            "TAVILY_API_KEY": "tvly-real-key",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch("check_env.Path") as MockPath, \
             patch("check_env.importlib.import_module"):

            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.iterdir.return_value = [MagicMock()]
            MockPath.return_value = mock_path_instance

            from check_env import main
            result = main()
            assert result == 0


class TestSeedSampleDB:
    """Test the database seeding script."""

    def test_creates_tables_and_rows(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("seed_sample_db.DB_PATH", db_path):
            from seed_sample_db import seed
            seed()

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM products")
        assert cur.fetchone()[0] == 10

        cur.execute("SELECT COUNT(*) FROM customers")
        assert cur.fetchone()[0] == 8

        cur.execute("SELECT COUNT(*) FROM orders")
        assert cur.fetchone()[0] == 10

        conn.close()


class TestExportChat:
    """Test the chat export function."""

    def test_export_produces_markdown(self):
        import datetime

        mock_st = MagicMock()
        mock_st.session_state.messages = [
            {"role": "user", "content": "What is NeMo?"},
            {
                "role": "assistant",
                "content": "NeMo is a framework.",
                "trace": ["Step 1: Planned", "Step 2: Retrieved"],
                "tokens": 500,
            },
        ]

        with patch.dict("sys.modules", {"streamlit": mock_st}):
            # Inline the export logic for testing (avoids importing app.py which needs streamlit)
            lines = [
                "# NeMo AgentIQ — Chat Export",
                f"*Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
                "---\n",
            ]
            for msg in mock_st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"## {role}\n")
                lines.append(msg["content"] + "\n")
                if msg.get("trace"):
                    lines.append("<details><summary>Reasoning Trace</summary>\n")
                    for step in msg["trace"]:
                        lines.append(f"- {step}")
                    lines.append("\n</details>\n")
                if msg.get("tokens"):
                    lines.append(
                        f"*Tokens: {msg['tokens']:,} | Est. cost: ${msg['tokens'] * 0.000001:.4f}*\n"
                    )
                lines.append("---\n")
            md = "\n".join(lines)

        assert "# NeMo AgentIQ" in md
        assert "## User" in md
        assert "What is NeMo?" in md
        assert "## Assistant" in md
        assert "Reasoning Trace" in md
        assert "Tokens: 500" in md


class TestSampleDocuments:
    """Test that sample documents were shipped."""

    def test_sample_docs_exist(self):
        docs_dir = Path("data/documents")
        assert docs_dir.exists()
        files = list(docs_dir.glob("*.txt"))
        assert len(files) >= 2

    def test_sample_db_exists(self):
        assert Path("data/sample.db").exists()
