"""Tests for the document ingestion pipeline."""

from unittest.mock import patch, MagicMock


class TestIngest:
    """Test ingestion pipeline logic."""

    def test_creates_sample_when_no_docs_dir(self, tmp_path):
        docs_dir = tmp_path / "documents"
        vector_path = str(tmp_path / "vectorstore")

        with patch("ingest.DOCS_DIR", str(docs_dir)), \
             patch("ingest.VECTOR_STORE_PATH", vector_path), \
             patch("ingest.NVIDIAEmbeddings"), \
             patch("ingest.FAISS") as MockFAISS, \
             patch("ingest.DirectoryLoader") as MockLoader:

            # Simulate loader returning one document
            mock_doc = MagicMock()
            mock_doc.page_content = "test content"
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [mock_doc]
            MockLoader.return_value = mock_loader_instance

            mock_vs = MagicMock()
            MockFAISS.from_documents.return_value = mock_vs

            from ingest import ingest
            ingest()

            # Sample doc should have been created
            assert (docs_dir / "sample.txt").exists()
            # FAISS should have been called
            MockFAISS.from_documents.assert_called_once()

    def test_skips_when_no_documents_found(self, tmp_path, caplog):
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        import logging

        with patch("ingest.DOCS_DIR", str(docs_dir)), \
             patch("ingest.VECTOR_STORE_PATH", str(tmp_path / "vs")), \
             patch("ingest.NVIDIAEmbeddings"), \
             patch("ingest.DirectoryLoader") as MockLoader, \
             caplog.at_level(logging.WARNING, logger="agentiq"):

            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = []
            MockLoader.return_value = mock_loader_instance

            from ingest import ingest
            ingest()

            assert "No documents found" in caplog.text
