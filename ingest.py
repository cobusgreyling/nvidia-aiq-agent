"""Document ingestion pipeline — vectorises documents into FAISS."""

import os
import yaml
from pathlib import Path
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

DOCS_DIR = "data/documents"
VECTOR_STORE_PATH = "data/vectorstore"


def ingest():
    """Load documents, split, embed, and store in FAISS."""
    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        # Create a sample document
        sample = docs_path / "sample.txt"
        sample.write_text(
            "NeMo AgentIQ is an enterprise agentic RAG system. "
            "It uses NVIDIA Nemotron models via NIM for reasoning and synthesis. "
            "LangGraph orchestrates multi-agent workflows across document, SQL, web, and API sources. "
            "NeMo Guardrails enforce PII filtering, hallucination checks, and policy controls."
        )
        print(f"Created sample document at {sample}")

    # Load all documents
    loaders = []
    for ext, loader_cls in [("*.txt", TextLoader), ("*.pdf", PyPDFLoader)]:
        loader = DirectoryLoader(DOCS_DIR, glob=ext, loader_cls=loader_cls)
        loaders.append(loader)

    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: {e}")

    if not documents:
        print("No documents found. Add files to data/documents/ and re-run.")
        return

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    # Embed and store
    embeddings = NVIDIAEmbeddings(model=MODEL_CONFIG["embeddings"]["model"])
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    ingest()
