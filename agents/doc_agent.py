"""Document RAG Agent — retrieves and summarises from vectorised documents."""

import os
import tempfile
import yaml
from pathlib import Path
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from log_config import setup_logging

logger = setup_logging()

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

VECTOR_STORE_PATH = "data/vectorstore"
DOCS_DIR = "data/documents"


class DocAgent:
    def __init__(self):
        self.embeddings = NVIDIAEmbeddings(
            model=MODEL_CONFIG["embeddings"]["model"]
        )
        self.llm = ChatNVIDIA(
            model=MODEL_CONFIG["sub_agents"]["model"],
            temperature=MODEL_CONFIG["sub_agents"]["temperature"],
            max_tokens=MODEL_CONFIG["sub_agents"]["max_tokens"],
        )
        self.vectorstore = None

    def load_vectorstore(self):
        """Load the FAISS vector store from disk."""
        if not Path(VECTOR_STORE_PATH).exists():
            return False
        self.vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True

    def ingest_uploaded_files(self, uploaded_files: list) -> int:
        """Ingest uploaded files into the vector store. Returns chunk count."""
        os.makedirs(DOCS_DIR, exist_ok=True)
        documents = []

        for uploaded_file in uploaded_files:
            # Save to temp file for loader compatibility
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Also save a permanent copy
            dest = Path(DOCS_DIR) / uploaded_file.name
            dest.write_bytes(uploaded_file.getbuffer())

            try:
                if suffix.lower() == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                documents.extend(loader.load())
            finally:
                os.unlink(tmp_path)

        if not documents:
            return 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        if self.vectorstore is not None:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Persist
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        self.vectorstore.save_local(VECTOR_STORE_PATH)

        logger.info("Ingested %d file(s) → %d chunks", len(uploaded_files), len(chunks))
        return len(chunks)

    def run(self, state: dict) -> dict:
        """Retrieve relevant documents and generate an answer."""
        if self.vectorstore is None:
            if not self.load_vectorstore():
                state["doc_results"] = "No documents ingested yet. Upload documents to get started."
                state["reasoning_trace"].append("Step: Doc agent — no vector store found")
                return state

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5 if state.get("depth") == "deep" else 3},
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        result = chain.invoke({"query": state["query"]})

        sources = []
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            sources.append(f"[{src} p.{page}]" if page else f"[{src}]")

        state["doc_results"] = result["result"]
        logger.info("Doc agent retrieved %d matches", len(sources))
        state["reasoning_trace"].append(
            f"Step: Doc agent retrieved {len(sources)} matches from {', '.join(set(sources))}"
        )
        return state
