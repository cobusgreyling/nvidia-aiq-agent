"""Document RAG Agent — retrieves and summarises from vectorised documents."""

import yaml
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

with open("config/models.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

VECTOR_STORE_PATH = "data/vectorstore"


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
        self.vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def run(self, state: dict) -> dict:
        """Retrieve relevant documents and generate an answer."""
        if self.vectorstore is None:
            self.load_vectorstore()

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
        state["reasoning_trace"].append(
            f"Step 3: Doc agent retrieved {len(sources)} matches from {', '.join(set(sources))}"
        )
        return state
