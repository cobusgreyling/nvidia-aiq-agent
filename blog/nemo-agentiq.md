# NeMo AgentIQ: Enterprise Agentic RAG With Multi-Source Reasoning

## TLDR

**Something is shifting in how we think about RAG.** The original retrieval-augmented generation pattern — embed documents, retrieve chunks, stuff into a prompt — was a breakthrough. But it was always a single-source, single-hop affair. NVIDIA's recent [AI-Q blueprint](https://nvidianews.nvidia.com/news/ai-agents) points to what comes next: **agents that autonomously decide which data sources to query, how deep to go, and how to synthesise a cited answer across all of them.** NeMo AgentIQ is my open-source take on that vision, built with **NVIDIA NIM** and **LangChain/LangGraph**.

---

## The Problem With Traditional RAG

I have written extensively about [multi-hop RAG](https://cobusgreyling.medium.com/multihop-rag-1c695794eeda) and the limitations of single-source retrieval. The core issue is this: enterprise knowledge does not live in one place. It is scattered across documents, databases, APIs, and the open web.

A sales manager asking *"How did Q3 revenue compare to forecast?"* needs data from:

- **Documents** — the quarterly report PDF
- **SQL** — the actual revenue figures from the finance database
- **APIs** — the CRM system's forecast data

Traditional RAG gives you one of those. An agentic system gives you all three, cross-referenced and cited.

This is what NVIDIA calls the **AI-Q approach** — and their internal benchmarks show it topping both the DeepResearch Bench and DeepResearch Bench II leaderboards while **cutting query costs by over 50%**.

---

## The Hybrid Model Trick

The cost reduction is not magic. It is architectural.

I explored this pattern in my piece on [agentic workflows](https://cobusgreyling.medium.com/agentic-workflows-034d2df458d3) — the idea that not every step in a pipeline requires a frontier-class model. NVIDIA's approach makes this explicit:

> **Use a large model to plan. Use small models to execute.**

In NeMo AgentIQ, the orchestrator runs on **Nemotron-4-340B** via NVIDIA NIM. It analyses the query once, selects data sources, and decides analysis depth. That is one expensive call — roughly 2K tokens.

The sub-agents — document retrieval, SQL generation, web search, API calls — run on **Nemotron-4-8B**. Four parallel calls at roughly 1K tokens each.

```
Expensive call (Nemotron-340B)     Cheap calls (Nemotron-8B)
         │                              │
    Orchestrator                   Sub-agents do
    plans once                     retrieval + summarisation
         │                              │
    1 call ≈ 2K tokens             4 calls ≈ 1K tokens each
```

The synthesis step uses the large model again for the final answer, but by then the context has been pre-filtered by four specialised agents. Total cost: roughly half of what a naive "send everything to GPT-4" approach would incur.

---

## Architecture

The system is built as a **LangGraph state machine** where each node is a specialised agent:

```
┌─────────────────────────────────────────────────────┐
│                   USER QUERY                         │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│            ORCHESTRATOR AGENT                        │
│         (LangGraph + LangChain)                     │
│                                                      │
│  • Plans query strategy                              │
│  • Selects data sources                              │
│  • Decides analysis depth                            │
│  • Routes to sub-agents                              │
│                                                      │
│  Model: NVIDIA Nemotron-4-340B (via NVIDIA NIM)     │
└──────┬─────────┬──────────┬────────────┬────────────┘
       ▼         ▼          ▼            ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│ DOC RAG  │ │ SQL    │ │ WEB    │ │ API      │
│ AGENT    │ │ AGENT  │ │ AGENT  │ │ AGENT    │
│          │ │        │ │        │ │          │
│LangChain │ │LangCh. │ │LangCh.│ │LangChain │
│Retriever │ │SQL Tool│ │Search  │ │Tool Call │
└────┬─────┘ └───┬────┘ └───┬────┘ └────┬─────┘
     ▼           ▼          ▼            ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│NVIDIA    │ │Postgres│ │Web     │ │REST/     │
│NeMo      │ │SQLite  │ │Search  │ │GraphQL   │
│Retriever │ │etc.    │ │API     │ │Endpoints │
│+ FAISS   │ │        │ │        │ │          │
└──────────┘ └────────┘ └────────┘ └──────────┘
       │         │          │            │
       └─────────┴──────┬───┴────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│              SYNTHESIS AGENT                         │
│         (Nemotron via NVIDIA NIM)                   │
│                                                      │
│  • Merges results from all sources                   │
│  • Generates cited, explained answer                 │
│  • Produces explainability trace                     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              GUARDRAILS LAYER                        │
│           (NVIDIA NeMo Guardrails)                  │
│                                                      │
│  • PII filtering                                     │
│  • Hallucination check                               │
│  • Policy enforcement                                │
└──────────────────────┬──────────────────────────────┘
                       ▼
                   RESPONSE
          (answer + source citations
           + reasoning trace)
```

The key insight from NVIDIA's announcement is that this is not just a technical architecture — it is an **economic one**. The routing layer decides *how much compute to spend* on each query. A simple factual lookup gets shallow treatment. A complex analytical question gets the full multi-source deep dive.

---

## The Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **LLM** | NVIDIA Nemotron (via NIM API) | Reasoning, planning, synthesis |
| **Embeddings** | NVIDIA NV-Embed-v2 (via NIM) | Document vectorisation |
| **Orchestration** | LangGraph | Multi-agent state machine, routing |
| **Agent Framework** | LangChain | Tools, retrievers, chains |
| **Vector Store** | FAISS | Document similarity search |
| **Guardrails** | NeMo Guardrails | Safety, PII, policy |
| **GUI** | Streamlit | Interactive web interface |
| **Tracing** | LangSmith (optional) | Explainability traces |

**LangChain** provides the agent primitives — tools, retrievers, chains — while **LangGraph** handles the state machine that routes between agents. This combination is powerful because LangGraph's conditional edges let the orchestrator dynamically decide which agents to invoke. No wasted compute on irrelevant sources.

The **NeMo Guardrails** layer sits at the output and enforces PII filtering, hallucination checking, and policy controls. In an enterprise context, this is non-negotiable.

---

## The GUI

NeMo AgentIQ ships with a **Streamlit** interface that exposes the full reasoning pipeline to the user:

```
┌─────────────────────────────────────────────────┐
│  NeMo AgentIQ                               ⚙️   │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────────────────────┐  ┌────────┐ │
│  │ Ask anything...                │  │  Send  │ │
│  └────────────────────────────────┘  └────────┘ │
│                                                  │
│  Data Sources:  ☑ Documents  ☑ SQL  ☑ Web  ☑ API│
│  Model:  [Nemotron-340B ▼]   Depth: [Auto ▼]   │
│                                                  │
├─────────────────────────────────────────────────┤
│  ANSWER                                          │
│  ┌─────────────────────────────────────────────┐ │
│  │ Based on 3 sources, revenue increased 23%   │ │
│  │ in Q3 driven by...                          │ │
│  │                                              │ │
│  │ Sources: [doc.pdf p.12] [sales_db] [web]    │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  REASONING TRACE                          [▼]    │
│  ┌─────────────────────────────────────────────┐ │
│  │ Step 1: Classified as financial query        │ │
│  │ Step 2: Selected sources → docs, sql, web   │ │
│  │ Step 3: Doc agent found 4 matches            │ │
│  │ Step 4: SQL agent queried revenue table      │ │
│  │ Step 5: Synthesised with citations           │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  TOKEN USAGE        COST                         │
│  ████░░ 3.2K        $0.004                       │
└─────────────────────────────────────────────────┘
```

Three features make this interface more than a chatbot:

1. **Source toggles** — the user explicitly chooses which data sources are in scope. This is critical for compliance-sensitive environments where you may not want the agent querying external APIs.

2. **Reasoning trace** — an expandable panel showing every step the agent took. This directly implements NVIDIA's emphasis on explainability: *"built-in mechanisms explaining how each AI answer is produced."*

3. **Token/cost meter** — real-time visibility into compute spend. When you are running a hybrid large/small model architecture, this transparency matters.

---

## Why This Matters

NVIDIA's announcement was not just about a toolkit. It was a statement about where enterprise AI is heading. As I discussed in [NVIDIA Says Small Language Models Are The Future of Agentic AI](https://cobusgreyling.medium.com/nvidia-says-small-language-models-are-the-future-of-agentic-ai-f1f7289d9565), the future is not about running the biggest model on every query. It is about **intelligent routing** — using large models where they create the most value and small models everywhere else.

NeMo AgentIQ is a working implementation of that principle. The orchestrator is the expensive brain. The sub-agents are the cheap hands. The result is an enterprise-grade research assistant that is both more capable *and* more cost-effective than a monolithic approach.

**The shift from RAG to agentic RAG is not incremental. It is architectural.** And the enterprises that figure out this routing problem first will have a significant cost advantage.

---

## Getting Started

```bash
git clone https://github.com/cobusgreyling/nvidia-aiq-agent.git
cd nvidia-aiq-agent
pip install -r requirements.txt
cp .env.example .env
# Add your NVIDIA NIM API key
python ingest.py
streamlit run app.py
```

---

## Additional Resources

- [NVIDIA AI Agents Announcement](https://nvidianews.nvidia.com/news/ai-agents)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [MultiHop-RAG](https://cobusgreyling.medium.com/multihop-rag-1c695794eeda) — Cobus Greyling
- [Agentic Workflows](https://cobusgreyling.medium.com/agentic-workflows-034d2df458d3) — Cobus Greyling
- [NVIDIA Says Small Language Models Are The Future of Agentic AI](https://cobusgreyling.medium.com/nvidia-says-small-language-models-are-the-future-of-agentic-ai-f1f7289d9565) — Cobus Greyling

---

*I'm Cobus Greyling. I explore how AI agents, LLMs, and conversational AI are reshaping enterprise workflows. Follow me on [Medium](https://cobusgreyling.medium.com) for more.*
