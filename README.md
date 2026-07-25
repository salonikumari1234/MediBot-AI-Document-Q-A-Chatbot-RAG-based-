# MediBot — AI Document Q&A Chatbot (RAG-based)

MediBot is a Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions grounded in your own documents. It was built by extending an open-source RAG tutorial project with a custom dataset, a runtime document-upload feature, improved source citations, and proper error handling.

Ask it a question, and it retrieves the most relevant passages from your PDF(s) using semantic search, then uses an LLM to generate an answer strictly grounded in that retrieved context — with the exact source pages cited.

## Demo

> Add a screenshot or short GIF of the app here once you have one — this makes a huge difference for anyone skimming your resume/GitHub.

## Features

- **Document-grounded answers** — responses are generated only from retrieved context, not the LLM's general knowledge, reducing hallucination
- **Runtime PDF upload** — upload any PDF through the UI and query it immediately, no restart or pre-processing required
- **Source citations** — every answer shows which document and page number it came from
- **Multi-document knowledge base** — pre-indexed with a 759-page medical encyclopedia and Ian Goodfellow's *Deep Learning* textbook
- **Friendly error handling** — clear messages for missing API keys, unavailable models, and rate limits instead of raw stack traces
- **Fast inference** — powered by Groq's LPU-hosted LLMs

## Architecture

```
Phase 1 — Build Knowledge Base
  Raw PDF(s) → Chunking → Embeddings (HuggingFace) → FAISS Vector Store

Phase 2 — Retrieval + Generation
  User Question → Embed Question → Semantic Search (FAISS) → Retrieved Context → LLM (Groq)

Phase 3 — Interface
  Streamlit chat UI ↔ Cached vector store ↔ RAG chain (LangChain)
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Orchestration | LangChain |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| UI | Streamlit |
| Language | Python |

## Setup

**1. Clone and enter the project**
```bash
git clone <your-repo-url>
cd medical-chatbot
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up API keys**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) (no credit card required).

**5. Build the vector store** (run once, or whenever you change the PDFs in `data/`)
```bash
python create_memory_for_llm.py
```

**6. Run the app**
```bash
streamlit run medibot.py
```

Open `http://localhost:8501` in your browser.

## Usage

- Ask questions directly about the pre-loaded documents (medical topics, deep learning concepts)
- Or upload your own PDF using the file uploader in the app, and ask questions about it immediately
- Each answer includes a **Sources** section citing the exact document and page it was drawn from

## What I Customized

This project started from an open-source RAG chatbot tutorial and was extended with:

- **Migrated the LLM backend** after diagnosing a production issue: the tutorial's hardcoded Groq model (`meta-llama/llama-4-maverick-17b-128e-instruct`) had been deprecated, causing a 404 error. Resolved by reviewing Groq's deprecation docs and migrating to their current recommended model.
- **Added a custom dataset** (Ian Goodfellow's *Deep Learning* textbook) alongside the original medical reference, expanding the knowledge base to a second domain.
- **Built a self-upload feature**, letting users add and query their own PDF at runtime via a session-scoped FAISS index, without needing to re-run the ingestion script or restart the app.
- **Redesigned source-citation display**, replacing raw Python object dumps with a clean, deduplicated, human-readable list of source files and page numbers.
- **Added structured error handling** for missing API keys, deprecated/unavailable models, rate limits, and missing documents, replacing raw stack traces with actionable user-facing messages.

## Future Improvements

- User authentication
- Multi-turn conversational memory
- Automated evaluation suite (test known Q&A pairs for retrieval/answer accuracy)
- Deployment to Streamlit Community Cloud


