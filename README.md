#  YouTube Earnings Call RAG Analyzer

> An AI-powered competitive intelligence system that transforms hours of earnings call videos into actionable business insights in minutes using Retrieval-Augmented Generation (RAG).

##  Overview

Financial analysts and business leaders spend countless hours listening to lengthy earnings calls. The **YouTube Earnings Call RAG Analyzer** automates this pipeline—extracting insights from YouTube transcripts to enable immediate semantic search and cross-company comparisons without high operational costs.

-  **Ingest:** Fetches transcripts directly from YouTube earnings call videos.
-  **Index:** Generates vector representations using lightweight, local open-source models.
-  **Query:** Returns synthesized, data-backed answers in seconds via Google Gemini.
-  **Compare:** Benchmarks competitors against shared financial metrics and operational updates.

##  Target Audience

- **Investment Analysts & Researchers:** Streamline quarterly earnings evaluations.
- **Corporate Strategists:** Monitor competitor moves and identify industry shifts.
- **MBA Students & Academics:** Quickly gather primary-source corporate intelligence for case studies.

##  System Architecture

┌─────────────────────────────────────────────────────────────┐
│                   YOUTUBE EARNINGS CALLS                    │
│                (JPMorgan, Robinhood, Circle)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               TRANSCRIPT FETCHING (YouTube API)             │
│                     youtube_transcript_api                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   TEXT CHUNKING (LangChain)                 │
│                 RecursiveCharacterTextSplitter              │
│                (chunk_size: 2000, overlap: 200)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 EMBEDDING GENERATION (Local)                │
│            HuggingFaceEmbeddings (all-MiniLM-L6-v2)         │
│                    Converts text → vectors                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    VECTOR STORAGE (FAISS)                   │
│              Indexed searchable knowledge base              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       RETRIEVAL (RAG)                       │
│           Finds most relevant chunks for each query         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                ANSWER GENERATION (Gemini LLM)               │
│         Gemini 3.6 Flash (via langchain-google-genai)       │
│                 Generates strategic insights                │
└─────────────────────────────────────────────────────────────┘

##  Key Features

* **Multi-Company Analysis:** Ingest and index transcripts across competitors (e.g., JPMorgan, Robinhood, Circle) simultaneously.
* **Comparative Q&A:** Ask cross-cutting questions to benchmark performance across multiple entities.
* **Cost-Optimized Pipeline:** Uses free, locally hosted embeddings (`all-MiniLM-L6-v2`) to eliminate vector embedding API costs.
* **Resilient Quota Handling:** Includes automated retry loops to accommodate rate limits during intensive inference tasks.
* **Granular Retrieval:** Configured with intelligent chunk sizes (2000 chars, 200 overlap) to preserve business context.

##  Technology Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| **Language** | Python 3.10+ | Core application runtime |
| **Orchestration** | LangChain | RAG pipeline coordination and prompt structuring |
| **Transcripts** | `youtube-transcript-api` | Captures transcripts and timestamps directly from video IDs |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Offline, cost-free dense vector generation |
| **Vector Index** | FAISS | High-performance in-memory similarity search |
| **LLM** | Google Gemini 3.6 Flash | Synthesizes retrieved chunks into strategic answers |


##  Business Impact

| Metric | Traditional Manual Review | With RAG Analyzer | Efficiency Gain |
| --- | --- | --- | --- |
| **Time per Call** | 2–3 hours | 2–3 seconds per query | **~99% faster** |
| **Coverage** | 1 company at a time | Multi-company queries | **Scalable** |
| **Cost** | High manual labor | Zero embedding costs + low-cost LLM API | **Cost-effective** |
