 # YouTube Earnings Call RAG Analyzer
An AI-powered competitive intelligence system that transforms hours of earnings call videos into actionable business insights in minutes.

## Overview
This project demonstrates how Retrieval-Augmented Generation (RAG) can revolutionize competitive analysis. Instead of manually listening to hours of earnings calls, this system:

✅ Ingests YouTube earnings call transcripts
✅ Indexes content using embedding models
✅ Answers strategic business questions instantly
✅ Compares competitors across key metrics

Built for: MBA students, analysts, strategists, and business leaders who want to leverage AI for faster, data-driven decisions.

## Architecture

┌─────────────────────────────────────────────────────────────┐
│                     YOUTUBE EARNINGS CALLS                  │
│              (JPMorgan, Robinhood, Circle)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              TRANSCRIPT FETCHING (YouTube API)              │
│                 youtube_transcript_api                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 TEXT CHUNKING (LangChain)                   │
│            RecursiveCharacterTextSplitter                   │
│                 (chunk_size: 2000, overlap: 200)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               EMBEDDING GENERATION (Free)                   │
│         HuggingFaceEmbeddings (all-MiniLM-L6-v2)           │
│                    Converts text → vectors                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  VECTOR STORAGE (FAISS)                     │
│            Indexed searchable knowledge base                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     RETRIEVAL (RAG)                         │
│         Finds most relevant chunks for each question        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ANSWER GENERATION (Gemini LLM)                 │
│         Gemini 3.6 Flash (via langchain-google-genai)       │
│                Generates strategic insights                 │
└─────────────────────────────────────────────────────────────┘

## Key Features
Multi-company analysis: Process multiple competitors simultaneously

Batch processing: Handle large transcripts efficiently

Interactive Q&A: Ask custom questions in real-time

Comparison mode: Compare competitors on specific topics

Free embeddings: No API cost (uses HuggingFace)

Quota management: Automatic retry logic for API limits

## Technology Stack
Component	Technology	Purpose
Language	Python 3.10+	Core programming
Embeddings	HuggingFace (all-MiniLM-L6-v2)	Free, offline text→vectors
Vector DB	FAISS	Efficient similarity search
LLM	Google Gemini 3.6 Flash	Answer generation
Framework	LangChain	RAG pipeline orchestration
Transcripts	YouTube Transcript API	Fetch video captions

## Business Impact
Time Savings
Manual analysis: 2-3 hours per earnings call

With this system: 2-3 seconds per query

Efficiency gain: ~99%

## Applications
Investment research: Rapid analysis of portfolio companies

Competitive intelligence: Track industry trends

Strategic planning: Identify market opportunities

Due diligence: Evaluate potential acquisition
