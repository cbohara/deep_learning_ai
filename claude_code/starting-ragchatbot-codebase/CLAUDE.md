# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) chatbot that lets users query course materials via a chat interface. Built with a FastAPI backend, ChromaDB vector store, Claude API for generation, and a vanilla HTML/CSS/JS frontend.

## Commands

```bash
# Install dependencies
uv sync

# Run the app (serves frontend + backend on port 8000)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

No test framework or linter is configured.

## Configuration

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Other settings (chunk size, embedding model, etc.) are configured via environment variables in `backend/config.py`.

## Architecture

**Query flow:** Frontend (`script.js`) → `/api/query` endpoint (`app.py`) → `RAGSystem.query()` → Claude tool-calling loop (`ai_generator.py` ↔ `search_tools.py` ↔ `vector_store.py`) → response with sources

**Document ingestion flow (on startup):** Course `.txt` files in `docs/` → `document_processor.py` parses metadata + chunks content → `vector_store.py` stores in two ChromaDB collections (`course_catalog` for metadata, `course_content` for searchable chunks)

**Key design decisions:**
- Claude Sonnet 4 with tool calling — the AI decides when/how to search rather than using a fixed retrieval step
- Conversation history managed server-side via `session_manager.py` (in-memory, 2-exchange default)
- Sentence-based chunking (800 chars, 100 char overlap) in `document_processor.py`
- Course documents must follow a specific format with `Course Title:`, `Course Link:`, `Course Instructor:`, and `Lesson N:` headers
- Frontend is served as static files by FastAPI (no build step)
