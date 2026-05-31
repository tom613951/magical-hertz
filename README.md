# Magical Hertz 🌍

A premium GIS Multi-Agent System and Advanced RAG repository built with LangChain, LangGraph, and Streamlit.

---

## 🌟 Modules

### 1. GeoGraph (GIS Multi-Agent System)
A StateGraph-based digital workspace simulating a team of GIS experts (Planner, Developer, QA Inspector) collaborating to solve geospatial problems and write automation code.
* **Run**: `streamlit run app.py`

### 2. Advanced RAG (Geospatial Knowledge Base)
A QA system featuring Hybrid Vector + Lexical (BM25) search, Query Translation/Expansion, and Flashrank Reranking.
* **Run**: `streamlit run rag_system/app.py`

---

## 🚀 Setup & Installation

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure API credentials**:
   Copy `.env.example` to `.env` and fill in keys for your selected LLM providers (e.g., OpenAI, Gemini, DeepSeek, Anthropic, or local Ollama).
