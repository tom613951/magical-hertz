import streamlit as st
import os
import io
from langchain_core.messages import SystemMessage, HumanMessage
from core.config import Config
from core.llm import get_llm
from core.database import RAGDatabase
from core.retrieval import AdvancedRetriever
from utils.parser import parse_file, chunk_text

# Configure page settings
st.set_page_config(
    page_title="GeoGraph RAG - Advanced Document QA",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for adaptive light/dark mode UI elements
st.markdown("""
<style>
    .chunk-card {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(128, 128, 128, 0.03);
    }
    .score-badge {
        font-size: 0.78em;
        padding: 2px 6px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        margin-right: 5px;
    }
    .badge-rerank { background-color: #ec4899; }
    .badge-vector { background-color: #3b82f6; }
    .badge-bm25 { background-color: #10b981; }
    .badge-source { background-color: #6b7280; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("📚 GeoGraph - Advanced RAG")
st.caption("Hybrid Keyword + Vector Search with Query Expansion & Flashrank Reranking (Self-Adaptive Theme)")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider Selection
    provider = st.selectbox(
        "LLM Provider",
        options=["openai", "deepseek", "gemini", "anthropic", "ollama"],
        index=["openai", "deepseek", "gemini", "anthropic", "ollama"].index(Config.DEFAULT_PROVIDER)
    )
    
    # Dynamic Key fields
    api_key_default = ""
    base_url_default = ""
    
    if provider == "openai":
        api_key_default = Config.OPENAI_API_KEY
        base_url_default = Config.OPENAI_API_BASE
    elif provider == "deepseek":
        api_key_default = Config.DEEPSEEK_API_KEY
        base_url_default = Config.DEEPSEEK_API_BASE
    elif provider == "gemini":
        api_key_default = Config.GOOGLE_API_KEY
    elif provider == "anthropic":
        api_key_default = Config.ANTHROPIC_API_KEY
        
    api_key = ""
    if provider != "ollama":
        api_key = st.text_input(
            f"{provider.capitalize()} API Key",
            value=api_key_default,
            type="password"
        )
        
    base_url = ""
    if provider in ["openai", "deepseek", "ollama"]:
        base_url_val = Config.OLLAMA_HOST if provider == "ollama" else base_url_default
        base_url = st.text_input("API Base / Host URL", value=base_url_val)
        
    # Model Selection
    default_model = Config.get_default_model(provider)
    model_name = st.text_input("Model Name", value=default_model)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1) # 0.0 is best for QA factual accuracy
    
    st.markdown("---")
    st.header("🧹 Database Utilities")
    
    # Database Clear Button
    if st.button("Clear Vector database", type="secondary", use_container_width=True):
        res = RAGDatabase.clear_database()
        st.success(res)
        st.rerun()

# Document Uploader Section
st.subheader("📁 1. Load Documents to Knowledge Base")
uploaded_files = st.file_uploader(
    "Upload PDF, Markdown, or TXT Files",
    type=["pdf", "md", "txt", "json", "geojson"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Build Knowledge Base", type="primary"):
        # Validate model credentials before starting
        if provider != "ollama" and not api_key:
            st.error("API Key is required to run embeddings!")
        else:
            with st.spinner("Processing documents and generating embeddings..."):
                try:
                    db = RAGDatabase(provider=provider, api_key=api_key)
                    all_chunks = []
                    
                    for uploaded_file in uploaded_files:
                        # Streamlit UploadedFile is file-like. Convert to BytesIO or read directly
                        filename = uploaded_file.name
                        file_bytes = io.BytesIO(uploaded_file.read())
                        
                        # Parse
                        raw_text = parse_file(file_bytes, filename)
                        
                        # Chunk
                        chunks = chunk_text(raw_text, filename)
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        # Write to database
                        db.add_documents(all_chunks)
                        st.success(f"Successfully processed {len(uploaded_files)} file(s) and embedded {len(all_chunks)} text chunks into Chroma!")
                    else:
                        st.warning("No text could be extracted from uploaded files.")
                except Exception as e:
                    st.error(f"Error building knowledge base: {str(e)}")

# Q&A Section
st.subheader("💬 2. Ask the Knowledge Base")
query_input = st.text_input("Enter your question:")

if st.button("Query Knowledge Base", use_container_width=True):
    if not query_input.strip():
        st.warning("Please enter a question first!")
    elif provider != "ollama" and not api_key:
        st.error("API Key is required in sidebar to query.")
    else:
        with st.spinner("Retrieving facts and generating answer..."):
            try:
                # 1. Instantiate database & retriever
                db = RAGDatabase(provider=provider, api_key=api_key)
                
                # Setup LLM
                llm = get_llm(
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    base_url=base_url if provider in ["openai", "deepseek", "ollama"] else None
                )
                
                retriever = AdvancedRetriever(db, llm)
                
                # 2. Hybrid Retrieve & Rerank
                retrieved_results = retriever.retrieve(query_input, top_k=5)
                
                if not retrieved_results:
                    st.warning("No relevant matching documents found in the database. Please upload documents first!")
                else:
                    # 3. Compile context text
                    context_blocks = []
                    for idx, res in enumerate(retrieved_results, 1):
                        doc = res["document"]
                        source = doc.metadata.get("source", "Unknown")
                        context_blocks.append(f"--- Context Segment {idx} (Source: {source}) ---\n{doc.page_content}")
                    
                    context_str = "\n\n".join(context_blocks)
                    
                    # 4. Generate Answer using Context
                    system_prompt = """You are a highly analytical QA assistant. Answer the user's question precisely and truthfully based ONLY on the provided context segments. 
If the context does not contain the answer, reply: "I cannot find the answer in the provided documents." Do not invent facts.

Context:
{context}
"""
                    response = llm.invoke([
                        SystemMessage(content=system_prompt.format(context=context_str)),
                        HumanMessage(content=query_input)
                    ])
                    
                    # 5. Render results in Tabs
                    tab_ans, tab_source, tab_queries = st.tabs([
                        "✏️ Answer", 
                        "🔍 Retrieved Chunks (Reranked)", 
                        "⚙️ Expanded Search Queries"
                    ])
                    
                    with tab_ans:
                        st.markdown("### Answer")
                        st.write(response.content)
                        
                    with tab_source:
                        st.markdown("### Top Retrieved Context Chunks (Flashrank Reranked)")
                        for idx, res in enumerate(retrieved_results, 1):
                            doc = res["document"]
                            source = doc.metadata.get("source", "Unknown")
                            chunk_id = doc.metadata.get("chunk_id", "?")
                            
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div>
                                    <span class="score-badge badge-rerank">Rerank Score: {res['rerank_score']:.3f}</span>
                                    <span class="score-badge badge-vector">Vector Sim: {res['vector_score']:.3f}</span>
                                    <span class="score-badge badge-bm25">BM25 Score: {res['bm25_score']:.3f}</span>
                                    <span class="score-badge badge-source">File: {source} (Chunk {chunk_id})</span>
                                </div>
                                <div style="margin-top: 10px; font-size: 0.95em;">
                                    {doc.page_content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with tab_queries:
                        st.markdown("### Expanded Queries generated by LLM for retrieval:")
                        from core.retrieval import expand_query
                        expanded_queries = expand_query(query_input, llm)
                        for q in expanded_queries:
                            st.markdown(f"- *\"{q}\"*")
                            
            except Exception as e:
                st.error(f"Error querying knowledge base: {str(e)}")
                st.exception(e)
