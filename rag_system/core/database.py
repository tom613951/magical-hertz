import os
import shutil
from langchain_chroma import Chroma
from core.config import Config

# Local DB persistence path
DB_DIR = r"C:\Users\26503\Documents\antigravity\magical-hertz\rag_system\chroma_db"

def get_embeddings(provider: str, api_key: str = None):
    """
    Factory function for Embeddings model matching the selected LLM provider.
    """
    provider = provider.lower()
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=api_key or Config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
    elif provider == "deepseek":
        # DeepSeek doesn't offer embedding models on their standard API, 
        # so we fall back to OpenAI embeddings or local HuggingFace.
        # Let's fallback to OpenAI if API key is present, otherwise Google.
        from langchain_openai import OpenAIEmbeddings
        key = api_key or Config.OPENAI_API_KEY or Config.DEEPSEEK_API_KEY
        return OpenAIEmbeddings(
            api_key=key,
            model="text-embedding-3-small"
        )
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenAIEmbeddings
        key = api_key or Config.GOOGLE_API_KEY
        return GoogleGenAIEmbeddings(
            google_api_key=key,
            model="models/embedding-001"
        )
    elif provider == "anthropic":
        # Anthropic doesn't have an embeddings API, fallback to OpenAI
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=api_key or Config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
    elif provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=Config.OLLAMA_HOST,
            model="nomic-embed-text"
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")

class RAGDatabase:
    """Manages the lifecycle of Chroma DB local vectorstore."""
    
    def __init__(self, provider: str, api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        self.embeddings = get_embeddings(provider, api_key)
        self.db = None
        self._init_db()

    def _init_db(self):
        """Load or create the Chroma database."""
        self.db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings,
            collection_name="rag_knowledge_base"
        )

    def add_documents(self, documents):
        """Add chunks to the vector database."""
        if not documents:
            return
        if self.db is None:
            self._init_db()
        self.db.add_documents(documents)

    def get_retriever(self, search_kwargs=None):
        """Get standard vectorstore retriever."""
        if self.db is None:
            self._init_db()
        return self.db.as_retriever(search_kwargs=search_kwargs or {"k": 10})

    @staticmethod
    def clear_database():
        """Delete local Chroma database files on disk."""
        if os.path.exists(DB_DIR):
            try:
                shutil.rmtree(DB_DIR)
                return "Successfully cleared local Vector Database."
            except Exception as e:
                return f"Error clearing Database: {str(e)}"
        return "Database is already empty."
