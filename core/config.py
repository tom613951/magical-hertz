import os
from dotenv import load_dotenv

# Load environmental variables from .env file
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))

    @classmethod
    def get_default_model(cls, provider: str) -> str:
        """Return a suitable default model name for each provider."""
        provider = provider.lower()
        if cls.DEFAULT_MODEL and cls.DEFAULT_PROVIDER == provider:
            return cls.DEFAULT_MODEL
            
        defaults = {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-1.5-flash",
            "anthropic": "claude-3-5-sonnet-20240620",
            "ollama": "llama3"
        }
        return defaults.get(provider, "")
