import os
from langchain_core.language_models.chat_models import BaseChatModel
from core.config import Config

def get_llm(
    provider: str = None,
    model_name: str = None,
    temperature: float = None,
    api_key: str = None,
    base_url: str = None
) -> BaseChatModel:
    """
    Factory function to get a LangChain ChatModel client based on configuration.
    Supports openai, deepseek, gemini, anthropic, and ollama.
    """
    provider = (provider or Config.DEFAULT_PROVIDER).lower()
    model = model_name or Config.get_default_model(provider)
    temp = temperature if temperature is not None else Config.DEFAULT_TEMPERATURE

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        key = api_key or Config.OPENAI_API_KEY
        base = base_url or Config.OPENAI_API_BASE
        
        # Enable usage of standard proxy URLs via env if key is present
        if not key:
            raise ValueError("OpenAI API Key is missing. Please configure it in the .env file or in the application.")
            
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=key,
            base_url=base
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        key = api_key or Config.DEEPSEEK_API_KEY
        base = base_url or Config.DEEPSEEK_API_BASE
        
        if not key:
            raise ValueError("DeepSeek API Key is missing. Please configure it in the .env file or in the application.")
            
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=key,
            base_url=base
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or Config.GOOGLE_API_KEY
        if not key:
            # Attempt to read from environment variable standard key name
            key = os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("Google API Key is missing. Please configure it in the .env file or in the application.")
            
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            google_api_key=key
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        key = api_key or Config.ANTHROPIC_API_KEY
        if not key:
            raise ValueError("Anthropic API Key is missing. Please configure it in the .env file or in the application.")
            
        return ChatAnthropic(
            model=model,
            temperature=temp,
            api_key=key
        )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            # Fallback to community if langchain-ollama isn't installed
            from langchain_community.chat_models import ChatOllama
            
        host = base_url or Config.OLLAMA_HOST
        return ChatOllama(
            model=model,
            temperature=temp,
            base_url=host
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
