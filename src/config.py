"""
Configuration module for PaperForge AI.
Centralizes all environment variable loading and model configuration.
"""
import os
from typing import Literal, Optional

# Load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# Model Configuration
# =============================================================================

def get_openai_model() -> str:
    """Get OpenAI model from environment or default."""
    return os.environ.get("OPENAI_MODEL", "gpt-4o")


def get_gemini_model() -> str:
    """Get Gemini model from environment or default."""
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def get_model_for_provider(provider: Literal["openai", "gemini"]) -> str:
    """Get the configured model for the given provider."""
    if provider == "openai":
        return get_openai_model()
    else:
        return get_gemini_model()


# =============================================================================
# API Keys
# =============================================================================

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and key != "your-openai-api-key-here":
        return key
    return None


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key and key != "your-gemini-api-key-here":
        return key
    return None


# =============================================================================
# Provider Configuration
# =============================================================================

def get_default_provider() -> Literal["openai", "gemini"]:
    """Get the default AI provider from environment."""
    provider = os.environ.get("DEFAULT_PROVIDER", "openai").lower()
    if provider in ("openai", "gemini"):
        return provider
    return "openai"


def get_available_providers() -> list[str]:
    """Get list of available AI providers based on configured API keys."""
    providers = []
    if get_openai_api_key():
        providers.append("openai")
    if get_gemini_api_key():
        providers.append("gemini")
    return providers


def get_api_key_for_provider(provider: Literal["openai", "gemini"]) -> Optional[str]:
    """Get the API key for the given provider."""
    if provider == "openai":
        return get_openai_api_key()
    else:
        return get_gemini_api_key()


# =============================================================================
# Utility
# =============================================================================

def print_config():
    """Print current configuration (for debugging)."""
    print("PaperForge AI Configuration:")
    print(f"  Default Provider: {get_default_provider()}")
    print(f"  OpenAI Model: {get_openai_model()}")
    print(f"  Gemini Model: {get_gemini_model()}")
    print(f"  Available Providers: {get_available_providers()}")
