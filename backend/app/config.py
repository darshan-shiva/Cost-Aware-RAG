
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        case_sensitive=False,
        env_file=".env",
        extra="ignore",
    )

    PROJECT_NAME: str = (
        "Cost-Aware RAG: Semantic Caching & Intelligent Model Routing"
    )
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    # Semantic Cache Settings
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    SIMULATED_CACHE: bool = False
    SIMILARITY_THRESHOLD: float = 0.65
    CACHE_MAX_ENTRIES: int = 1000

    # LLM Provider
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_COMPLETION_TOKENS: int = 1024

    # Model Routing Profiles
    # Prices are USD per 1,000 tokens.
    # Confirm the provider's current pricing before treating these
    # estimates as actual monetary charges.
    LIGHTWEIGHT_MODEL_ID: str = "openai/gpt-oss-20b"
    LIGHTWEIGHT_COST_INPUT_PER_1K: float = 0.000075
    LIGHTWEIGHT_COST_OUTPUT_PER_1K: float = 0.00030

    COMPLEX_MODEL_ID: str = "openai/gpt-oss-120b"
    COMPLEX_COST_INPUT_PER_1K: float = 0.00015
    COMPLEX_COST_OUTPUT_PER_1K: float = 0.00060

    # Persistence
    DATABASE_PATH: str = str(
        Path(__file__).resolve().parent.parent / "metrics.db"
    )


settings = Settings()
