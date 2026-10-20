from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", case_sensitive=False)

    PROJECT_NAME: str = "Cost-Aware RAG: Semantic Caching & Intelligent Model Routing"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    
    # Semantic Cache Settings
    SIMULATED_CACHE: bool = True
    SIMILARITY_THRESHOLD: float = 0.82
    CACHE_MAX_ENTRIES: int = 1000
    
    # Model Routing Profiles & Costs (per 1k tokens in USD)
    LIGHTWEIGHT_MODEL_ID: str = "llama-3.1-8b-instruct"
    LIGHTWEIGHT_COST_INPUT_PER_1K: float = 0.00015
    LIGHTWEIGHT_COST_OUTPUT_PER_1K: float = 0.00060

    COMPLEX_MODEL_ID: str = "gpt-4o"
    COMPLEX_COST_INPUT_PER_1K: float = 0.00250
    COMPLEX_COST_OUTPUT_PER_1K: float = 0.01000

    # Persistence
    DATABASE_PATH: str = str(Path(__file__).resolve().parent.parent / "metrics.db")

settings = Settings()

