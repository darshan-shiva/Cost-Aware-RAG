
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
    VERSION: str = "0.3.0"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Semantic Cache Settings
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    SIMULATED_CACHE: bool = False
    # Dimension of the lexical hashing embedder used when EMBEDDING_PROVIDER
    # is "simulated" (memory-constrained hosts). It is not a semantic model.
    SIMULATED_EMBEDDING_DIM: int = 256
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

    # Fallback and retries. Retries are handled by the Groq SDK for transient
    # transport errors; a model fallback is attempted at most once.
    LLM_MAX_RETRIES: int = 1
    LLM_FALLBACK_ENABLED: bool = True
    # How long a fetched list of account-available models is trusted.
    MODEL_AVAILABILITY_TTL_SECONDS: int = 600

    # Routing policy: "complexity" (default heuristic), "fixed-lightweight",
    # or "fixed-complex". Fixed policies are useful as evaluation baselines.
    ROUTING_POLICY: str = "complexity"
    # Estimated prompt tokens above which RAG prompts go to the complex tier.
    ROUTING_CONTEXT_TOKEN_THRESHOLD: int = 2500
    # Number of distinct source documents in the evidence above which a
    # comparative question is treated as multi-document synthesis.
    ROUTING_MULTI_DOC_THRESHOLD: int = 2
    LIGHTWEIGHT_MAX_CONTEXT_TOKENS: int = 8000
    COMPLEX_MAX_CONTEXT_TOKENS: int = 8000

    # Only used by offline tools (eval/run_eval.py) and file-backed tests.
    # The web app keeps every visitor's documents, cache and telemetry in a
    # private in-memory workspace and never writes them to this file.
    DATABASE_PATH: str = str(
        Path(__file__).resolve().parent.parent / "metrics.db"
    )

    # Document ingestion
    MAX_UPLOAD_MB: float = 10.0
    MAX_PDF_PAGES: int = 300
    MAX_DOCUMENT_CHARS: int = 2_000_000
    CHUNK_SIZE_CHARS: int = 900
    CHUNK_OVERLAP_CHARS: int = 150
    MIN_CHUNK_CHARS: int = 40
    EMBEDDING_BATCH_SIZE: int = 32

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_MAX_TOP_K: int = 12
    # Minimum cosine similarity for a chunk to count as evidence. Calibrate
    # per embedding provider (see README, Evaluation).
    RETRIEVAL_MIN_RELEVANCE: float = 0.30

    # RAG cache validity: a semantic (non-exact) cache candidate is reused
    # only when the new question retrieves overlapping evidence.
    CACHE_MIN_EVIDENCE_OVERLAP: float = 0.5

    # Telemetry privacy: when false, query text is replaced by a hash.
    TELEMETRY_STORE_QUERY_TEXT: bool = True

    # Destructive admin endpoints (metrics reset). In production they require
    # this token in the X-Admin-Token header; when unset they are disabled.
    ADMIN_TOKEN: str = ""

    # Rate limits (SlowAPI syntax)
    QUERY_RATE_LIMIT: str = "30/minute"
    UPLOAD_RATE_LIMIT: str = "10/minute"
    SESSION_RATE_LIMIT: str = "20/minute"

    # Visitor sessions. Each browser session gets a private in-memory
    # workspace that is deleted when the visitor resets it, when it has been
    # idle or alive too long, or when the server restarts. The limits keep
    # memory bounded on a small (512 MB) instance.
    SESSION_IDLE_TTL_MINUTES: float = 30.0
    SESSION_MAX_AGE_HOURS: float = 6.0
    SESSION_MAX_ACTIVE: int = 50
    SESSION_MAX_DOCUMENTS: int = 10
    SESSION_MAX_UPLOAD_MB: float = 25.0

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


settings = Settings()
