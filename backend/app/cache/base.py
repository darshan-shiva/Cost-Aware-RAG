from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
from pydantic import BaseModel, Field

class CacheEntry(BaseModel):
    key: str
    query: str
    response: str
    embedding: List[float]
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)
    hit_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CacheLookupResult(BaseModel):
    is_hit: bool
    similarity_score: Optional[float] = None
    matched_entry: Optional[CacheEntry] = None
    is_simulated: bool = True
    lookup_latency_ms: float = 0.0

class BaseCache(ABC):
    @abstractmethod
    def lookup(self, query: str, threshold: Optional[float] = None) -> CacheLookupResult:
        """Lookup query against stored entries using similarity threshold."""
        pass

    @abstractmethod
    def store(
        self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> CacheEntry:
        """Store query and response with its vector embedding."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached entries."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and capacity statistics."""
        pass
