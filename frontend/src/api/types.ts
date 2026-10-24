export interface HealthResponse {
  status: string;
  version: string;
  simulated_mode: boolean;
  timestamp: string;
}

export interface QueryRequest {
  query: string;
  similarity_threshold?: number;
  force_refresh?: boolean;
}

export interface QueryResponse {
  request_id: string;
  query: string;
  response: string;
  cache_hit: boolean;
  cached_similarity?: number | null;
  is_cache_simulated: boolean;
  selected_model: string;
  routing_reason: string;
  latency_ms: number;
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_cost: number;
  actual_cost?: number | null;
  timestamp: string;
}

export interface RequestRecord {
  request_id: string;
  timestamp: string;
  query: string;
  response_preview: string;
  cache_hit: boolean;
  cached_similarity?: number | null;
  is_cache_simulated: boolean;
  selected_model: string;
  routing_reason: string;
  latency_ms: number;
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_cost: number;
  actual_cost?: number | null;
  has_error: boolean;
  error_message?: string | null;
}

export interface MetricsSummary {
  total_requests: number;
  cache_hits: number;
  cache_misses: number;
  cache_hit_rate: number;
  total_estimated_spend: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_latency_ms: number;
  routing_breakdown: Record<string, number>;
  cache_breakdown: Record<string, number>;
  is_simulated: boolean;
  error_count: number;
  error_rate: number;
}

// ------------------------------------------------------------------ extended

export interface HealthDetails extends HealthResponse {
  embedding_provider?: string | null;
  llm_configured?: boolean | null;
  database_ok?: boolean | null;
  document_count?: number | null;
  environment?: string | null;
}

export interface ModelStats {
  requests: number;
  avg_latency_ms: number;
  total_estimated_cost: number;
  input_tokens: number;
  output_tokens: number;
}

export interface MetricsSummaryExtended extends MetricsSummary {
  total_estimated_cost_avoided: number;
  total_input_tokens: number;
  total_output_tokens: number;
  mode_breakdown: Record<string, number>;
  outcome_breakdown: Record<string, number>;
  match_type_breakdown: Record<string, number>;
  error_category_breakdown: Record<string, number>;
  model_stats: Record<string, ModelStats>;
  avg_cache_lookup_ms: number | null;
  avg_embedding_ms: number | null;
  avg_retrieval_ms: number | null;
  avg_generation_ms: number | null;
  avg_latency_cache_hit_ms: number | null;
  avg_latency_cache_miss_ms: number | null;
  fallback_count: number;
  pricing_note: string;
}

export interface RequestRecordExtended extends RequestRecord {
  mode?: string;
  collection?: string | null;
  outcome?: string | null;
  cache_match_type?: string | null;
  cache_lookup_ms?: number | null;
  embedding_ms?: number | null;
  retrieval_ms?: number | null;
  generation_ms?: number | null;
  retrieved_chunk_count?: number | null;
  top_retrieval_score?: number | null;
  citation_count?: number | null;
  error_category?: string | null;
  retry_count?: number;
  fallback_model?: string | null;
  estimated_cost_avoided?: number;
  token_source?: string | null;
}

export interface PublicConfig {
  embedding_provider: string;
  embedding_simulated: boolean;
  llm_configured: boolean;
  routing_policy: string;
  models: {
    model_id: string;
    tier: string;
    cost_input_per_1k: number;
    cost_output_per_1k: number;
    max_context_tokens: number;
  }[];
  pricing_is_estimate: boolean;
  cache_similarity_threshold: number;
  retrieval_top_k: number;
  retrieval_max_top_k: number;
  retrieval_min_relevance: number;
  rag_cache_min_evidence_overlap: number;
  max_upload_mb: number;
  allowed_file_types: string[];
  chunk_size_chars: number;
  chunk_overlap_chars: number;
  telemetry_stores_query_text: boolean;
}

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed';

export interface DocumentInfo {
  document_id: string;
  collection: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  content_sha256: string;
  version: number;
  status: DocumentStatus;
  error_message?: string | null;
  page_count: number;
  chunk_count: number;
  char_count: number;
  embedding_model?: string | null;
  needs_reindex: boolean;
  created_at: string;
  updated_at: string;
}

export interface UploadResult {
  document: DocumentInfo;
  outcome: 'created' | 'replaced' | 'duplicate';
  message: string;
}

export interface CollectionInfo {
  name: string;
  document_count: number;
  ready_count: number;
  chunk_count: number;
}

export interface RAGQueryRequest {
  query: string;
  collection: string;
  document_ids?: string[];
  top_k?: number;
  min_relevance?: number;
  similarity_threshold?: number;
  force_refresh?: boolean;
}

export interface Citation {
  label: string;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_id: string;
  chunk_index: number;
  document_version: number;
  relevance_score: number;
  excerpt: string;
}

export interface RetrievedSource {
  label?: string | null;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_id: string;
  relevance_score: number;
  in_prompt: boolean;
  cited: boolean;
}

export type RAGOutcome = 'answered' | 'insufficient_evidence' | 'uncited' | 'no_documents';

export interface RAGQueryResponse {
  request_id: string;
  mode: 'rag';
  query: string;
  collection: string;
  answer: string;
  outcome: RAGOutcome;
  citations: Citation[];
  retrieval: {
    top_k: number;
    min_relevance: number;
    retrieved_count: number;
    relevant_count: number;
    top_score?: number | null;
    sources: RetrievedSource[];
    suspicious_passages: number;
    performed: boolean;
  };
  cache: {
    hit: boolean;
    match_type?: 'exact' | 'semantic' | null;
    similarity?: number | null;
    lookup_ms: number;
    rejection_reason?: string | null;
    original_request_id?: string | null;
    original_timestamp?: string | null;
    bypassed: boolean;
  };
  routing: {
    selected_model?: string | null;
    tier?: string | null;
    reason: string;
    policy: string;
    factors: string[];
    fallback_used: boolean;
    retry_count: number;
  };
  usage: {
    input_tokens: number;
    output_tokens: number;
    token_source: 'provider' | 'none';
    estimated_cost: number;
    estimated_cost_avoided: number;
    actual_cost?: number | null;
  };
  latency: {
    total_ms: number;
    embedding_ms?: number | null;
    cache_lookup_ms?: number | null;
    retrieval_ms?: number | null;
    generation_ms?: number | null;
  };
  timestamp: string;
}
