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
  cache_breakdown: {
    hits: number;
    misses: number;
  };
  is_simulated: boolean;
}
