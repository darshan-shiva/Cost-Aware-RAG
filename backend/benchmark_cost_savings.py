
"""Estimate potential LLM inference savings from semantic cache hits.

Uses synthetic benchmark data and configured model prices.
Does not call Groq or modify application telemetry.
Run from the backend directory:
    python benchmark_cost_savings.py
"""

from app.config import settings


# Illustrative workload, not measured production traffic.
TOTAL_QUERIES = 1000
CACHE_HIT_RATES = [0.10, 0.25, 0.40, 0.60, 0.80]

# Example average token usage per uncached query.
AVG_INPUT_TOKENS = 150
AVG_OUTPUT_TOKENS = 300

# Illustrative routing distribution for cache misses.
LIGHTWEIGHT_SHARE = 0.70
COMPLEX_SHARE = 0.30


def estimate_cost(model, input_tokens, output_tokens):
    input_cost = input_tokens / 1000 * model.cost_input_per_1k
    output_cost = output_tokens / 1000 * model.cost_output_per_1k
    return input_cost + output_cost


def main():
    router_prices = [
        ("Lightweight", settings.LIGHTWEIGHT_COST_INPUT_PER_1K,
         settings.LIGHTWEIGHT_COST_OUTPUT_PER_1K, LIGHTWEIGHT_SHARE),
        ("Complex", settings.COMPLEX_COST_INPUT_PER_1K,
         settings.COMPLEX_COST_OUTPUT_PER_1K, COMPLEX_SHARE),
    ]

    # Per-request costs using the project's configured price assumptions.
    per_model_costs = []
    for name, input_rate, output_rate, share in router_prices:
        cost = (
            AVG_INPUT_TOKENS / 1000 * input_rate
            + AVG_OUTPUT_TOKENS / 1000 * output_rate
        )
        per_model_costs.append((name, cost, share))

    weighted_cost = sum(
        cost * share for _, cost, share in per_model_costs
    )

    baseline_cost = TOTAL_QUERIES * weighted_cost

    print("Cost-Aware RAG — Illustrative Cost-Savings Benchmark")
    print("=" * 65)
    print(f"Total queries: {TOTAL_QUERIES:,}")
    print(f"Assumed average input tokens: {AVG_INPUT_TOKENS}")
    print(f"Assumed average output tokens: {AVG_OUTPUT_TOKENS}")
    print(f"Lightweight/complex routing mix: "
          f"{LIGHTWEIGHT_SHARE:.0%}/{COMPLEX_SHARE:.0%}")
    print("\nConfigured estimated cost per uncached query:")
    for name, cost, share in per_model_costs:
        print(f"  {name}: ${cost:.6f} (assumed share: {share:.0%})")

    print(f"\nBaseline estimated spend without caching: ${baseline_cost:.6f}")
    print("\nScenario results")
    print("-" * 65)
    print(
        f"{'Hit rate':>9} {'Cache hits':>11} {'Remaining spend':>18} "
        f"{'Estimated saved':>18} {'Savings':>9}"
    )

    for hit_rate in CACHE_HIT_RATES:
        hits = round(TOTAL_QUERIES * hit_rate)
        remaining_queries = TOTAL_QUERIES - hits
        remaining_cost = remaining_queries * weighted_cost
        saved_cost = baseline_cost - remaining_cost
        savings_pct = (
            saved_cost / baseline_cost * 100 if baseline_cost else 0
        )

        print(
            f"{hit_rate:>8.0%} {hits:>11,} "
            f"${remaining_cost:>17.6f} "
            f"${saved_cost:>17.6f} {savings_pct:>8.1f}%"
        )

    print("\nAssumptions and limitations:")
    print("- All workloads and token counts are illustrative.")
    print("- Prices come from local configuration, not provider billing.")
    print("- Cache hits are assumed to avoid all LLM generation charges.")
    print("- Embedding, storage, and infrastructure costs are excluded.")
    print("- Results are projections, not measured production savings.")


if __name__ == "__main__":
    main()
