"""
Published DigitalOcean Serverless Inference rates, USD per 1M tokens.

Source: https://docs.digitalocean.com/products/inference/details/pricing/
Verified: 2026-09-22. Re-check before publishing and date-stamp the article.

Note: DigitalOcean-hosted (open-source) models have no cache-WRITE column on
the pricing page, unlike the Anthropic and OpenAI rows. We therefore assume
cache writes are billed at the standard input rate with no premium. If you see
a 'cache_created_input_tokens' > 0 with a cost that doesn't reconcile against
your billing page, revisit this assumption before publishing.
"""

RATES = {
    "deepseek-v4.1-flash": {
        "input": 0.30,
        "output": 1.20,
        "cache_read": 0.006,
        "context": 1_048_576,
        # Sent as reasoning_effort. Dropped automatically if the API rejects it.
        "default_effort": "low",
        "label": "DeepSeek V4.1 Flash",
    },
    "deepseek-v4-flash-0731": {
        "input": 0.08,
        "output": 0.252,
        "cache_read": 0.0252,
        "context": 1_048_576,
        "default_effort": None,
        "label": "DeepSeek V4 Flash 0731",
    },
    "glm-5.3-flash": {
        "input": 0.15,
        "output": 0.50,
        "cache_read": 0.03,
        "context": 1_048_576,
        "default_effort": "low",
        "label": "GLM-5.3 Flash",
    },
}

# Reasoning effort buckets accepted by deepseek-v4.1-flash on DO, as reported
# by the API's own validation error. The model card documents a continuous
# 1-100 integer instead; that gap is a finding, not a bug in this harness.
EFFORT_BUCKETS = ["low", "high", "xhigh", "max"]


def cost(model: str, prompt_tokens: int, cached_tokens: int,
         completion_tokens: int,
         input_rate: float = None, cache_rate: float = None,
         output_rate: float = None) -> float:
    """Billed USD for one call. Rate overrides drive the counterfactuals."""
    r = RATES[model]
    input_rate = r["input"] if input_rate is None else input_rate
    cache_rate = r["cache_read"] if cache_rate is None else cache_rate
    output_rate = r["output"] if output_rate is None else output_rate

    uncached = max(0, prompt_tokens - cached_tokens)
    return (
        uncached * input_rate
        + cached_tokens * cache_rate
        + completion_tokens * output_rate
    ) / 1_000_000


def input_side_breakeven(model_a: str, model_b: str) -> float:
    """
    Cache hit rate h at which model_a's INPUT bill equals model_b's, ignoring
    output. Returns None when no crossover exists in [0, 1].

        (1-h)*in_a + h*cache_a == (1-h)*in_b + h*cache_b
    """
    a, b = RATES[model_a], RATES[model_b]
    num = a["input"] - b["input"]
    den = num + (b["cache_read"] - a["cache_read"])
    if den == 0:
        return None
    h = num / den
    return h if 0.0 <= h <= 1.0 else None


def full_breakeven(model_a: str, model_b: str,
                   prompt_tokens: float, completion_tokens: float):
    """
    Same crossover, but including output tokens at a measured prompt:completion
    ratio. Output cost is independent of h, so if a's output premium exceeds the
    largest possible input saving, there is no hit rate that rescues it.

    Returns (h, None) on a crossover, or (None, reason) when there isn't one.
    """
    a, b = RATES[model_a], RATES[model_b]
    out_delta = (a["output"] - b["output"]) * completion_tokens / 1_000_000

    def in_cost(m, h):
        r = RATES[m]
        return prompt_tokens * ((1 - h) * r["input"] + h * r["cache_read"]) / 1e6

    # Cost difference a-b is monotonic in h; check the endpoints.
    lo = in_cost(model_a, 0.0) - in_cost(model_b, 0.0) + out_delta
    hi = in_cost(model_a, 1.0) - in_cost(model_b, 1.0) + out_delta

    if lo <= 0 and hi <= 0:
        return None, f"{RATES[model_a]['label']} is cheaper at every hit rate"
    if lo > 0 and hi > 0:
        return None, (
            f"{RATES[model_a]['label']} is more expensive at every hit rate; "
            f"the output premium ({out_delta*1000:.4f} m$/call) exceeds the "
            f"maximum achievable input saving"
        )

    # Bisect.
    a_h, b_h = 0.0, 1.0
    for _ in range(200):
        mid = (a_h + b_h) / 2
        v = in_cost(model_a, mid) - in_cost(model_b, mid) + out_delta
        if (v > 0) == (lo > 0):
            a_h = mid
        else:
            b_h = mid
    return (a_h + b_h) / 2, None
