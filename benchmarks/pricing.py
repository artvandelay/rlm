"""
OpenRouter pricing per million tokens (input/output).
Prices are in USD per million tokens.

Source: https://openrouter.ai/models (as of Jan 2025)
Prices may need to be updated if OpenRouter changes pricing.
"""

# Pricing: (input_price_per_million, output_price_per_million)
# Source: https://openrouter.ai/models (as of Jan 2025)
PRICING = {
    "openai/gpt-5.1": (
        1.25,
        10.0,
    ),  # https://openrouter.ai/models/openai/gpt-5.1 - $1.25/M input, $10/M output
    "openai/gpt-4o-mini": (0.15, 0.6),  # https://openrouter.ai/models/openai/gpt-4o-mini
    "z-ai/glm-4.7": (0.16, 0.80),  # https://openrouter.ai/z-ai/glm-4.7
    "minimax/minimax-m2.1": (0.12, 0.48),  # https://openrouter.ai/models/minimax/minimax-m2.1
    "xiaomi/mimo-v2-flash": (0.10, 0.10),  # Not available (404 error)
}

# Fallback pricing for unknown models
DEFAULT_PRICING = (0.50, 2.0)  # Conservative estimate


def get_pricing(model_id: str) -> tuple[float, float]:
    """Get pricing for a model. Returns (input_price, output_price) per million tokens."""
    return PRICING.get(model_id, DEFAULT_PRICING)


def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """Calculate cost in USD for given token usage."""
    input_price, output_price = get_pricing(model_id)

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + output_cost
