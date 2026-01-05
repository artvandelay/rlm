"""
Phase 2: Understand iterative reasoning.

RLM can examine data, make decisions, and iterate - not just one-shot generation.
This is where it gets interesting!
"""

import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

print("=" * 70)
print("PHASE 2: Iterative Reasoning - Examining Context")
print("=" * 70)

# Enable logging so we can visualize later
logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "openai/gpt-4o-mini",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
    },
    environment="local",
    logger=logger,
    verbose=True,
)

# Give it a large context that requires analysis
context = {
    "sales_data": [
        {"month": "Jan", "revenue": 10000, "costs": 8000, "region": "North"},
        {"month": "Feb", "revenue": 15000, "costs": 9000, "region": "North"},
        {"month": "Mar", "revenue": 12000, "costs": 8500, "region": "North"},
        {"month": "Jan", "revenue": 8000, "costs": 7000, "region": "South"},
        {"month": "Feb", "revenue": 9000, "costs": 7500, "region": "South"},
        {"month": "Mar", "revenue": 11000, "costs": 8000, "region": "South"},
    ]
}

question = "Which region had better profit growth from Jan to Mar? Show calculations."

print("\nContext: Sales data for 2 regions over 3 months")
print("Question:", question)
print("\n" + "=" * 70)
print("Watch how RLM:")
print("  1. Examines the context structure")
print("  2. Writes code to calculate profits")
print("  3. Compares growth rates")
print("  4. Iterates through the analysis")
print("=" * 70 + "\n")

result = rlm.completion(prompt=context, root_prompt=question)

print("\n" + "=" * 70)
print("FINAL ANSWER:", result.response)
print(f"Time taken: {result.execution_time:.2f}s")
print(f"Log saved to: {logger.log_dir}")
print("=" * 70)
