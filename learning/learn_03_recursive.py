"""
Phase 3: The real power - recursive sub-LM calls.

RLM can break down problems and call the LLM multiple times during execution.
This is where "Recursive Language Models" really shines!
"""

import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

print("=" * 70)
print("PHASE 3: Recursive Sub-Calls - The Real Magic")
print("=" * 70)

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
    max_iterations=40,  # Allow more iterations for complex tasks
)

# A task that benefits from decomposition
context = """
Review 1: "This product is absolutely amazing! Best purchase ever. 5 stars!"
Review 2: "Terrible quality. Broke after one day. Very disappointed."
Review 3: "It's okay, nothing special. Does what it says but overpriced."
Review 4: "Love it! Great value for money. Highly recommend to everyone."
Review 5: "Worst customer service. Product is fine but support is awful."
"""

question = """
Analyze the sentiment of each review (positive/negative/neutral).
Then calculate the overall sentiment distribution.

HINT: You can use llm_query() to analyze each review separately!
Example: sentiment = llm_query(f"Is this review positive, negative, or neutral? Reply with just one word: {review}")
"""

print("\nContext: 5 product reviews")
print("Task:", question)
print("\n" + "=" * 70)
print("Watch how RLM:")
print("  1. Sees it can break down the problem")
print("  2. Calls llm_query() for EACH review")
print("  3. Aggregates the results")
print("  4. Returns the final analysis")
print("\nThis is RECURSIVE reasoning!")
print("=" * 70 + "\n")

result = rlm.completion(prompt=context, root_prompt=question)

print("\n" + "=" * 70)
print("FINAL ANSWER:")
print(result.response)
print(f"\nTime taken: {result.execution_time:.2f}s")
print(f"Token usage: {result.usage_summary.to_dict()}")
print(f"\nLog saved to: {logger.log_dir}")
print("=" * 70)
