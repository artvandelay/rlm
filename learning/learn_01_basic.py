"""
Phase 1: See RLM in action with a simple task that benefits from code execution.

This shows the core difference: RLM can write and execute code to solve problems,
rather than just generating text.
"""
import os
from dotenv import load_dotenv
from rlm import RLM

load_dotenv()

print("=" * 70)
print("PHASE 1: Basic RLM - Code Execution")
print("=" * 70)

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "openai/gpt-4o-mini",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
    },
    environment="local",
    verbose=True,  # Watch what happens!
)

# Task: Something that's easy with code but hard to do in your head
prompt = """
Calculate the sum of all prime numbers between 1 and 1000.
Show your work.
"""

print("\nPrompt:", prompt)
print("\n" + "=" * 70)
print("Watch how RLM:")
print("  1. Writes Python code to find primes")
print("  2. Executes the code")
print("  3. Returns the answer")
print("=" * 70 + "\n")

result = rlm.completion(prompt)

print("\n" + "=" * 70)
print("FINAL ANSWER:", result.response)
print(f"Time taken: {result.execution_time:.2f}s")
print("=" * 70)

