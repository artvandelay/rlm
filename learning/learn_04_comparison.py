"""
Phase 4: Side-by-side comparison - RLM vs Regular LLM.

See the difference in accuracy and capability!
"""

import os

import openai
from dotenv import load_dotenv

from rlm import RLM

load_dotenv()

print("=" * 70)
print("PHASE 4: RLM vs Regular LLM - The Difference")
print("=" * 70)

# The task: something that requires precise calculation
task = """
I have a list of numbers: [123, 456, 789, 234, 567, 890, 345, 678, 901]

Calculate:
1. The sum of all numbers
2. The average
3. The standard deviation
4. How many numbers are above the average

Give exact numerical answers.
"""

print("\nTask:", task)
print("\n" + "=" * 70)

# First: Regular LLM (just text generation)
print("\n[1] REGULAR LLM (OpenAI API directly)")
print("-" * 70)

client = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": task}],
)

regular_answer = response.choices[0].message.content
print("Answer:", regular_answer[:500])  # First 500 chars
print("\n⚠️  Notice: The LLM tries to calculate in its 'head' - may be imprecise!")

# Second: RLM (with code execution)
print("\n" + "=" * 70)
print("\n[2] RLM (with code execution)")
print("-" * 70)

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "openai/gpt-4o-mini",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
    },
    environment="local",
    verbose=True,
)

result = rlm.completion(task)

print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
print("\nRegular LLM:")
print("  - Tries to calculate mentally")
print("  - May make arithmetic errors")
print("  - Can't verify its work")
print("\nRLM:")
print("  - Writes Python code")
print("  - Executes it for exact results")
print("  - Can verify and iterate")
print("\nFinal RLM Answer:", result.response)
print("=" * 70)
