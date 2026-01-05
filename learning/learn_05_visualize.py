"""
Phase 5: Generate a complex trajectory for visualization.

This creates a rich log file that you can explore in the visualizer
to really understand what's happening under the hood.
"""

import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

print("=" * 70)
print("PHASE 5: Complex Task for Visualization")
print("=" * 70)

logger = RLMLogger(log_dir="./logs", file_name="complex_trajectory")

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
    max_iterations=50,
)

# A multi-step task that will create an interesting trajectory
context_data = {
    "employees": [
        {"name": "Alice", "department": "Engineering", "salary": 120000, "years": 5},
        {"name": "Bob", "department": "Sales", "salary": 90000, "years": 3},
        {"name": "Charlie", "department": "Engineering", "salary": 150000, "years": 8},
        {"name": "Diana", "department": "Sales", "salary": 95000, "years": 4},
        {"name": "Eve", "department": "Engineering", "salary": 110000, "years": 2},
        {"name": "Frank", "department": "Marketing", "salary": 85000, "years": 6},
    ]
}

task = """
Analyze the employee data in the context variable:

Tasks:
1. Calculate the average salary by department
2. Find which department has the highest average salary
3. Generate a brief summary report

Use code to do the analysis. Break down the problem step by step.
"""

print("\nRunning complex analysis task...")
print("This will create a detailed trajectory with multiple iterations.")
print("\n" + "=" * 70 + "\n")

result = rlm.completion(prompt=context_data, root_prompt=task)

print("\n" + "=" * 70)
print("TASK COMPLETE!")
print("=" * 70)
print("\nFinal Report:")
print(result.response)
print(f"\nExecution time: {result.execution_time:.2f}s")
print("Iterations: Check the verbose output above")
print(f"\n📊 Log file saved to: {logger.log_dir}")
print("\n" + "=" * 70)
print("NEXT STEP: Visualize this trajectory!")
print("=" * 70)
print("\n1. Open a new terminal")
print("2. cd visualizer/")
print("3. npm install (first time only)")
print("4. npm run dev")
print("5. Open http://localhost:3001")
print("6. Upload the .jsonl file from ./logs/")
print("\nYou'll see:")
print("  - Each iteration's thought process")
print("  - Code blocks executed")
print("  - Outputs and errors")
print("  - Token usage over time")
print("  - The full reasoning chain")
print("=" * 70)
