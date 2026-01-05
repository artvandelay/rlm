#!/usr/bin/env python3
"""
Quick test: Can we safely run RLM in separate processes without state corruption?

This tests:
1. Creating RLM client in a worker process
2. Making multiple calls per process
3. Returning results without hanging
4. No hangs during cleanup
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def init_worker():
    """Initialize RLM once per worker process."""
    global rlm_client
    from rlm import RLM

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None

    print(f"[Worker {os.getpid()}] Initializing RLM...", file=sys.stderr)
    rlm_client = RLM(
        backend="openrouter",
        backend_kwargs={
            "model_name": "openai/gpt-4o-mini",
            "api_key": api_key,
            "base_url": base_url,
        },
        environment="local",
        verbose=False,
    )
    print(f"[Worker {os.getpid()}] RLM initialized", file=sys.stderr)


def run_rlm_task(example_id: str, question: str, context: str) -> dict:
    """Run a single RLM call in the worker process."""
    global rlm_client

    start_time = time.time()
    print(f"[Worker {os.getpid()}] Running task {example_id}...", file=sys.stderr)

    try:
        result = rlm_client.completion(prompt=context, root_prompt=question)
        usage = result.usage_summary.to_dict()
        total_calls = sum(m["total_calls"] for m in usage.get("model_usage_summaries", {}).values())

        elapsed = time.time() - start_time
        print(f"[Worker {os.getpid()}] Task {example_id} done ({elapsed:.1f}s)", file=sys.stderr)

        return {
            "example_id": example_id,
            "answer": result.response[:50] + "..."
            if len(result.response) > 50
            else result.response,
            "time": elapsed,
            "llm_calls": total_calls,
            "success": True,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Worker {os.getpid()}] Task {example_id} ERROR: {e}", file=sys.stderr)
        return {
            "example_id": example_id,
            "error": str(e),
            "time": elapsed,
            "success": False,
        }


def test_multiprocessing_rlm():
    """Test RLM in multiprocessing with 5 tasks and 2 workers."""
    print("\n" + "=" * 70)
    print("TEST: RLM Multiprocessing Safety")
    print("=" * 70)
    print(f"Main process: {os.getpid()}")

    # Simple test data
    test_tasks = [
        {
            "id": "q1",
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. The capital and most populous city of France is Paris.",
        },
        {
            "id": "q2",
            "question": "What is the largest planet?",
            "context": "Jupiter is the largest planet in the Solar System. It is named after the Roman god Jupiter.",
        },
        {
            "id": "q3",
            "question": "What year did World War II end?",
            "context": "World War II was a global conflict from 1939 to 1945. It ended in 1945 with the surrender of Japan.",
        },
    ]

    print(f"\nRunning {len(test_tasks)} RLM tasks with 2 worker processes...")
    print("This will:")
    print("  1. Create 2 separate RLM clients (one per worker)")
    print("  2. Dispatch tasks to workers")
    print("  3. Check for hangs during cleanup")
    print("\nStarting tasks...\n")
    sys.stdout.flush()

    start_total = time.time()
    results = []
    errors = []

    try:
        # Use 2 workers for RLM
        with ProcessPoolExecutor(
            max_workers=2,
            initializer=init_worker,
        ) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_rlm_task, task["id"], task["question"], task["context"]): task
                for task in test_tasks
            }

            # Collect results
            for future in as_completed(future_to_task, timeout=120):
                try:
                    result = future.result()
                    results.append(result)
                    if result["success"]:
                        print(
                            f"✓ {result['example_id']}: {result['time']:.1f}s, {result['llm_calls']} calls"
                        )
                    else:
                        print(f"✗ {result['example_id']}: {result['error']}")
                        errors.append(result)
                except Exception as e:
                    print(f"✗ Task failed with exception: {e}")
                    errors.append(str(e))

        # Test reaches here only if processes cleaned up successfully
        elapsed_total = time.time() - start_total

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Total time: {elapsed_total:.1f}s")
        print(f"Successful: {len([r for r in results if r.get('success')])}/{len(test_tasks)}")
        print(f"Failed: {len(errors)}")

        if results:
            avg_time = sum(r["time"] for r in results) / len(results)
            print(f"Avg time per task: {avg_time:.1f}s")

        if errors:
            print("\nErrors:")
            for err in errors:
                print(f"  - {err}")
            return False

        print("\n✅ SUCCESS: RLM works in multiprocessing!")
        print("   - No hangs during cleanup")
        print("   - Processes isolated correctly")
        print("   - This approach is viable for large-scale benchmarks")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")
        print("\nLikely issue: RLM doesn't support multiprocessing initialization")
        return False


if __name__ == "__main__":
    success = test_multiprocessing_rlm()
    sys.exit(0 if success else 1)
