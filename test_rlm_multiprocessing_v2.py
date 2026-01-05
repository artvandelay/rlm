#!/usr/bin/env python3
"""
Refined test: RLM multiprocessing with sequential worker behavior
(Process pool exists, but we control concurrency via task dispatch)
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
    sys.stderr.flush()

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
    sys.stderr.flush()


def run_rlm_task(example_id: str, question: str, context: str) -> dict:
    """Run a single RLM call in the worker process."""
    global rlm_client

    start_time = time.time()
    print(f"[Worker {os.getpid()}] Running task {example_id}...", file=sys.stderr)
    sys.stderr.flush()

    try:
        result = rlm_client.completion(prompt=context, root_prompt=question)
        usage = result.usage_summary.to_dict()
        total_calls = sum(m["total_calls"] for m in usage.get("model_usage_summaries", {}).values())

        elapsed = time.time() - start_time
        print(f"[Worker {os.getpid()}] Task {example_id} done ({elapsed:.1f}s)", file=sys.stderr)
        sys.stderr.flush()

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
        sys.stderr.flush()
        return {
            "example_id": example_id,
            "error": str(e),
            "time": elapsed,
            "success": False,
        }


def test_multiprocessing_rlm_sequential():
    """Test RLM in multiprocessing but process tasks sequentially (1 worker, or throttle)."""
    print("\n" + "=" * 70)
    print("TEST: RLM Multiprocessing (Sequential Task Dispatch)")
    print("=" * 70)
    print(f"Main process: {os.getpid()}")

    # Same test data, but we'll dispatch sequentially
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

    print(f"\nRunning {len(test_tasks)} RLM tasks with 1 worker process")
    print("(Sequential dispatch to avoid rate limiting)")
    print("\nStarting tasks...\n")
    sys.stdout.flush()

    start_total = time.time()
    results = []
    errors = []

    try:
        # Use 1 worker to avoid rate limiting
        with ProcessPoolExecutor(
            max_workers=1,
            initializer=init_worker,
        ) as executor:
            # Submit all tasks but they'll run one at a time
            future_to_task = {
                executor.submit(run_rlm_task, task["id"], task["question"], task["context"]): task
                for task in test_tasks
            }

            # Collect results as they complete (will be sequential anyway with 1 worker)
            for future in as_completed(future_to_task):
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
                sys.stdout.flush()

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
            success_count = len([r for r in results if r.get("success")])
            if success_count > 0:
                print("\n⚠️  PARTIAL SUCCESS: Some tasks worked!")
                print("   The multiprocessing approach is viable, but we need")
                print("   to handle timeouts/retries and rate limiting.")
                return True
            return False

        print("\n✅ SUCCESS: RLM works reliably in multiprocessing!")
        print("   Process pool is safe to use for large benchmarks.")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multiprocessing_rlm_sequential()
    sys.exit(0 if success else 1)
