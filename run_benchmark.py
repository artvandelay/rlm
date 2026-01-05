#!/usr/bin/env python
import os
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY not found in environment.")
        print("Set it in .env file to run benchmarks.")
        return

    try:
        from benchmarks.runner import BenchmarkConfig, BenchmarkRunner
    except ImportError:
        # Add current dir to path
        sys.path.append(os.getcwd())
        from benchmarks.runner import BenchmarkConfig, BenchmarkRunner

    print("=" * 60)
    print("RLM Multi-Model Benchmark Suite")
    print("=" * 60)
    print("\nComparing:")
    print("  1x Large Regular LLM (baseline)")
    print("  4x RLM models (large → tiny)")
    print(f"\nRunning {10} examples from HotpotQA...")
    print("=" * 60 + "\n")

    # Run with 10 samples
    config = BenchmarkConfig(max_samples=10)
    runner = BenchmarkRunner(config)

    # Only run HotpotQA for now
    runner.load_tasks(["hotpotqa"])
    runner.run()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print(f"Results saved with run_id: {runner.run_id}")
    print("\nTo view detailed report, run:")
    print(
        f"  python benchmarks/viewer.py --file benchmarks/results/hotpot_qa_results_{runner.run_id}.jsonl"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
