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
        from benchmarks.config import BenchmarkConfig, ModelConfig
        from benchmarks.runner import BenchmarkRunner
    except ImportError:
        # Add current dir to path
        sys.path.append(os.getcwd())
        from benchmarks.config import BenchmarkConfig, ModelConfig
        from benchmarks.runner import BenchmarkRunner

    print("=" * 60)
    print("RLM Quick Test: 2 Models, 5 Examples")
    print("=" * 60)
    print("\nComparing:")
    print("  1. GPT-4o (Regular) - Baseline")
    print("  2. GPT-4o-mini (RLM) - Small model with RLM")
    print("\nRunning 5 examples from HotpotQA...")
    print("=" * 60 + "\n")

    # Quick test config: 2 models, 5 samples
    config = BenchmarkConfig(max_samples=5)
    config.models = [
        ModelConfig(
            name="GPT-4o (Regular)", model_id="openai/gpt-4o", backend="openrouter", use_rlm=False
        ),
        ModelConfig(
            name="GPT-4o-mini (RLM)",
            model_id="openai/gpt-4o-mini",
            backend="openrouter",
            use_rlm=True,
        ),
    ]

    runner = BenchmarkRunner(config)
    runner.load_tasks(["hotpotqa"])
    runner.run()

    print("\n" + "=" * 60)
    print("Quick test complete!")
    print(f"Results saved with run_id: {runner.run_id}")
    print("\nTo view detailed report, run:")
    print(
        f"  python benchmarks/viewer.py --file benchmarks/results/hotpot_qa_results_{runner.run_id}.jsonl"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
