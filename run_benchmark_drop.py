#!/usr/bin/env python
"""
DROP Dataset Benchmark: Numerical reasoning over paragraphs
- Baseline: openai/gpt-5.1 (non-RLM)
- RLM models: glm-4.7, minimax-m2.1, gpt-4o-mini

DROP requires counting, arithmetic, sorting - perfect for RLM's code execution!
"""
import os
import sys
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not found in environment.")
        print("Set it in .env file to run benchmarks.")
        return
        
    try:
        from benchmarks.runner import BenchmarkRunner
        from benchmarks.config import BenchmarkConfig, ModelConfig
    except ImportError:
        sys.path.append(os.getcwd())
        from benchmarks.runner import BenchmarkRunner
        from benchmarks.config import BenchmarkConfig, ModelConfig

    print("="*70)
    print("DROP Benchmark: Numerical Reasoning Challenge")
    print("="*70)
    print("\nDataset: DROP (Discrete Reasoning Over Paragraphs)")
    print("Challenge: Counting, arithmetic, sorting, comparison")
    print("\nModels:")
    print("  [Baseline - No RLM]")
    print("    1. GPT-5.1 (openai/gpt-5.1)")
    print("\n  [RLM Models - With Code Execution]")
    print("    2. GLM-4.7 (z-ai/glm-4.7)")
    print("    3. MiniMax M2.1 (minimax/minimax-m2.1)")
    print("    4. GPT-4o-mini (openai/gpt-4o-mini)")
    print("\n" + "="*70)
    
    # Configure benchmark
    NUM_SAMPLES = 10
    
    config = BenchmarkConfig(max_samples=NUM_SAMPLES)
    config.models = [
        # Baseline: GPT-5.1 without RLM
        ModelConfig(
            name="GPT-5.1 (Regular)",
            model_id="openai/gpt-5.1",
            backend="openrouter",
            use_rlm=False
        ),
        # RLM Models - should excel at numerical reasoning
        ModelConfig(
            name="GLM-4.7 (RLM)",
            model_id="z-ai/glm-4.7",
            backend="openrouter",
            use_rlm=True
        ),
        ModelConfig(
            name="MiniMax-M2.1 (RLM)",
            model_id="minimax/minimax-m2.1",
            backend="openrouter",
            use_rlm=True
        ),
        ModelConfig(
            name="GPT-4o-mini (RLM)",
            model_id="openai/gpt-4o-mini",
            backend="openrouter",
            use_rlm=True
        ),
    ]
    
    print(f"\nRunning {NUM_SAMPLES} examples from DROP...")
    print("="*70 + "\n")
    
    runner = BenchmarkRunner(config)
    runner.load_tasks(["drop"])
    runner.run()
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print(f"Results saved with run_id: {runner.run_id}")
    print("\nTo view detailed report, run:")
    print(f"  python benchmarks/viewer.py --file benchmarks/results/drop_results_{runner.run_id}.jsonl")
    print("="*70)

if __name__ == "__main__":
    main()

