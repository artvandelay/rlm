#!/usr/bin/env python3
"""
Run benchmark: 5 examples × 4 models (1 baseline + 3 RLM)
Without Xiaomi Mimo v2 Flash
"""

from benchmarks.config import BenchmarkConfig, ModelConfig
from benchmarks.runner import BenchmarkRunner

config = BenchmarkConfig(max_samples=5)
config.models = [
    # Baseline (non-RLM) - runs in parallel
    ModelConfig("GPT-5.1", "openai/gpt-5.1", "openrouter", False),
    # RLM models - run in parallel process pools
    ModelConfig("GPT-4o-mini (RLM)", "openai/gpt-4o-mini", "openrouter", True),
    ModelConfig("Z-AI GLM-4.7 (RLM)", "z-ai/glm-4.7", "openrouter", True),
    ModelConfig("MiniMax M2.1 (RLM)", "minimax/minimax-m2.1", "openrouter", True),
]

if __name__ == "__main__":
    runner = BenchmarkRunner(config)
    runner.load_tasks(["hotpotqa"], shuffle=True)
    runner.run()
