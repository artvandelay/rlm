#!/usr/bin/env python3
"""Run final benchmark: 10 examples, 5 models (1 baseline + 4 RLM)"""

from benchmarks.config import BenchmarkConfig, ModelConfig
from benchmarks.runner import BenchmarkRunner

config = BenchmarkConfig(max_samples=10)
config.models = [
    ModelConfig("GPT-5.1", "openai/gpt-5.1", "openrouter", False),
    ModelConfig("GPT-4o-mini (RLM)", "openai/gpt-4o-mini", "openrouter", True),
    ModelConfig("Xiaomi Mimo v2 Flash (RLM)", "xiaomi/mimo-v2-flash", "openrouter", True),
    ModelConfig("Z-AI GLM-4.7 (RLM)", "z-ai/glm-4.7", "openrouter", True),
    ModelConfig("MiniMax M2.1 (RLM)", "minimax/minimax-m2.1", "openrouter", True),
]

print("=" * 70)
print("FINAL BENCHMARK: 10 examples × 5 models (1 baseline + 4 RLM)")
print("=" * 70)

runner = BenchmarkRunner(config)
runner.load_tasks(["hotpotqa"], shuffle=True)
runner.run()

print("\n" + "=" * 70)
print("✓ Benchmark Complete!")
print("=" * 70)
