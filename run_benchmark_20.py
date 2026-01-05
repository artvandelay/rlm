#!/usr/bin/env python3
"""
Run benchmark: 20 examples × 5 models (1 baseline + 4 RLM)

Expected time: ~13-15 minutes
- Baseline (parallel): ~15 seconds for all 20 examples
- Each RLM model (sequential): ~3-4 minutes per model
"""

from benchmarks.config import BenchmarkConfig, ModelConfig
from benchmarks.runner import BenchmarkRunner

config = BenchmarkConfig(max_samples=20)
config.models = [
    # Baseline (non-RLM) - runs in parallel
    ModelConfig("GPT-5.1", "openai/gpt-5.1", "openrouter", False),
    # RLM models - run sequentially for stability
    ModelConfig("GPT-4o-mini (RLM)", "openai/gpt-4o-mini", "openrouter", True),
    ModelConfig("Xiaomi Mimo v2 Flash (RLM)", "xiaomi/mimo-v2-flash", "openrouter", True),
    ModelConfig("Z-AI GLM-4.7 (RLM)", "z-ai/glm-4.7", "openrouter", True),
    ModelConfig("MiniMax M2.1 (RLM)", "minimax/minimax-m2.1", "openrouter", True),
]

if __name__ == "__main__":
    runner = BenchmarkRunner(config)
    runner.load_tasks(["hotpotqa"], shuffle=True)
    runner.run()
