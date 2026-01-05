import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str  # Display name
    model_id: str  # API model ID
    backend: str  # "openrouter" or "openai"
    use_rlm: bool  # True for RLM, False for regular LLM


@dataclass
class BenchmarkConfig:
    # Models to compare
    models: list[ModelConfig] = None

    # Run config
    max_samples: int = 10
    output_dir: str = "benchmarks/results"

    def __post_init__(self):
        if self.models is None:
            # Default: 1 large regular LLM vs 4 RLM models (ranging from large to tiny)
            self.models = [
                # Baseline: Large model without RLM
                ModelConfig(
                    name="GPT-4o (Regular)",
                    model_id="openai/gpt-4o",
                    backend="openrouter",
                    use_rlm=False,
                ),
                # RLM models: Large to tiny
                ModelConfig(
                    name="GPT-4o-mini (RLM)",
                    model_id="openai/gpt-4o-mini",
                    backend="openrouter",
                    use_rlm=True,
                ),
                ModelConfig(
                    name="Qwen-2.5-7B (RLM)",
                    model_id="qwen/qwen-2.5-7b-instruct",
                    backend="openrouter",
                    use_rlm=True,
                ),
                ModelConfig(
                    name="Llama-3.2-3B (RLM)",
                    model_id="meta-llama/llama-3.2-3b-instruct",
                    backend="openrouter",
                    use_rlm=True,
                ),
                ModelConfig(
                    name="Gemma-2-2B (RLM)",
                    model_id="google/gemma-2-2b-it",
                    backend="openrouter",
                    use_rlm=True,
                ),
            ]

    @property
    def api_key(self):
        return os.getenv("OPENROUTER_API_KEY")
