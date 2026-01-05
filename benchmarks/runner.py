"""
Benchmark runner with optimized execution strategy:
- Non-RLM models: Parallel execution via threads (thread-safe HTTP calls)
- RLM models: Parallel execution via multiprocessing (isolated process pools per model)
  Each RLM model runs in its own process pool, allowing multiple models to execute concurrently.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import openai
from tqdm import tqdm

from rlm import RLM

from .config import BenchmarkConfig, ModelConfig
from .evaluators.metrics import exact_match_score, f1_score

# Global variables for multiprocessing workers (initialized per process)
_rlm_worker_client = None
_rlm_worker_config = None


def _init_rlm_worker(model_config_dict: dict):
    """Initialize RLM client in worker process. Must be top-level for pickling."""
    global _rlm_worker_client, _rlm_worker_config

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None

    _rlm_worker_config = model_config_dict
    _rlm_worker_client = RLM(
        backend=model_config_dict["backend"],
        backend_kwargs={
            "model_name": model_config_dict["model_id"],
            "api_key": api_key,
            "base_url": base_url,
        },
        environment="local",
        verbose=False,
    )


def _run_rlm_task_in_process(task_data: dict) -> dict:
    """Execute single RLM call in worker process. Must be top-level for pickling."""
    global _rlm_worker_client

    question = task_data["question"]
    context = task_data["context"]
    model_id = task_data["model_id"]

    start_time = time.time()

    try:
        result = _rlm_worker_client.completion(prompt=context, root_prompt=question)
        usage = result.usage_summary.to_dict()
        total_calls = sum(m["total_calls"] for m in usage.get("model_usage_summaries", {}).values())

        return {
            "answer": result.response,
            "time": time.time() - start_time,
            "model": model_id,
            "llm_calls": total_calls,
            "usage": usage,
        }
    except Exception as e:
        return {"answer": f"Error: {e}", "time": 0, "error": str(e), "llm_calls": 0}


# Task Registry: Add new datasets here
def _load_task_registry():
    """Registry of available tasks. Lazy imports to avoid circular dependencies."""
    from .tasks.boolq import BoolQTask
    from .tasks.drop import DROPTask
    from .tasks.hotpotqa import HotpotQATask
    from .tasks.musique import MusiqueTask
    from .tasks.squad_v2 import SQuADv2Task

    return {
        "hotpotqa": lambda config, shuffle: HotpotQATask(
            max_samples=config.max_samples, shuffle=shuffle
        ),
        "musique": lambda config, shuffle: MusiqueTask(max_samples=config.max_samples),
        "drop": lambda config, shuffle: DROPTask(max_samples=config.max_samples),
        "squad_v2": lambda config, shuffle: SQuADv2Task(
            max_samples=config.max_samples, answerable_only=True
        ),
        "boolq": lambda config, shuffle: BoolQTask(max_samples=config.max_samples),
    }


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.tasks = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._task_registry = _load_task_registry()

        # Cache API credentials
        self._api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self._base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None

    def _create_openai_client(self):
        """Create a regular OpenAI client (thread-safe)."""
        return openai.OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _create_rlm_client(self, model_config: ModelConfig):
        """Create an RLM client (NOT thread-safe, use one per model sequentially)."""
        return RLM(
            backend=model_config.backend,
            backend_kwargs={
                "model_name": model_config.model_id,
                "api_key": self._api_key,
                "base_url": self._base_url,
            },
            environment="local",
            verbose=False,
        )

    def load_tasks(self, task_names: list[str], shuffle: bool = False):
        """Load tasks by name from registry."""
        for name in task_names:
            if name in self._task_registry:
                self.tasks.append(self._task_registry[name](self.config, shuffle))
            else:
                available = ", ".join(self._task_registry.keys())
                print(f"Unknown task '{name}'. Available: {available}")

    def _run_openai_call(
        self, client, model_id: str, question: str, context: str
    ) -> dict[str, Any]:
        """Execute a single OpenAI API call (thread-safe)."""
        start_time = time.time()
        try:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based on the context. Be concise."
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            # Extract usage data if available
            usage_data = {}
            if hasattr(response, "usage") and response.usage:
                usage_data = {
                    "model_usage_summaries": {
                        model_id: {
                            "total_calls": 1,
                            "total_input_tokens": response.usage.prompt_tokens or 0,
                            "total_output_tokens": response.usage.completion_tokens or 0,
                        }
                    }
                }

            return {
                "answer": response.choices[0].message.content,
                "time": time.time() - start_time,
                "model": model_id,
                "llm_calls": 1,
                "usage": usage_data,
            }
        except Exception as e:
            return {"answer": f"Error: {e}", "time": 0, "error": str(e), "llm_calls": 0}

    def _run_rlm_call(self, client, model_id: str, question: str, context: str) -> dict[str, Any]:
        """Execute a single RLM call (NOT thread-safe)."""
        start_time = time.time()
        try:
            result = client.completion(prompt=context, root_prompt=question)
            usage = result.usage_summary.to_dict()
            total_calls = sum(
                m["total_calls"] for m in usage.get("model_usage_summaries", {}).values()
            )
            return {
                "answer": result.response,
                "time": time.time() - start_time,
                "model": model_id,
                "llm_calls": total_calls,
                "usage": usage,
            }
        except Exception as e:
            return {"answer": f"Error: {e}", "time": 0, "error": str(e), "llm_calls": 0}

    def _run_non_rlm_models_parallel(self, examples: list) -> dict:
        """Run all non-RLM models on all examples in parallel."""
        non_rlm_models = [m for m in self.config.models if not m.use_rlm]
        if not non_rlm_models:
            return {}

        results = {}
        client = self._create_openai_client()
        total_tasks = len(examples) * len(non_rlm_models)

        print(
            f"  Running {len(non_rlm_models)} non-RLM model(s) on {len(examples)} examples in parallel..."
        )
        sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=min(total_tasks, 20)) as executor:
            future_to_key = {
                executor.submit(
                    self._run_openai_call,
                    client,
                    model_config.model_id,
                    example.question,
                    example.context,
                ): (example.id, model_config.name)
                for example in examples
                for model_config in non_rlm_models
            }

            for future in tqdm(
                as_completed(future_to_key),
                total=len(future_to_key),
                desc="  Non-RLM",
                leave=False,
            ):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = {
                        "answer": f"Error: {e}",
                        "time": 0,
                        "error": str(e),
                        "llm_calls": 0,
                    }

        return results

    def _run_rlm_models_parallel(self, examples: list) -> dict:
        """Run all RLM models in parallel using separate process pools per model."""
        rlm_models = [m for m in self.config.models if m.use_rlm]
        if not rlm_models:
            return {}

        results = {}

        print(
            f"  Running {len(rlm_models)} RLM model(s) on {len(examples)} examples in parallel (process pools)..."
        )
        sys.stdout.flush()

        # Create a ProcessPoolExecutor for each RLM model
        # Each executor uses 1 worker to avoid rate limiting (as tested)
        executors = {}
        all_futures = {}

        for model_config in rlm_models:
            # Convert ModelConfig to dict for pickling
            model_config_dict = {
                "name": model_config.name,
                "model_id": model_config.model_id,
                "backend": model_config.backend,
            }

            executor = ProcessPoolExecutor(
                max_workers=1,
                initializer=_init_rlm_worker,
                initargs=(model_config_dict,),
            )
            executors[model_config.name] = executor

            # Submit all examples for this model
            for example in examples:
                task_data = {
                    "example_id": example.id,
                    "question": example.question,
                    "context": example.context,
                    "model_id": model_config.model_id,
                }
                future = executor.submit(_run_rlm_task_in_process, task_data)
                all_futures[future] = (example.id, model_config.name)

        # Collect results from all executors as they complete
        try:
            for future in tqdm(
                as_completed(all_futures),
                total=len(all_futures),
                desc="  RLM",
                leave=False,
            ):
                key = all_futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = {
                        "answer": f"Error: {e}",
                        "time": 0,
                        "error": str(e),
                        "llm_calls": 0,
                    }
        finally:
            # Shutdown all executors
            for executor in executors.values():
                executor.shutdown(wait=True)

        return results

    def _assemble_results(self, examples: list, all_results: dict) -> list[dict]:
        """Assemble per-example results."""
        results = []

        for example in examples:
            result_entry = {
                "id": example.id,
                "question": example.question,
                "gold_answer": example.gold_answer,
                "models": {},
            }

            for model_config in self.config.models:
                key = (example.id, model_config.name)
                if key not in all_results:
                    print(f"Warning: Missing result for {key}")
                    continue

                model_result = all_results[key]
                if not isinstance(model_result["answer"], str):
                    model_result["answer"] = str(model_result["answer"])

                em = exact_match_score(model_result["answer"], example.gold_answer)
                f1 = f1_score(model_result["answer"], example.gold_answer)

                result_entry["models"][model_config.name] = {**model_result, "em": em, "f1": f1}

            results.append(result_entry)

        return results

    def _save_results(self, results: list[dict], task_name: str) -> str:
        """Save results to JSONL file."""
        output_file = os.path.join(
            self.config.output_dir, f"{task_name}_results_{self.run_id}.jsonl"
        )

        with open(output_file, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")

        print(f"✓ Saved {len(results)} results to {output_file}")
        return output_file

    def _print_summary(self, results: list[dict], task_name: str, output_file: str):
        """Print benchmark summary statistics."""
        print(f"\n{'=' * 70}")
        print(f"Results Summary for {task_name}:")
        print(f"{'=' * 70}")

        for model_config in self.config.models:
            mode = "RLM" if model_config.use_rlm else "Direct"
            avg_f1 = sum(r["models"][model_config.name]["f1"] for r in results) / len(results)
            avg_time = sum(r["models"][model_config.name]["time"] for r in results) / len(results)
            avg_calls = sum(r["models"][model_config.name]["llm_calls"] for r in results) / len(
                results
            )
            print(
                f"{model_config.name:<30} ({mode:>6}) | F1: {avg_f1:.3f} | Time: {avg_time:>5.1f}s | Calls: {avg_calls:.1f}"
            )

        print(f"{'=' * 70}")
        print(f"Details: {output_file}")

    def run(self):
        """Run all loaded tasks with optimized execution strategy."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Separate models by type
        non_rlm = [m for m in self.config.models if not m.use_rlm]
        rlm = [m for m in self.config.models if m.use_rlm]

        print("=" * 70)
        print("BENCHMARK CONFIGURATION")
        print("=" * 70)
        print(f"Non-RLM models (parallel threads): {len(non_rlm)}")
        for m in non_rlm:
            print(f"  - {m.name}")
        print(f"RLM models (parallel processes):   {len(rlm)}")
        for m in rlm:
            print(f"  - {m.name}")
        print("=" * 70)
        sys.stdout.flush()

        # Run each task
        for task in self.tasks:
            print(f"\n📋 Task: {task.dataset_name}")
            sys.stdout.flush()
            examples = task.get_examples()
            print(f"   Examples: {len(examples)}")

            # Run non-RLM models in parallel (threads)
            all_results = {}
            if non_rlm:
                non_rlm_results = self._run_non_rlm_models_parallel(examples)
                all_results.update(non_rlm_results)

            # Run RLM models in parallel (process pools)
            if rlm:
                rlm_results = self._run_rlm_models_parallel(examples)
                all_results.update(rlm_results)

            # Assemble, save, and summarize
            results = self._assemble_results(examples, all_results)
            output_file = self._save_results(results, task.dataset_name)
            self._print_summary(results, task.dataset_name, output_file)

        print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    from .config import BenchmarkConfig

    config = BenchmarkConfig()
    runner = BenchmarkRunner(config)
    runner.load_tasks(["hotpotqa"])
    runner.run()
