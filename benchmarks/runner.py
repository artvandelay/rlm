import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import openai

from rlm import RLM
from .config import BenchmarkConfig, ModelConfig
from .tasks.hotpotqa import HotpotQATask
from .tasks.musique import MusiqueTask
from .evaluators.metrics import f1_score, exact_match_score

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.tasks = []
        # Generate unique run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _create_client(self, model_config: ModelConfig):
        """Create either RLM or regular client based on config."""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
        
        # Fallback to OpenRouter
        if not api_key and os.getenv("OPENROUTER_API_KEY"):
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
        
        if model_config.use_rlm:
            # Create RLM instance
            return RLM(
                backend=model_config.backend,
                backend_kwargs={
                    "model_name": model_config.model_id,
                    "api_key": api_key,
                    "base_url": base_url
                },
                environment="local",
                verbose=False
            )
        else:
            # Create regular OpenAI client
            return openai.OpenAI(api_key=api_key, base_url=base_url)

    def load_tasks(self, task_names: List[str]):
        for name in task_names:
            if name == "hotpotqa":
                self.tasks.append(HotpotQATask(max_samples=self.config.max_samples))
            elif name == "musique":
                self.tasks.append(MusiqueTask(max_samples=self.config.max_samples))
            else:
                print(f"Unknown task: {name}")

    def run_model(self, client, model_config: ModelConfig, question: str, context: str) -> Dict[str, Any]:
        """Run a single model (RLM or regular) on a question."""
        start_time = time.time()
        
        try:
            if model_config.use_rlm:
                # RLM: context as prompt, question as root_prompt
                result = client.completion(prompt=context, root_prompt=question)
                duration = time.time() - start_time
                
                # Extract LLM call count from usage summary
                usage = result.usage_summary.to_dict()
                total_calls = sum(
                    model_usage['total_calls'] 
                    for model_usage in usage.get('model_usage_summaries', {}).values()
                )
                
                return {
                    "answer": result.response,
                    "time": duration,
                    "model": model_config.model_id,
                    "llm_calls": total_calls,
                    "usage": usage
                }
            else:
                # Regular LLM: standard prompt
                full_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based on the context. Be concise."
                
                response = client.chat.completions.create(
                    model=model_config.model_id,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.0
                )
                duration = time.time() - start_time
                
                return {
                    "answer": response.choices[0].message.content,
                    "time": duration,
                    "model": model_config.model_id,
                    "llm_calls": 1,  # Regular LLM always makes 1 call
                }
        except Exception as e:
            return {
                "answer": f"Error: {e}",
                "time": 0,
                "error": str(e),
                "llm_calls": 0
            }

    def run(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize clients for all models
        clients = {}
        for model_config in self.config.models:
            print(f"Initializing {model_config.name}...")
            clients[model_config.name] = self._create_client(model_config)
        
        for task in self.tasks:
            print(f"\nRunning Task: {task.dataset_name}")
            examples = task.get_examples()
            results = []
            
            for example in tqdm(examples, desc=f"{task.dataset_name}"):
                result_entry = {
                    "id": example.id,
                    "question": example.question,
                    "gold_answer": example.gold_answer,
                    "models": {}
                }
                
                # Run each model
                for model_config in self.config.models:
                    client = clients[model_config.name]
                    model_result = self.run_model(client, model_config, example.question, example.context)
                    
                    # Ensure answer is string
                    if not isinstance(model_result['answer'], str):
                        model_result['answer'] = str(model_result['answer'])
                    
                    # Calculate metrics
                    em = exact_match_score(model_result['answer'], example.gold_answer)
                    f1 = f1_score(model_result['answer'], example.gold_answer)
                    
                    result_entry["models"][model_config.name] = {
                        **model_result,
                        "em": em,
                        "f1": f1
                    }
                
                results.append(result_entry)
            
            # Save results with run_id
            output_file = os.path.join(
                self.config.output_dir, 
                f"{task.dataset_name}_results_{self.run_id}.jsonl"
            )
            with open(output_file, "w") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Results Summary for {task.dataset_name}:")
            print(f"{'='*60}")
            for model_config in self.config.models:
                avg_f1 = sum(r["models"][model_config.name]["f1"] for r in results) / len(results)
                avg_calls = sum(r["models"][model_config.name]["llm_calls"] for r in results) / len(results)
                print(f"{model_config.name:<25} | F1: {avg_f1:.3f} | Avg Calls: {avg_calls:.1f}")
            print(f"{'='*60}")
            print(f"Saved details to {output_file}")

if __name__ == "__main__":
    config = BenchmarkConfig()
    runner = BenchmarkRunner(config)
    runner.load_tasks(["hotpotqa"])
    runner.run()
