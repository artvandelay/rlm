import json
import os

import requests

from .base_task import BaseTask, BenchmarkExample


class MusiqueTask(BaseTask):
    def __init__(self, split: str = "validation", max_samples: int = None):
        super().__init__("musique", split, max_samples)
        self.url = "https://raw.githubusercontent.com/stonybrooknlp/musique/main/data/musique_ans_v1.0_dev.jsonl"

    def load(self):
        print(f"Loading Musique ({self.split})...")
        data = []

        # Try downloading directly since HF dataset loading is flaky
        cache_file = f"benchmarks/data/musique_{self.split}.jsonl"
        if not os.path.exists("benchmarks/data"):
            os.makedirs("benchmarks/data", exist_ok=True)

        if not os.path.exists(cache_file):
            print(f"Downloading from {self.url}...")
            try:
                response = requests.get(self.url)
                response.raise_for_status()
                with open(cache_file, "w") as f:
                    f.write(response.text)
            except Exception as e:
                print(f"Failed to download Musique: {e}")
                return

        # Load from file
        try:
            with open(cache_file) as f:
                for line in f:
                    data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading Musique file: {e}")
            return

        for item in data:
            context_text = ""
            for p in item["paragraphs"]:
                context_text += f"Title: {p['title']}\n{p['paragraph_text']}\n\n"

            self.examples.append(
                BenchmarkExample(
                    id=item["id"],
                    question=item["question"],
                    context=context_text.strip(),
                    gold_answer=item["answer"],
                    reasoning_steps=[
                        decomp["question"] for decomp in item.get("question_decomposition", [])
                    ],
                )
            )

        print(f"Loaded {len(self.examples)} Musique examples.")
