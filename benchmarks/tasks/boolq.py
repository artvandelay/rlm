from datasets import load_dataset

from .base_task import BaseTask, BenchmarkExample


class BoolQTask(BaseTask):
    """
    BoolQ (Boolean Questions) dataset.

    Yes/No questions from natural search queries paired with Wikipedia passages.
    Tests reading comprehension and binary classification.

    Simpler than multi-hop QA but good for testing basic reasoning.
    """

    def __init__(self, split: str = "validation", max_samples: int = None):
        super().__init__("boolq", split, max_samples)

    def load(self):
        print(f"Loading BoolQ ({self.split})...")
        dataset = load_dataset("boolq", split=self.split)

        for item in dataset:
            # Convert boolean to yes/no string
            gold_answer = "yes" if item["answer"] else "no"

            self.examples.append(
                BenchmarkExample(
                    id=str(item["idx"]) if "idx" in item else str(hash(item["question"])),
                    question=item["question"],
                    context=item["passage"].strip(),
                    gold_answer=gold_answer,
                    reasoning_steps=None,
                )
            )

        print(f"Loaded {len(self.examples)} BoolQ examples.")
