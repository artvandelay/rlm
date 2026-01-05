from datasets import load_dataset

from .base_task import BaseTask, BenchmarkExample


class SQuADv2Task(BaseTask):
    """
    SQuAD v2.0 (Stanford Question Answering Dataset)

    Reading comprehension dataset with questions that may be unanswerable.
    Tests ability to determine when a question cannot be answered from context.

    Note: This includes ~50% unanswerable questions. For answerable-only,
    filter examples where answers['text'] is not empty.
    """

    def __init__(
        self, split: str = "validation", max_samples: int = None, answerable_only: bool = False
    ):
        super().__init__("squad_v2", split, max_samples)
        self.answerable_only = answerable_only

    def load(self):
        print(f"Loading SQuAD v2 ({self.split})...")
        dataset = load_dataset("squad_v2", split=self.split)

        for item in dataset:
            # Check if answerable
            answers = item["answers"]["text"]
            is_answerable = len(answers) > 0

            if self.answerable_only and not is_answerable:
                continue

            gold_answer = answers[0] if is_answerable else "unanswerable"

            self.examples.append(
                BenchmarkExample(
                    id=item["id"],
                    question=item["question"],
                    context=item["context"].strip(),
                    gold_answer=gold_answer,
                    reasoning_steps=None,
                )
            )

        print(f"Loaded {len(self.examples)} SQuAD v2 examples.")
