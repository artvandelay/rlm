from datasets import load_dataset

from .base_task import BaseTask, BenchmarkExample


class DROPTask(BaseTask):
    """
    DROP (Discrete Reasoning Over Paragraphs) dataset.

    Requires numerical reasoning, counting, sorting, and arithmetic operations
    over text passages. More challenging than simple extraction.

    Examples:
    - "How many touchdowns were scored in the first quarter?"
    - "How many years apart were these two events?"
    """

    def __init__(self, split: str = "validation", max_samples: int = None):
        super().__init__("drop", split, max_samples)

    def load(self):
        print(f"Loading DROP ({self.split})...")
        dataset = load_dataset("drop", split=self.split)

        for item in dataset:
            # DROP provides passage and question
            context_text = item["passage"]

            # Handle multiple valid answers (DROP can have multiple correct answers)
            # We'll use the first answer as gold
            answers = item["answers_spans"]["spans"]
            gold_answer = answers[0] if answers else ""

            self.examples.append(
                BenchmarkExample(
                    id=item["query_id"],
                    question=item["question"],
                    context=context_text.strip(),
                    gold_answer=gold_answer,
                    reasoning_steps=None,
                )
            )

        print(f"Loaded {len(self.examples)} DROP examples.")
