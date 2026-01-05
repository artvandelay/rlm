import random

from datasets import load_dataset

from .base_task import BaseTask, BenchmarkExample


class HotpotQATask(BaseTask):
    def __init__(
        self,
        split: str = "validation",
        max_samples: int = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__("hotpot_qa", split, max_samples)
        self.shuffle = shuffle
        self.seed = seed

    def load(self):
        print(f"Loading HotpotQA ({self.split})...")
        # Load 'distractor' configuration: includes hard negative paragraphs
        dataset = load_dataset("hotpot_qa", "distractor", split=self.split)

        for item in dataset:
            # Format context from [title, sentences] pairs
            # context['sentences'] is a list of lists of strings
            # context['title'] is a list of strings

            context_text = ""
            titles = item["context"]["title"]
            sentences = item["context"]["sentences"]

            for title, sent_list in zip(titles, sentences, strict=False):
                paragraph = "".join(sent_list)
                context_text += f"Title: {title}\n{paragraph}\n\n"

            self.examples.append(
                BenchmarkExample(
                    id=item["id"],
                    question=item["question"],
                    context=context_text.strip(),
                    gold_answer=item["answer"],
                    reasoning_steps=None,  # HotpotQA doesn't provide explicit steps in this config
                )
            )

        # Shuffle if requested
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.examples)
            print(f"Shuffled {len(self.examples)} examples with seed={self.seed}")

        print(f"Loaded {len(self.examples)} HotpotQA examples.")
