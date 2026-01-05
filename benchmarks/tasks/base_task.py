from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BenchmarkExample:
    id: str
    question: str
    context: str  # The text corpus to reason over
    gold_answer: str
    reasoning_steps: list[str] | None = None


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""

    def __init__(self, dataset_name: str, split: str = "validation", max_samples: int = None):
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.examples: list[BenchmarkExample] = []

    @abstractmethod
    def load(self):
        """Load the dataset and populate self.examples."""
        pass

    def get_examples(self) -> list[BenchmarkExample]:
        """Return the loaded examples."""
        if not self.examples:
            self.load()
        if self.max_samples:
            return self.examples[: self.max_samples]
        return self.examples
