# Available Benchmark Datasets

This document lists all datasets available for benchmarking RLM models.

## Quick Start

```bash
# HotpotQA (multi-hop reasoning)
python run_benchmark_custom.py  # Uses HotpotQA by default

# DROP (numerical reasoning)
python run_benchmark_drop.py

# Custom dataset
python -c "
from benchmarks.runner import BenchmarkRunner
from benchmarks.config import BenchmarkConfig

config = BenchmarkConfig(max_samples=10)
runner = BenchmarkRunner(config)
runner.load_tasks(['boolq'])  # or 'squad_v2', 'drop', etc.
runner.run()
"
```

## Available Datasets

### 1. HotpotQA (Distractor) ✅ Tested

**Type:** Multi-hop question answering  
**Size:** 7,405 validation examples  
**Difficulty:** Medium-Hard  
**Best for:** Testing multi-hop reasoning and distractor filtering

**Characteristics:**
- Requires connecting facts from 2+ paragraphs
- Includes irrelevant "distractor" paragraphs as noise
- Questions require bridging entities across documents

**Example:**
```
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Answer: yes
Context: 10 paragraphs (2 relevant, 8 distractors)
```

**Usage:**
```python
runner.load_tasks(["hotpotqa"])
```

---

### 2. DROP (Discrete Reasoning Over Paragraphs) ✅ Tested

**Type:** Numerical reasoning over text  
**Size:** 9,535 validation examples  
**Difficulty:** Hard  
**Best for:** Testing arithmetic, counting, sorting, comparison

**Characteristics:**
- Requires counting entities in text
- Arithmetic operations (addition, subtraction, differences)
- Temporal reasoning (years apart, before/after)
- Sorting and comparison

**Example:**
```
Question: How many field goals did Kris Brown kick?
Answer: 2
Context: Game summary with multiple scoring plays
```

**Usage:**
```python
runner.load_tasks(["drop"])
```

**RLM Advantage:** Code execution for precise counting and arithmetic!

---

### 3. BoolQ (Boolean Questions) ✅ Available

**Type:** Yes/No questions from natural queries  
**Size:** 3,270 validation examples  
**Difficulty:** Easy-Medium  
**Best for:** Testing basic reading comprehension

**Characteristics:**
- Binary classification (yes/no)
- Questions from real Google queries
- Paired with Wikipedia passages
- Tests simple inference

**Example:**
```
Question: does ethanol take more energy make that produces
Answer: no
Context: Paragraph about ethanol production energy balance
```

**Usage:**
```python
runner.load_tasks(["boolq"])
```

---

### 4. SQuAD v2 (Reading Comprehension) ✅ Available

**Type:** Extractive question answering  
**Size:** 11,873 validation examples (5,928 answerable)  
**Difficulty:** Medium  
**Best for:** Testing span extraction and unanswerable detection

**Characteristics:**
- ~50% unanswerable questions
- Answer is always a span from the passage (if answerable)
- Tests ability to recognize when answer is not present
- Single paragraph context

**Example:**
```
Question: In what country is Normandy located?
Answer: France
Context: Paragraph about the Normans
```

**Usage:**
```python
# Answerable only (default)
runner.load_tasks(["squad_v2"])

# Include unanswerable questions
from benchmarks.tasks.squad_v2 import SQuADv2Task
task = SQuADv2Task(max_samples=10, answerable_only=False)
```

---

### 5. Musique ❌ Not Available

**Type:** Complex multi-hop reasoning (3-4 hops)  
**Status:** Dataset URL broken, needs fix  
**Difficulty:** Very Hard

**Note:** Implementation exists but download URL is 404. Needs to be updated with correct source.

---

## Dataset Comparison

| Dataset | Type | Difficulty | Size | Multi-hop | Numerical | Best RLM Use Case |
|---------|------|------------|------|-----------|-----------|-------------------|
| **HotpotQA** | QA | Medium-Hard | 7,405 | ✓ | ✗ | Multi-hop reasoning |
| **DROP** | QA | Hard | 9,535 | ✗ | ✓ | Counting & arithmetic |
| **BoolQ** | Binary | Easy-Medium | 3,270 | ✗ | ✗ | Basic comprehension |
| **SQuAD v2** | Extractive | Medium | 5,928 | ✗ | ✗ | Span extraction |

## Benchmark Results Summary

### HotpotQA (10 examples)
- **Winner:** GPT-4o-mini (RLM) - F1: 0.569 (2.1x better than GPT-5.1)
- **Key insight:** RLM excels at multi-hop reasoning

### DROP (10 examples)
- **Winner:** MiniMax-M2.1 (RLM) - F1: 0.382 (1.6x better than GPT-5.1)
- **Key insight:** Code execution helps with counting/arithmetic

## Adding New Datasets

To add a new dataset:

1. Create `benchmarks/tasks/your_dataset.py`:
```python
from datasets import load_dataset
from .base_task import BaseTask, BenchmarkExample

class YourDatasetTask(BaseTask):
    def __init__(self, split: str = "validation", max_samples: int = None):
        super().__init__("your_dataset", split, max_samples)
        
    def load(self):
        dataset = load_dataset("your_dataset", split=self.split)
        for item in dataset:
            self.examples.append(BenchmarkExample(
                id=item['id'],
                question=item['question'],
                context=item['context'],
                gold_answer=item['answer'],
                reasoning_steps=None
            ))
```

2. Register in `benchmarks/runner.py`:
```python
elif name == "your_dataset":
    from .tasks.your_dataset import YourDatasetTask
    self.tasks.append(YourDatasetTask(max_samples=self.config.max_samples))
```

3. Run benchmark:
```bash
python -c "
from benchmarks.runner import BenchmarkRunner
from benchmarks.config import BenchmarkConfig

config = BenchmarkConfig(max_samples=10)
runner = BenchmarkRunner(config)
runner.load_tasks(['your_dataset'])
runner.run()
"
```

## Popular Datasets to Consider Adding

- **StrategyQA** - Implicit multi-hop reasoning
- **2WikiMultihopQA** - Wikipedia multi-hop QA
- **CommonsenseQA** - Multiple choice common sense
- **PIQA** - Physical commonsense reasoning
- **HellaSwag** - Commonsense inference

See `benchmarks/tasks/` for implementation examples.

