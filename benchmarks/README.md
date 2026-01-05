# RLM Benchmarking Suite

A comprehensive benchmarking framework for comparing Recursive Language Models (RLMs) against standard LLM inference on multi-hop reasoning tasks.

## Overview

This suite demonstrates that **smaller models using RLM architecture can outperform larger models using standard inference** on complex reasoning tasks by:
- Breaking down problems into sub-steps
- Using code execution for precise information extraction
- Making recursive sub-LM calls to decompose complex queries

## Quick Start

### Run Quick Test (2 models, 5 examples)
```bash
python run_benchmark_quick.py
```

### Run Full Benchmark (5 models, 10 examples)
```bash
python run_benchmark.py
```

### View Results
```bash
python benchmarks/viewer.py --file benchmarks/results/hotpot_qa_results_YYYYMMDD_HHMMSS.jsonl
```

This generates:
- Console summary with metrics table
- Detailed markdown report: `benchmarks/results/report_YYYYMMDD_HHMMSS.md`

## Results Format

### Timestamped Files
All results use matching run IDs (timestamp-based):
- **Raw data**: `hotpot_qa_results_20260105_014257.jsonl`
- **Report**: `report_20260105_014257.md`

### Tracked Metrics
- **Exact Match (EM)**: Binary correctness
- **F1 Score**: Token overlap with gold answer
- **LLM Calls**: Number of sub-calls (1 for regular, 3-10 for RLM)
- **Time**: Execution time per example

## Example Results

From our quick test (5 examples, gpt-4o vs gpt-4o-mini):

| Model | F1 Score | Avg Calls | Win Rate |
|-------|----------|-----------|----------|
| **GPT-4o (Regular)** | 0.178 | 1.0 | 0% |
| **GPT-4o-mini (RLM)** | **0.586** | 4.8 | **100%** |

**Key Finding**: Smaller model (gpt-4o-mini) with RLM beat larger model (gpt-4o) by **3.3x on F1 score** using ~5 LLM calls.

## Configuration

### Default Models (`benchmarks/config.py`)
1. **GPT-4o (Regular)** - Baseline large model
2. **GPT-4o-mini (RLM)** - Strong small model
3. **Qwen-2.5-7B (RLM)** - 7B parameter model
4. **Llama-3.2-3B (RLM)** - 3B parameter model
5. **Gemma-2-2B (RLM)** - Tiny 2B parameter model

### Custom Configuration
```python
from benchmarks.config import BenchmarkConfig, ModelConfig

config = BenchmarkConfig(max_samples=10)
config.models = [
    ModelConfig(
        name="My Model (RLM)",
        model_id="provider/model-id",
        backend="openrouter",
        use_rlm=True
    ),
    # ... more models
]
```

## Supported Datasets

### HotpotQA (Distractor)
- **Type**: Multi-hop question answering
- **Difficulty**: Requires reasoning across multiple paragraphs
- **Distraction**: Includes irrelevant paragraphs as noise
- **Size**: 7,405 validation examples

### Musique (Work in Progress)
- **Type**: Complex multi-hop reasoning (3-4 hops)
- **Difficulty**: Harder than HotpotQA
- **Status**: Dataset loading in progress

## Architecture

```
benchmarks/
├── __init__.py
├── config.py              # Model configurations
├── runner.py              # Benchmark orchestrator
├── viewer.py              # Results viewer + report generator
├── tasks/
│   ├── base_task.py       # Abstract task interface
│   ├── hotpotqa.py        # HotpotQA dataset adapter
│   └── musique.py         # Musique dataset adapter
├── evaluators/
│   └── metrics.py         # F1 and EM metrics
└── results/               # Generated results and reports
```

## Key Improvements Made

### 1. Enhanced RLM Prompt
Updated `rlm/utils/prompts.py` with explicit format instructions:
- Clear examples: `FINAL(yes)`, `FINAL(Animorphs)`
- Warnings against tuple returns: `('FINAL', 'answer')` ❌
- Emphasis on concise answers for better metric alignment

### 2. Multi-Model Support
- Compare 1 large regular LLM vs N RLM models
- Track LLM call counts per model
- Head-to-head win/loss tracking

### 3. Comprehensive Reporting
- Markdown reports with tables
- Sample comparisons with winners highlighted
- Key insights (best F1, fastest, most efficient)

## Why RLM Wins

1. **Code Execution**: Precise information extraction from context
2. **Recursive Decomposition**: Break complex queries into sub-problems
3. **Concise Outputs**: Format aligns better with evaluation metrics
4. **Iterative Reasoning**: Can verify and refine answers

## Requirements

- Python 3.11+
- OpenRouter API key (or OpenAI API key)
- `datasets`, `tqdm`, `openai` packages (auto-installed via uv)

## Environment Setup

```bash
# Ensure .env file exists with:
OPENROUTER_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
```

## Future Work

- [ ] Add Musique dataset support (URL fix needed)
- [ ] Expand to more datasets (StrategyQA, 2WikiMultihopQA)
- [ ] Add statistical significance testing
- [ ] Cost analysis (tokens × price per model)
- [ ] Visualization dashboard for trajectories

## Citation

If you use this benchmarking suite, please cite the original RLM paper:

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models}, 
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601}, 
}
```

