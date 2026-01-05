# 🚀 RLM Quick Start

## TL;DR - Get Started in 30 Seconds

```bash
# Activate environment
source ~/pyenv/rlm/bin/activate

# Run the interactive learning path
./run_learning_path.sh

# OR run individual phases
python learn_01_basic.py
```

---

## What is RLM?

**Regular LLM:** Input → Text Output  
**RLM:** Input → Code + Execution + Iteration → Precise Output

RLM = **R**ecursive **L**anguage **M**odel
- Writes Python code to solve problems
- Executes code and sees results
- Can call itself recursively via `llm_query()`
- Iterates until it finds the answer

---

## The 5-Phase Learning Path

| Phase | Time | What You Learn | File |
|-------|------|----------------|------|
| 1 | 5 min | Basic code execution | `learn_01_basic.py` |
| 2 | 10 min | Iterative reasoning | `learn_02_iterative.py` |
| 3 | 15 min | Recursive sub-calls | `learn_03_recursive.py` |
| 4 | 10 min | RLM vs Regular LLM | `learn_04_comparison.py` |
| 5 | 15 min | Visualization | `learn_05_visualize.py` |

**Total:** ~55 minutes to understand RLM deeply

---

## Key Concepts (The "Aha!" Moments)

### 1. The REPL Loop
```
User Input → RLM
  ↓
  Model generates code
  ↓
  Code executes in REPL
  ↓
  Model sees results
  ↓
  Repeat until FINAL(answer)
```

### 2. The Context Variable
Your input becomes `context` in the REPL:
```python
rlm.completion("Analyze this data: [1,2,3]")
# Inside REPL: context = "Analyze this data: [1,2,3]"
```

### 3. Recursive Calls
Code can call the LLM:
```python
# Inside REPL execution:
sentiment = llm_query("Is this review positive? 'Great product!'")
# Returns: "positive"
```

### 4. Signaling Completion
Model uses `FINAL()` to return answer:
```python
# Inside REPL:
result = calculate_something()
FINAL(result)  # This becomes the final answer
```

---

## When to Use RLM vs Regular LLM

### ✅ Use RLM for:
- Precise calculations
- Data analysis
- Multi-step reasoning
- Long context processing
- Tasks requiring verification
- Problems that benefit from code

### ❌ Use Regular LLM for:
- Simple text generation
- Creative writing
- Quick one-shot answers
- When latency matters most

---

## Quick Examples

### Example 1: Calculation
```python
from rlm import RLM

rlm = RLM(backend="openai", backend_kwargs={...}, verbose=True)
result = rlm.completion("Calculate factorial of 100")
# RLM writes code, executes it, returns exact answer
```

### Example 2: Data Analysis
```python
data = {"sales": [100, 150, 200], "costs": [80, 90, 110]}
result = rlm.completion(data, root_prompt="What's the profit margin?")
# RLM analyzes the data structure and calculates
```

### Example 3: Recursive Reasoning
```python
reviews = ["Great!", "Terrible", "Okay"]
result = rlm.completion(
    reviews,
    root_prompt="Analyze sentiment of each review using llm_query()"
)
# RLM calls llm_query() for each review, then aggregates
```

---

## Cheat Sheet

### Basic Setup
```python
import os
from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "openai/gpt-4o-mini",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
    },
    environment="local",
    max_iterations=30,
    logger=logger,
    verbose=True,
)

result = rlm.completion("Your prompt here")
print(result.response)
```

### Available Environments
- `local` - Same process (fast, less isolated)
- `docker` - Container (isolated, requires Docker)
- `modal` - Cloud sandbox (fully isolated, requires Modal)

### Useful Parameters
- `verbose=True` - See what's happening
- `max_iterations=50` - Allow more steps
- `logger=RLMLogger(...)` - Save trajectories
- `root_prompt="..."` - Question visible to model

---

## Visualize Trajectories

```bash
# After running with logger enabled
cd visualizer/
npm install  # first time only
npm run dev  # opens at localhost:3001
```

Upload `.jsonl` files from `./logs/` to see:
- Each iteration's reasoning
- Code blocks executed
- Outputs and errors
- Token usage
- Full execution timeline

---

## Next Steps

1. ✅ Run `./run_learning_path.sh`
2. ✅ Complete all 5 phases
3. ✅ Visualize a trajectory
4. 📖 Read [LEARNING_PATH.md](LEARNING_PATH.md) for details
5. 📖 Read [docs/getting-started.md](docs/getting-started.md)
6. 🧪 Experiment with your own tasks
7. 📄 Read the [paper](https://arxiv.org/abs/2512.24601)

---

## Help & Resources

- **Docs:** `docs/getting-started.md`, `docs/api/rlm.md`
- **Examples:** `examples/` directory
- **Contributing:** `AGENTS.md`, `CONTRIBUTING.md`
- **Paper:** https://arxiv.org/abs/2512.24601
- **Blog:** https://alexzhang13.github.io/blog/2025/rlm/

---

**Ready?** Run `./run_learning_path.sh` and start learning! 🎓

