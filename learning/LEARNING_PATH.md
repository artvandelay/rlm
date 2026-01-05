# 🎓 RLM Learning Path

A progressive guide to understanding Recursive Language Models from first principles.

## Prerequisites

```bash
source ~/pyenv/rlm/bin/activate
cd /Users/jigar/projects/messing-around/rlm
```

Make sure your `.env` file has `OPENROUTER_API_KEY` set.

---

## 📚 The Learning Journey

### **Phase 1: Basic Code Execution** (5 min)
**File:** `learn_01_basic.py`

**What you'll learn:**
- RLM can write and execute Python code
- See the difference from regular text generation
- Understand the basic REPL loop

**Run it:**
```bash
python learn_01_basic.py
```

**Key insight:** Watch how the model writes code to solve the problem instead of trying to calculate in its "head".

---

### **Phase 2: Iterative Reasoning** (10 min)
**File:** `learn_02_iterative.py`

**What you'll learn:**
- RLM examines context step-by-step
- Multiple iterations of code execution
- How context is passed to the REPL
- Logging trajectories for later analysis

**Run it:**
```bash
python learn_02_iterative.py
```

**Key insight:** The model can inspect data, make decisions, write more code, and iterate - not just one-shot generation.

---

### **Phase 3: Recursive Sub-Calls** (15 min)
**File:** `learn_03_recursive.py`

**What you'll learn:**
- Using `llm_query()` inside code
- Breaking down problems into sub-problems
- Parallel/batched LLM calls
- True recursive reasoning

**Run it:**
```bash
python learn_03_recursive.py
```

**Key insight:** This is where "Recursive" in RLM comes from - the model can call itself during execution!

---

### **Phase 4: RLM vs Regular LLM** (10 min)
**File:** `learn_04_comparison.py`

**What you'll learn:**
- Direct comparison with regular LLM API
- Where RLM excels (precision, verification)
- When code execution matters

**Run it:**
```bash
python learn_04_comparison.py
```

**Key insight:** See the accuracy difference when tasks require precise computation.

---

### **Phase 5: Visualize Trajectories** (15 min)
**File:** `learn_05_visualize.py`

**What you'll learn:**
- How to log trajectories
- Complex multi-step reasoning
- Understanding the full execution flow

**Run it:**
```bash
python learn_05_visualize.py
```

**Then visualize:**
```bash
# In a new terminal
cd visualizer/
npm install  # first time only
npm run dev  # opens at localhost:3001
```

Upload the `.jsonl` file from `./logs/` to see:
- Iteration timeline
- Code execution results
- Token usage
- Full reasoning chain

**Key insight:** Visual understanding of how RLM thinks and iterates.

---

## 🎯 What You'll Understand After This

1. **Core Concept:** RLM = LLM + REPL environment
2. **Iteration:** Model generates code → executes → sees results → repeats
3. **Recursion:** Code can call `llm_query()` for sub-problems
4. **Context:** Your input becomes the `context` variable
5. **Completion:** Model signals done with `FINAL(answer)`

---

## 🚀 Next Steps

After completing all phases:

1. **Experiment with your own tasks**
   - Long document analysis
   - Multi-step reasoning
   - Data processing pipelines

2. **Try different environments**
   - Docker (isolated containers)
   - Modal (cloud sandboxes)

3. **Explore the codebase**
   - `rlm/core/rlm.py` - Main RLM class
   - `rlm/environments/` - Different execution environments
   - `rlm/clients/` - LLM client integrations

4. **Read the paper**
   - [arXiv: Recursive Language Models](https://arxiv.org/abs/2512.24601)
   - [Blog post](https://alexzhang13.github.io/blog/2025/rlm/)

---

## 💡 Pro Tips

- **Start with `verbose=True`** - See what's happening
- **Enable logging** - Analyze trajectories later
- **Increase `max_iterations`** for complex tasks
- **Use `root_prompt`** for Q&A over long contexts
- **Experiment with `llm_query()`** - This is the power feature!

---

## 🐛 Troubleshooting

**"Operation not permitted" errors:**
- Some commands need `required_permissions: ['all']`
- This is normal for accessing `.env` files

**RLM runs but gives wrong answers:**
- Check `verbose=True` output to see what code it's running
- Increase `max_iterations` if it's running out of steps
- Try a stronger model (gpt-4o instead of gpt-4o-mini)

**Modal/Docker not working:**
- Modal: Run `modal setup` first
- Docker: Make sure Docker Desktop is running

---

## 📖 Additional Resources

- [Getting Started Guide](docs/getting-started.md)
- [API Reference](docs/api/rlm.md)
- [AGENTS.md](AGENTS.md) - For contributing
- [Examples](examples/) - More example scripts

Happy learning! 🎉

