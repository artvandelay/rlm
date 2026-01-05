# 👋 START HERE - Your RLM Learning Journey

## ✅ Setup Complete!

Your environment is ready to go:
- ✅ Python 3.12 virtual environment at `~/pyenv/rlm`
- ✅ All dependencies installed
- ✅ OpenRouter API key configured
- ✅ Learning materials created

---

## 🎯 Best Way to Learn RLM

### **Option 1: Interactive Learning Path** (Recommended)

Run the interactive script that guides you through all 5 phases:

```bash
source ~/pyenv/rlm/bin/activate
./run_learning_path.sh
```

This will let you:
- Choose which phase to run
- Run them one at a time or all at once
- See exactly what's happening at each step

### **Option 2: Manual Exploration**

Run each phase individually:

```bash
source ~/pyenv/rlm/bin/activate

# Phase 1: See basic code execution (5 min)
python learn_01_basic.py

# Phase 2: See iterative reasoning (10 min)
python learn_02_iterative.py

# Phase 3: See recursive sub-calls (15 min)
python learn_03_recursive.py

# Phase 4: Compare RLM vs Regular LLM (10 min)
python learn_04_comparison.py

# Phase 5: Generate complex trajectory (15 min)
python learn_05_visualize.py
```

---

## 📚 What Each Phase Teaches

| Phase | Time | Key Learning | "Aha!" Moment |
|-------|------|--------------|---------------|
| **1** | 5 min | Basic code execution | "It writes and runs Python code!" |
| **2** | 10 min | Iterative reasoning | "It can examine data step-by-step!" |
| **3** | 15 min | Recursive sub-calls | "It can call itself recursively!" |
| **4** | 10 min | RLM vs Regular LLM | "This is way more accurate!" |
| **5** | 15 min | Visualization | "Now I see the full reasoning chain!" |

**Total time:** ~55 minutes to deeply understand RLM

---

## 🎓 Learning Outcomes

After completing all phases, you'll understand:

1. ✅ **What RLM is**: LLM + REPL + Recursion
2. ✅ **How it works**: The iteration loop
3. ✅ **Why it's powerful**: Precision + verification + decomposition
4. ✅ **When to use it**: Tasks requiring code/calculation
5. ✅ **How to use it**: API, parameters, best practices

---

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Quick reference and cheat sheet
- **[LEARNING_PATH.md](LEARNING_PATH.md)** - Detailed learning guide
- **[docs/getting-started.md](docs/getting-started.md)** - Official getting started
- **[docs/api/rlm.md](docs/api/rlm.md)** - Complete API reference
- **[AGENTS.md](AGENTS.md)** - Contributing guide

---

## 🚀 Quick Start (30 seconds)

If you just want to see it work right now:

```bash
source ~/pyenv/rlm/bin/activate
python learn_01_basic.py
```

Watch as RLM:
1. Receives a math problem
2. Writes Python code to solve it
3. Executes the code
4. Returns the precise answer

---

## 🎨 Visualize Trajectories

After running Phase 5 (or any script with logging):

```bash
cd visualizer/
npm install  # first time only
npm run dev  # opens at localhost:3001
```

Upload the `.jsonl` files from `./logs/` to see:
- Full iteration timeline
- Code execution at each step
- Model's reasoning process
- Token usage breakdown

---

## 💡 Pro Tips for Learning

1. **Start with verbose mode** - Always use `verbose=True` to see what's happening
2. **Read the output** - Pay attention to each iteration
3. **Compare with regular LLM** - Phase 4 shows the difference clearly
4. **Visualize trajectories** - Phase 5 + visualizer = deep understanding
5. **Experiment** - After the learning path, try your own prompts

---

## 🎯 Recommended Learning Flow

```
START HERE.md (you are here)
    ↓
./run_learning_path.sh (interactive)
    ↓
Complete all 5 phases (~55 min)
    ↓
Visualize trajectories (visualizer)
    ↓
Read QUICK_START.md (reference)
    ↓
Experiment with your own tasks
    ↓
Read the paper (deep dive)
```

---

## 🤔 Common Questions

**Q: What makes RLM different from regular LLM?**  
A: RLM can write and execute code, iterate on results, and call itself recursively. Regular LLM just generates text.

**Q: When should I use RLM vs regular LLM?**  
A: Use RLM for tasks requiring precision, calculation, data analysis, or multi-step reasoning. Use regular LLM for simple text generation.

**Q: Is it slower than regular LLM?**  
A: Yes, because it iterates. But it's much more accurate for complex tasks.

**Q: Can I use my own LLM provider?**  
A: Yes! RLM supports OpenAI, Anthropic, OpenRouter, Portkey, LiteLLM, and local vLLM.

**Q: What's the "recursive" part?**  
A: Code running in the REPL can call `llm_query()` to make sub-LM calls. See Phase 3!

---

## 🎉 Ready to Start?

```bash
source ~/pyenv/rlm/bin/activate
./run_learning_path.sh
```

**Or jump straight to Phase 1:**

```bash
source ~/pyenv/rlm/bin/activate
python learn_01_basic.py
```

---

## 📞 Need Help?

- Check the docs in `docs/`
- Read `QUICK_START.md` for quick reference
- Look at examples in `examples/`
- Read the paper: https://arxiv.org/abs/2512.24601

---

**Let's go! Start your RLM learning journey now.** 🚀

