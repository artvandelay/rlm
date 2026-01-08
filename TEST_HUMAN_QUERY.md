# Testing human_query() Function

## Test Queries That Should Trigger human_query()

These queries are designed to be ambiguous or require user preferences, which should prompt the LLM to ask for clarification:

### 1. Ambiguous Timeframe
```
"Summarize the key points about anemia prevention"
```
**Expected behavior:** LLM should ask: "Do you want a summary for pregnant women, children, or general population?"

### 2. Preference-Based
```
"Create a treatment plan"
```
**Expected behavior:** LLM should ask: "What type of treatment plan? For anemia? What severity level?"

### 3. Multiple Options
```
"What should I do about low HB levels?"
```
**Expected behavior:** LLM should ask: "Are you asking for yourself, a patient, or general guidelines? What's the current HB level?"

### 4. Vague Request
```
"Tell me what's important"
```
**Expected behavior:** LLM should ask: "What topic are you interested in? From the context, I can help with HB levels, anemia, treatment guidelines, etc."

### 5. Comparison Request
```
"Which is better?"
```
**Expected behavior:** LLM should ask: "Which options are you comparing? Please specify."

## How to Test

1. Run: `RLM_VERBOSE=true python cli_chat.py`
2. Load context: `/Users/jigar/projects/messing-around/rlm/context/asha`
3. Use one of the test queries above
4. Watch for `human_query()` calls in verbose output
5. The CLI should prompt you for input when the LLM calls `human_query()`

## Expected CLI Output

When `human_query()` is called, you should see:
```
🤔 RLM Question: [The question from the LLM]
Options: [option1, option2, ...] (if provided)
Your response: [waiting for input]
```

