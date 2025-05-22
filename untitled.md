

## 🧠 **Types of LLM Agents: Autonomy Spectrum**

| **Level** | **Agent Type**       | **Capabilities**                                                                 | **Limitations**                                                            | **Examples**                            |
|----------|----------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------|
| 🔹 1     | **Generator Agent**   | Generates responses based on prompt and retrieved context (e.g., RAG).          | ❌ No memory<br>❌ No planning<br>❌ No external action                     | Basic chatbot, RAG Q&A                  |
| 🔸 2     | **Tool-Calling Agent**| Can call external tools or APIs based on context or prompt analysis.            | ❌ Still reactive<br>❌ No self-correction or persistence                   | Copilot, search-augmented chatbots      |
| 🟡 3     | **Planning Agent**    | Plans multi-step workflows, checks outcomes before moving forward.              | ❌ Task-bound<br>❌ Doesn’t self-initiate or persist across workflows      | AI debuggers, automation sequences       |
| 🟢 4     | **Autonomous Agent**  | Self-initiates, adapts, persists across sessions, uses memory and feedback.     | ⚠️ Limited innovation and long-term adaptability                          | Advanced personal assistants (future)    |



---

## 🔍 **LLM Agent Evaluation Dimensions**

### 1️⃣ **Tool Correctness**
Assesses **whether the right tools were called**, and **how correctly they were used**.

**Key Dimensions:**
- **Tool Selection**: Did the agent select the correct tools?
- **Input Parameters**: Were parameters passed into the tools accurate?
- **Output Accuracy**: Was the tool’s output correct relative to expected result?

📌 *Partial correctness allowed:*  
- Order may not matter → use set comparison.  
- Some params may be incorrect → % of correct params.

🧪 **Code Example**
```python
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCallParams, ToolCall

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund.",
    tools_called=[ToolCall(name="WebSearchTool"), ToolCall(name="QueryTool")],
    expected_tools=[ToolCall(name="WebSearchTool")]
)

metric = ToolCorrectnessMetric(
    evaluation_params=[ToolCallParams.TOOL, ToolCallParams.INPUT_PARAMETERS],
    should_consider_ordering=False
)
metric.measure(test_case)
print(metric.score, metric.reason)
```

---

### 2️⃣ **Tool Efficiency**
Evaluates **how optimal the tool usage path was**, given available tools.

**Key Metrics:**
- ✅ **Redundant Tool Usage** – Unnecessary tools used?
- ✅ **Tool Frequency** – Tools called more times than needed?
- ✅ **Goal-oriented Efficiency** – Was the tool trajectory optimal for the user goal?

🧪 **Code Example**
```python
from deepeval.metrics import ToolEffiencyMetric
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund.",
    tools_called=[ToolCall(name="WebSearchTool")],
)

metric = ToolEffiencyMetric(
    available_tools=[ToolCall(name="WebSearchTool"), ToolCall(name="QueryTool")]
)
metric.measure(test_case)
print(metric.score, metric.reason)
```

> 🧠 *LLM-as-Judge* approach can be used for more complex multi-step tasks to evaluate trajectory relevance and efficiency.

---

## 📊 Visual: LLM Agent Maturity vs Evaluation Complexity

```
Autonomy Level
↑
│   🟢 Autonomous Agent       ➤ Needs persistent memory + environment monitoring + goal tracking.
│   🟡 Planning Agent         ➤ Needs step-wise evaluation (plan quality, success check, retry logic).
│   🔸 Tool-Calling Agent     ➤ Needs evaluation of tool correctness and efficiency.
│   🔹 Generator Agent        ➤ Needs traditional metrics (relevance, factuality, coherence).
└───────────────────────────────→ Evaluation Complexity
```

---

## ✍️ 2-Line Takeaway (Interview Ready)

> **Tool-Correctness** checks if the agent selected and used tools appropriately, while **Tool-Efficiency** examines how optimally those tools were used to reach the goal—both are essential to robust agent evaluation.

Here’s a **comprehensive visual and structured guide** to **Evaluating Agentic Workflows** using the **Task Completion** metric—covering **why it matters, how it works, and how to implement it using DeepEval**.

---

## 🧩 Why Evaluate Agentic Workflows?

While **Tool Correctness** and **Tool Efficiency** tell us **how** the agent used tools, **Agentic Workflow Evaluation** answers a higher-order question:

> ✅ **Did the agent actually complete the task successfully?**

Agentic workflows include:
1. Understanding the task from the user’s prompt.
2. Planning a reasoning chain (if applicable).
3. Calling tools or taking steps.
4. Producing a final result that solves the user's need.

---

## 🎯 **Metric: Task Completion (a.k.a. Goal Accuracy)**

This metric evaluates **end-to-end success**—from user input to final output—**with reasoning context and tool usage included**.

### ✅ Applicable Scenarios
| Task Type                    | Task Completion Definition                                   |
|-----------------------------|--------------------------------------------------------------|
| Web Shopping Agent          | Did the agent buy the correct product (based on attributes)?|
| Travel Planner Agent        | Did it create a valid itinerary matching input criteria?     |
| Debugging Agent             | Did it fix the bug and verify the fix?                      |
| General Chat-based Agents   | Was the response appropriate and actionable?                |

---

## ⚙️ How Task Completion Works in **DeepEval**

### 1. **LLM-Derived Task Understanding**  
   Infers what the task is from the input string.

### 2. **Tool-Use + Reasoning Assessment**  
   Evaluates tools used, steps taken, and checks final outcome.

### 3. **No Ground Truth Needed**  
   Uses a language model (e.g., GPT-4) to **judge correctness based on reasoning**, even when predefined outputs are unavailable.

---

## 🧪 **Code Example**

```python
from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall

# Define the metric
metric = TaskCompletionMetric(
    threshold=0.7,   # Minimum acceptable success score
    model="gpt-4",   # Model used to reason and judge completion
    include_reason=True
)

# Define the agent behavior test
test_case = LLMTestCase(
    input="Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine.",
    actual_output=(
        "Day 1: Eiffel Tower, dinner at Le Jules Verne. "
        "Day 2: Louvre Museum, lunch at Angelina Paris. "
        "Day 3: Montmartre, evening at a wine bar."
    ),
    tools_called=[
        ToolCall(
            name="Itinerary Generator",
            description="Creates travel plans.",
            input_parameters={"destination": "Paris", "days": 3},
            output=[
                "Day 1: Eiffel Tower, Le Jules Verne.",
                "Day 2: Louvre Museum, Angelina Paris.",
                "Day 3: Montmartre, wine bar."
            ],
        ),
        ToolCall(
            name="Restaurant Finder",
            description="Finds top restaurants in a city.",
            input_parameters={"city": "Paris"},
            output=["Le Jules Verne", "Angelina Paris", "local wine bars"],
        ),
    ],
)

# Evaluate
metric.measure(test_case)
print(metric.score)
print(metric.reason)
```

---

## 📊 Visual: Evaluation Scope Comparison

| **Metric**         | **Focus**                       | **Question It Answers**                             |
|--------------------|----------------------------------|------------------------------------------------------|
| Tool Correctness   | Tool selection, input/output     | Did the agent use the right tools correctly?         |
| Tool Efficiency    | Optimal tool usage               | Was the agent efficient with tools it used?          |
| **Task Completion**| Full task + reasoning + output   | **Did the agent actually solve the user’s problem?** |

---

## 📌 Summary (1-Liner for Interview)

> **Task Completion** evaluates whether an agent fulfilled the user’s intent end-to-end by analyzing reasoning, tool usage, and final output—without requiring ground truth.

---

## ✍️ 2-Liner Takeaway

> Unlike fixed tool-use metrics, Task Completion provides flexible and scalable evaluation for diverse agent behaviors by judging end-to-end goal success using LLMs.  
> It's ideal when explicit outputs aren't predefined, as in open-ended or real-world agentic tasks.

Here's a **comprehensive and visual walkthrough** for evaluating **Agentic Workflows**, including the **Task Completion metric** and **G-Eval for custom evaluation criteria**. These tools help go *beyond tool usage* and evaluate the **agent’s full reasoning + task-solving capacity**, even in open-ended or real-world contexts.

---

## 🧠 Why Evaluate Agentic Workflows?

While tool correctness/efficiency tell you *how* tools were used, **Agentic Workflow Evaluation** answers the ultimate question:

> ✅ **Did the agent complete the task, and how well did it follow the workflow logic to do so?**

This involves:
- Understanding the **task intent**
- Performing reasoning + planning (optional)
- Using tools effectively
- Delivering a **useful final answer**

---

## 🎯 Metric 1: **Task Completion**

| 📌 **Goal**          | Determine if the LLM agent solved the user’s task successfully. |
|----------------------|---------------------------------------------------------------|
| 🧰 **Components**    | Input intent → Tool usage → Final output → Task success       |
| 📈 **Scoring**       | Done using LLM-based judgment (e.g., GPT-4)                   |
| ⚖️ **No Ground Truth** | Suitable for open-world tasks with infinite permutations    |

### ✅ **Best for**:
- Travel planners
- Multi-hop research agents
- Agents where outputs are *subjectively useful* rather than *precisely defined*

---

### 🧪 Code Example: Task Completion (DeepEval)

```python
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall

# Define metric
metric = TaskCompletionMetric(
    threshold=0.7,
    model="gpt-4",
    include_reason=True
)

# Define test case
test_case = LLMTestCase(
    input="Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine.",
    actual_output=(
        "Day 1: Eiffel Tower, dinner at Le Jules Verne. "
        "Day 2: Louvre Museum, lunch at Angelina Paris. "
        "Day 3: Montmartre, evening at a wine bar."
    ),
    tools_called=[
        ToolCall(
            name="Itinerary Generator",
            description="Creates travel plans.",
            input_parameters={"destination": "Paris", "days": 3},
            output=[ ... ]
        ),
        ToolCall(
            name="Restaurant Finder",
            description="Finds top restaurants.",
            input_parameters={"city": "Paris"},
            output=[ ... ]
        )
    ]
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
```

> ✅ **Summary (1-liner)**:  
> Task Completion evaluates whether an agent *fulfills the user's goal* end-to-end, not just whether tools were used properly.

---

## 🔧 Metric 2: **G-Eval (Custom CoT Evaluation)**

| 🔍 **Purpose**                | Define **custom evaluation criteria** in natural language |
|------------------------------|------------------------------------------------------------|
| 📚 **Uses LLM + CoT**         | Reason through *why* the output meets or fails the criteria |
| 🧠 **Flexible Inputs**        | Can check output tone, explanation completeness, tool usage, etc. |
| 🧪 **Best for**               | UX feedback, policy alignment, transparency, tone, accuracy |

### 📌 Example Use Case: **Transparency in Tool Use**
> Does the agent explain what tools it used to reach the final answer?

### 🧪 Code: G-Eval for Transparency

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

transparency_metric = GEval(
    name="Transparency",
    criteria="Determine whether the tool invocation information is captured in the actual output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.TOOLS_CALLED]
)
```

> 🔍 Example Evaluation Prompt (CoT style):
> “The agent said ‘The restaurant is fully booked,’ but did not mention checking other dates or options. Was that transparent about the steps it took?”

---

## 📊 Visual Summary: Full Agent Evaluation Spectrum

```
User Input ──► Reasoning ──► Tool Calls ──► Output
                     ▲               ▲            ▲
                     │               │            │
          Reasoning Eval     Tool Correctness     Output Quality
                                               │
                                               ▼
                                    ✅ Task Completion Metric
                                     ✅ G-Eval (custom logic)
```

---

## ✍️ 2-Liner Takeaway

> **Task Completion** captures whether an agent fulfilled the user’s goal from start to finish, while **G-Eval** allows for nuanced evaluations using chain-of-thought-based custom logic.  
> These tools help scale evaluation of real-world, flexible workflows where rigid correctness alone isn’t enough.

Would you like a **combined evaluation framework template** showing how to plug in tool, reasoning, and task evaluations for complex multi-agent systems?