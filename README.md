````md
*This project has been created as part of the 42 curriculum by elbarry.*

# call_me_maybe – Function Calling with LLMs 🤖📞

Welcome to **call_me_maybe**! 🚀  
This project explores how **Large Language Models (LLMs)** can transform natural language into **structured function calls** using **constrained decoding** and schema-aware generation.

The goal is not to generate plain text answers, but to produce **100% valid JSON outputs** that can be executed reliably by machines.

It is a core AI-focused project of the **42 curriculum**, centered around:

* Function calling systems
* Token-level constrained decoding
* Structured JSON generation
* LLM interaction pipelines
* Reliability and validation

---

## 📝 Description

The program receives:

* A list of natural language prompts
* A JSON file containing available function definitions

It must determine:

* Which function should be called
* Which arguments are required
* The correct argument types

The output is a **strictly valid JSON file** containing structured function calls.

### 📌 Example

#### Input prompt

```text
What is the sum of 40 and 2?
````

#### Generated output

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

Unlike traditional chatbots, the system does not answer the question directly — it generates the proper machine-readable instruction.

---

## 🎯 Project Objectives

This project focuses on:

* Understanding how LLMs generate tokens
* Building a constrained decoding pipeline
* Enforcing JSON schema validity
* Improving reliability with small models
* Designing robust AI-driven systems
* Handling malformed inputs gracefully

---

## ⚙️ Technologies Used

| Technology   | Purpose                |
| ------------ | ---------------------- |
| Python 3.10+ | Core implementation    |
| Qwen3-0.6B   | Language model         |
| pydantic     | Validation and schemas |
| numpy        | Numerical utilities    |
| uv           | Dependency management  |
| flake8       | Code style             |
| mypy         | Static typing          |

---

## 🧠 Constrained Decoding

The core challenge of the project is ensuring that the LLM generates:

✔ Valid JSON
✔ Correct schema structure
✔ Proper argument types
✔ Allowed function names only

Instead of trusting the model blindly, the decoder filters invalid tokens during generation.

### 🔄 Generation Pipeline

```text
Prompt
   ↓
Tokenization
   ↓
Input IDs
   ↓
LLM logits
   ↓
Token filtering
   ↓
Valid next token selection
   ↓
Structured JSON output
```

At every step:

* Invalid tokens are rejected
* JSON structure is enforced
* Schema constraints are validated
* Only valid continuations remain possible

This guarantees **100% parseable outputs**.

---

## 📂 Project Structure

```text
.
├── src/
├── llm_sdk/
├── data/
│   ├── input/
│   └── output/
├── README.md
├── pyproject.toml
├── uv.lock
└── Makefile
```

---

## 📥 Input Files

### `function_calling_tests.json`

Contains prompts to process.

Example:

```json
[
  {
    "prompt": "Reverse the string 'hello'"
  }
]
```

### `functions_definition.json`

Contains available functions and schemas.

Example:

```json
[
  {
    "name": "fn_reverse_string",
    "parameters": {
      "s": {
        "type": "string"
      }
    }
  }
]
```

---

## 📤 Output Format

The program generates:

```text
data/output/function_calling_results.json
```

Example:

```json
[
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```

---

## ⚙️ Installation & Usage

### 📦 Install dependencies

```bash
uv sync
```

### ▶ Run the program

```bash
uv run python -m src
```

### ▶ Custom paths

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

---

## 🧪 Testing Strategy

The implementation was tested against:

* Empty prompts
* Invalid JSON files
* Missing files
* Large numbers
* Special characters
* Incorrect argument types
* Ambiguous prompts
* Multiple parameter functions

Validation included:

* JSON parsing verification
* Schema compliance checks
* Type validation
* Reliability testing

---

## 📊 Performance Analysis

The system aims to achieve:

| Metric                      | Goal            |
| --------------------------- | --------------- |
| JSON validity               | 100%            |
| Function selection accuracy | 90%+            |
| Robustness                  | High            |
| Processing speed            | Under 5 minutes |

The project demonstrates that **small LLMs can become highly reliable when guided structurally rather than purely through prompting**.

---

## 🧩 Design Decisions

### Why constrained decoding?

Prompting alone is unreliable for structured generation.

Constrained decoding ensures:

* Structural correctness
* Deterministic outputs
* Better reliability
* Safer execution pipelines

### Why pydantic?

Used for:

* Runtime validation
* Type enforcement
* Cleaner schema management
* Safer data handling

### Why Qwen3-0.6B?

The project specifically targets lightweight models to demonstrate that:

> Proper decoding strategies matter more than raw model size.

---

## ⚠️ Error Handling

The program gracefully handles:

* Missing input files
* Invalid JSON
* Invalid schemas
* Unsupported parameter types
* Generation failures
* Malformed prompts

The application should never crash unexpectedly.

---

## 📚 Resources

* JSON Schema
* Constrained Decoding
* Tokenization
* Function Calling in LLMs
* Pydantic Documentation
* Qwen Documentation
* [https://docs.pydantic.dev/](https://docs.pydantic.dev/)
* [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
* [https://en.wikipedia.org/wiki/JSON](https://en.wikipedia.org/wiki/JSON)
* [https://en.wikipedia.org/wiki/Tokenizer_(lexical_analysis)](https://en.wikipedia.org/wiki/Tokenizer_%28lexical_analysis%29)

---

## 🤖 AI Usage

AI tools were used for:

* Documentation assistance
* README structuring
* Clarifying theoretical concepts
* Reviewing explanations

All implementation details, architecture, constrained decoding logic, validation systems, and project decisions were fully understood, designed, tested, and implemented manually in compliance with the **42 AI policy**.

---

## 🚀 Bonus Features

Potential bonus implementations include:

* Multi-model support
* Custom tokenizer implementation
* Advanced recovery systems
* Caching optimizations
* Visualization of token generation
* Nested argument support

---

✨ Thanks for checking out my **call_me_maybe** project! 🚀

```
