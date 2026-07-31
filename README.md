*This project has been created as part of the 42 curriculum by elbarry.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling project developed as part of the 42 curriculum.

The goal is to use a Large Language Model (LLM) to determine which function should be called from a natural language request while producing a valid JSON output containing the selected function and its parameters.

Unlike traditional text generation, this project implements constrained decoding to guarantee that the generated JSON always follows the expected structure.

The program:

- loads function definitions from a JSON file;
- loads user prompts from another JSON file;
- selects the most appropriate function using constrained decoding;
- extracts function parameters;
- generates valid JSON results.

---

# Instructions

## Requirements

- Python 3.12+
- uv
- Internet connection during the first installation (to download model dependencies)

## Installation

```bash
make install
```

or

```bash
uv sync
```

## Run

```bash
make run
```

or

```bash
uv run python -m src
```

The generated output is written to:

```
data/output/function_calling_results.json
```

## Debug

```bash
make debug
```

## Lint

```bash
make lint
```

---

# Algorithm Explanation

The project implements constrained decoding instead of unrestricted text generation.

Rather than allowing the language model to generate arbitrary tokens, every decoding step restricts the possible tokens according to the expected JSON grammar.

The decoder follows a state machine representing the JSON structure:

```
{
    "name": "...",
    "parameters": {
        ...
    }
}
```

The decoding process is divided into several stages:

1. Generate the opening brace.
2. Generate the `"name"` key.
3. Generate the selected function name.
4. Generate the `"parameters"` key.
5. Extract parameters from the prompt.
6. Return the final JSON object.

Function selection is entirely driven by constrained decoding using the model logits.

Parameter extraction is performed using deterministic heuristics adapted to parameter types:

- numbers
- integers
- booleans
- strings
- paths
- encodings
- database names
- templates
- names

This hybrid approach guarantees valid JSON while improving parameter extraction accuracy.

---

# Design Decisions

Several design choices were made during development.

### State machine

A finite-state machine simplifies constrained decoding and guarantees the JSON structure.

### Hybrid architecture

The LLM is responsible for selecting the most appropriate function.

Parameter extraction is implemented separately using Python heuristics, which improves determinism and simplifies validation.

### Robust parser

Input files are validated before execution.

Missing files or invalid JSON do not crash the program.

### Type safety

The project uses:

- type annotations
- Pydantic models
- mypy
- flake8

to improve reliability and maintainability.

---

# Performance Analysis

The implementation focuses on three objectives.

### Accuracy

The constrained decoder reliably predicts the correct function for most prompts.

Parameter extraction performs well for numeric, boolean and common string parameters.

Some highly ambiguous prompts remain challenging.

### Speed

The project completes within the required execution time.

JSON parsing and heuristic extraction have negligible overhead.

### Reliability

The decoder always generates valid JSON respecting the expected schema.

Error handling prevents crashes caused by invalid input files.

---

# Challenges Faced

Several challenges were encountered during development.

### Constrained decoding

Implementing token filtering while preserving valid model predictions required careful state management.

### Token vocabulary

The tokenizer vocabulary had to be decoded and indexed efficiently.

### Parameter extraction

Natural language prompts vary significantly.

Several heuristics were added for:

- quoted strings
- numbers
- file paths
- encodings
- database names
- template extraction

### Robustness

The parser was improved to correctly handle:

- missing files
- malformed JSON
- empty inputs

without terminating unexpectedly.

---

# Testing Strategy

The project was validated using multiple testing approaches.

### Functional tests

- simple arithmetic requests
- greeting requests
- string manipulation
- file operations

### Robustness tests

- missing input files
- invalid JSON
- empty datasets
- unknown prompts

### Code quality

The project was checked using:

- flake8
- mypy

### Manual verification

Generated JSON files were manually inspected to ensure they matched the expected schema.

---

# Example Usage

Example prompt:

```text
Add 12 and 45
```

Generated output:

```json
{
    "name": "fn_add_numbers",
    "parameters": {
        "a": 12,
        "b": 45
    }
}
```

Run the project:

```bash
make run
```

Output:

```
data/output/function_calling_results.json
```

---

# Resources

## Documentation

- Python Documentation
- JSON Documentation
- Pydantic Documentation
- Hugging Face Transformers Documentation
- Qwen Model Documentation

## AI Usage

Artificial intelligence tools were used during the development of this project to assist with:

- brainstorming implementation ideas;
- discussing constrained decoding strategies;
- improving documentation and README structure;
- reviewing code quality;
- identifying potential bugs and edge cases.

All architectural decisions, implementation, debugging, testing, and final validation were performed and reviewed by the project author.

---

# Project Structure

```
.
├── data/
│   ├── input/
│   └── output/
├── llm_sdk/
├── src/
├── Makefile
├── README.md
├── pyproject.toml
└── uv.lock
```
