# LLM-DSPy Plugin

This plugin adds support for using DSPy modules directly through LLM's command system. It allows you to leverage DSPy's powerful modules like ChainOfThought, ChainOfProgram, and others with a simple command interface.

## Installation

```bash
llm install -e path/to/llm-dspy
```

## Usage

The plugin adds a new `dspy` command that can be used to run any DSPy module with a specified signature:

```bash
# Basic usage
llm dspy "ChainOfThought(question -> answer)" "What is 15% of 85?"

# Chain of thought with multiple inputs/outputs
llm dspy "ChainOfThought(context, question -> answer, confidence)" "Here's some context..." "What can you tell me?"

# Using other DSPy modules
llm dspy "ReAct(task -> solution)" "Write a function to find prime numbers up to n"
llm dspy "ProgramOfThought(problem -> code, explanation)" "Implement a binary search tree"
```

The command format is:
```
llm dspy "ModuleName(input_fields -> output_fields)" "Your prompt here"
```

## Supported DSPy Modules

The plugin supports any DSPy module that can be initialized with a signature and called with a prompt string. Some commonly used modules include:

- ChainOfThought
- ChainOfProgram
- ReAct
- ProgramOfThought

## Error Handling

The plugin will raise clear error messages if:
- The module name is not found in DSPy
- The signature format is invalid
- The module fails to process the prompt

## Development

To run unit tests:
```bash
pytest tests/test_dspy_command.py
```

To run integration tests (requires LLM to be installed):
```bash
pytest tests/test_integration.py
```

Note: Integration tests will install and uninstall the plugin as part of testing. 