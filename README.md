# LLM-DSPy Plugin

This plugin adds support for using DSPy modules directly through LLM's command system. It allows you to leverage DSPy's powerful modules like ChainOfThought, ProgramOfThought, and others with a simple command interface.

## Installation

```bash
llm install -e path/to/llm-dspy
```

## Usage

The plugin adds a new `dspy` command that can be used to run any DSPy module with a specified signature:

```bash
# Basic usage
llm dspy "ChainOfThought(question -> answer)" "What is 15% of 85?"
llm dspy "ProgramOfThought(question -> answer:int)" "How many letter Rs are in the word Strawberry"

# Chain of thought with multiple inputs/outputs
llm dspy "ChainOfThought(context, question -> answer, confidence)" "Here's some context..." "What can you tell me?"

# Enable verbose logging
llm dspy -v "ChainOfThought(question -> answer)" "What is 15% of 85?"
llm dspy --verbose "ChainOfThought(question -> answer)" "What is 15% of 85?"
```

The command format is:
```
llm dspy [options] "ModuleName(input_fields -> output_fields : type)" "Your prompt here"
```

### Options

- `-v, --verbose`: Enable verbose logging to see detailed DSPy execution information

### Input Format

- For single input signatures, all arguments after the signature are joined together
- For multiple input signatures, each input should be properly quoted if it contains spaces
- Output fields can optionally specify types (e.g., `answer:int`)

## Supported DSPy Modules

The plugin supports any DSPy module that can be initialized with a signature and called with a prompt string. Some commonly used modules include:

- ChainOfThought
- ProgramOfThought
- Predict

## Error Handling

The plugin will raise clear error messages if:
- The module name is not found in DSPy
- The signature format is invalid
- The module fails to process the prompt
- The number of inputs doesn't match the signature

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