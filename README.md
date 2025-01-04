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

Each input field in the signature becomes a command-line option. For example, if your signature is `ChainOfThought(context, question -> answer)`, you'll get `--context` and `--question` options.

Special field names trigger RAG functionality:
- `context`
- `background`
- `documents`
- `knowledge`

When these fields are present in the signature, their corresponding options expect an LLM collection name. The plugin will:
1. Use the collection to search for relevant content
2. Use another field's value as the search query (looking for `question`, `query`, or `prompt` fields)
3. If no query field is found, use all other input values combined

Examples:
```bash
# Basic RAG with context and question
llm dspy "ChainOfThought(context, question -> answer)" \
  --context my_collection \
  --question "What insights can you find?"

# Using background with a prompt
llm dspy "ChainOfThought(background, prompt, style -> response)" \
  --background knowledge_base \
  --prompt "Analyze this" \
  --style "concise"

# Multiple RAG sources
llm dspy "ChainOfThought(context, documents, query -> answer)" \
  --context primary_source \
  --documents secondary_source \
  --query "What are the differences?"
```

The plugin automatically detects which fields should trigger RAG functionality based on their names in the signature.

### RAG Support

The plugin supports Retrieval-Augmented Generation (RAG) using LLM's embeddings functionality. When using RAG, provide:
- `--context`: Name of the LLM collection to search
- `--question`: Question to use for searching the collection

Example:
```bash
# First, create and populate an LLM collection
llm embed my_collection document1.txt document2.txt

# Then use it with DSPy
llm dspy "ChainOfThought(context, question -> answer)" \
  --context my_collection \
  --question "What insights can you find in the documents?"

# You can also use it with other inputs
llm dspy "ChainOfThought(context, question, style -> answer)" \
  --context my_collection \
  --question "What are the key points?" \
  "Make it concise"
```

The plugin will:
1. Use the collection to search for relevant context using the question
2. Pass both the retrieved context and your question to the DSPy module
3. Any additional command-line arguments will be used for other input fields in the signature

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