# LLM-DSPy Plugin

This plugin adds support for using DSPy modules directly through LLM's command system. It allows you to leverage DSPy's powerful modules like ChainOfThought, ProgramOfThought, and others with a simple command interface.

## Installation

```bash
llm install -e path/to/llm-dspy
```

## Usage

The plugin adds a new `dspy` command that can be used to run any DSPy module with a specified signature. There are several ways to provide input:

### 1. Single Input Field - Positional Argument
When the module has only one input field, you can provide the value as a positional argument:
```bash
llm dspy "ChainOfThought(foo -> bar)" "input for foo"
```

### 2. Single Input Field - Stdin
When the module has only one input field, you can pipe the input:
```bash
cat "input for foo" | llm dspy "ChainOfThought(foo -> bar)"
```

### 3. Multiple Input Fields - Named Options
When the module has multiple input fields, use named options that match the field names:
```bash
llm dspy "ChainOfThought(foo, baz -> bar)" --foo "input for foo" --baz "input for baz"
```

### 4. RAG Support - Collection Names
Any input field can reference an LLM collection name for RAG functionality:
```bash
llm dspy "ChainOfThought(foo, baz -> bar)" --foo "input for foo" --baz "collection_name"
```

### 5. Stdin with Named Options
You can use stdin for any input field by setting its value to "stdin":
```bash
cat "input for foo" | llm dspy "ChainOfThought(foo, baz -> bar)" --foo stdin --baz "collection_name"
```

### Command Format
```
llm dspy [options] "ModuleName(input_fields -> output_fields)" [input]
```

### Input Rules
1. For single input field:
   - Use positional argument OR
   - Use stdin OR
   - Use named option

2. For multiple input fields:
   - Must use named options (--field-name value)
   - Any field can use stdin by setting value to "stdin"
   - Any field can reference a collection name for RAG

3. RAG Support:
   - If an input value matches an LLM collection name, it's used for retrieval
   - The collection is searched using other input values as queries
   - Retrieved context is automatically incorporated into the prompt

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

### Enhanced RAG Support

The plugin now includes advanced Retrieval-Augmented Generation (RAG) capabilities:

1. **Query Transformation**
   - Automatically transforms user questions into optimized search queries
   - Breaks down complex questions into sub-questions for multi-hop reasoning

2. **Context Processing**
   - Retrieves relevant passages from multiple searches
   - Rewrites and focuses context to be more relevant to the question
   - Combines information from multiple sources

3. **Multi-hop Reasoning**
   - Follows chains of questions to gather comprehensive information
   - Builds a reasoning path to explain the answer
   - Combines evidence from multiple searches

Example usage:
```bash
# Complex question requiring multiple lookups
llm dspy "ChainOfThought(context, question -> answer)" \
  --context my_collection \
  --question "What are the similarities and differences between neural networks and decision trees in terms of training time and interpretability?"

# Question requiring both historical and current information
llm dspy "ChainOfThought(context, question -> answer)" \
  --context historical_data \
  --question "How has climate change affected Arctic wildlife populations, and what are the projected future impacts?"
```

The plugin will:
1. Transform the question into optimal search queries
2. Perform multiple searches to gather comprehensive information
3. Rewrite and focus the context for relevance
4. Generate a well-reasoned answer with supporting evidence

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