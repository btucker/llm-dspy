# LLM-DSPy Plugin

This plugin adds support for using DSPy modules as templates in LLM. It allows you to leverage DSPy's powerful modules like ChainOfThought, ChainOfProgram, and others directly through LLM's template system.

## Installation

```bash
llm install -e path/to/llm-dspy
```

## Usage

Create a template file (e.g. `templates.yaml`) with a DSPy template:

```yaml
chain-of-thought:
  type: dspy
  dspy:
    module: ChainOfThought
  system: You are a helpful assistant that thinks through problems step by step.
  prompt: |
    Please help me solve this problem:
    $input

chain-of-program:
  type: dspy
  dspy:
    module: ChainOfProgram
    config:
      max_steps: 5
    signature: ProgramSignature
  system: You are a helpful programming assistant that breaks down tasks into steps.
  prompt: |
    Please help me implement this programming task:
    $input

qa-template:
  type: dspy
  dspy:
    module: ChainOfThought
    signature:
      input_fields: ["question"]
      output_fields: ["answer", "confidence"]
      description: A question-answering signature that includes confidence scores
  system: You are a helpful question-answering assistant.
  prompt: |
    Please answer this question:
    $input

sentiment-template:
  type: dspy
  dspy:
    module: ChainOfThought
    signature: "sentence -> sentiment: bool"
  system: You are a sentiment analysis assistant.
  prompt: |
    Please analyze the sentiment of this text:
    $input

rag-template:
  type: dspy
  dspy:
    module: RAG
    signature: "context: list[str], question: str -> answer: str"
  system: You are a retrieval-augmented question answering assistant.
  prompt: |
    Please answer this question using the provided context:
    $input
```

Then use it with LLM:

```bash
llm -t chain-of-thought "What is 15% of 85?"
llm -t chain-of-program "Write a function to find prime numbers up to n"
llm -t qa-template "What is the capital of France?"
llm -t sentiment-template "This movie was absolutely fantastic!"
llm -t rag-template "Based on the provided documents, when was the company founded?"
```

## Supported DSPy Modules

The plugin supports any DSPy module that can be initialized with configuration parameters and called with a prompt string. Some commonly used modules include:

- ChainOfThought
- ChainOfProgram
- ReAct
- ProgramOfThought

## Configuration

The template accepts the following configuration:

- `dspy.module`: The name of the DSPy module to use (required)
- `dspy.config`: Configuration parameters to pass to the DSPy module (optional)
- `dspy.signature`: Signature specification in one of three formats (optional):
  1. Predefined signature: Use the signature class name as a string (e.g. `"ProgramSignature"`)
  2. Custom signature: Specify `input_fields`, `output_fields`, and optionally a `description`
  3. Inline signature: Use a string with the format `"inputs -> outputs"`, where:
     - Simple format: `"question -> answer"`
     - With types: `"sentence -> sentiment: bool"`
     - Multiple fields: `"context: list[str], question: str -> answer: str"`
- `system`: System prompt to set context (optional)
- `prompt`: Template for the user prompt (optional) 