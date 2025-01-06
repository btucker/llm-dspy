# LLM-DSPy Plugin

This plugin adds support for using [DSPy](https://dspy.ai/) modules directly through [llm](https://llm.datasette.io/en/stable/index.html)'s command system. It allows you to leverage DSPy's modules like ChainOfThought, ProgramOfThought, and others with a simple command interface.

## Installation

```bash
llm install -e path/to/llm-dspy
```

## Usage

The plugin adds a new `dspy` command that can be used to run any DSPy module with a specified signature. There are several ways to provide input:

### 1. Single Input Field - Positional Argument
When the module has only one input field, you can provide the value as a positional argument:
```bash
llm dspy "ChainOfThought(question: str -> answer: str)" "Explain how photosynthesis works in simple terms."

# With type annotation for classification
llm dspy "Predict(text: str -> sentiment: Literal['positive', 'negative', 'neutral'])" "This product exceeded all my expectations!"
```

### 2. Single Input Field - Stdin
When the module has only one input field, you can pipe the input:
```bash
echo "What are the main differences between REST and GraphQL?" | llm dspy "ChainOfThought(question: str -> answer: str)"

# Classify code complexity
cat source_file.py | llm dspy "Predict(code: str -> complexity: Literal['O(1)', 'O(n)', 'O(n^2)', 'O(2^n)'])"
```

### 3. Multiple Input Fields - Named Options
When the module has multiple input fields, use named options that match the field names:
```bash
llm dspy "ChainOfThought(topic: str, audience: str -> explanation: str)" \
  --topic "quantum computing" \
  --audience "high school students"

# With structured output fields
llm dspy "ProgramOfThought(bug_report: str, system_context: str -> root_cause: str, severity: Literal['low', 'medium', 'high'], fix_steps: List[str])" \
  --bug_report "Application crashes when uploading files larger than 1GB" \
  --system_context "Node.js backend with S3 storage"
```

### 4. RAG Support - Collection Names
Any input field can reference an LLM collection name for RAG functionality:
```bash
llm dspy "ChainOfThought(context: str, question: str -> answer: str)" \
  --context "company_docs" \
  --question "What is our refund policy for international customers?"

# With fact extraction
llm dspy "ProgramOfThought(context: str, query: str -> dates: List[str], amounts: List[float], entities: List[str])" \
  --context "financial_records" \
  --query "Extract all transaction dates, amounts, and involved parties from Q2 reports"
```

### 5. Stdin with Named Options
You can use stdin for any input field by setting its value to "stdin":
```bash
cat research_paper.txt | llm dspy "ChainOfThought(paper: str, style: str -> summary: str)" \
  --paper stdin \
  --style "technical but accessible"

# Code review with structured feedback
cat pull_request.diff | llm dspy "ProgramOfThought(diff: str, standards: str -> feedback: List[str], risk_level: Literal['low', 'medium', 'high'], approval: bool)" \
  --diff stdin \
  --standards "company_coding_standards"
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
  --context "medical_research" \
  --question "What are the latest findings on mRNA vaccine effectiveness?"

# Medical research with structured analysis
llm dspy "ProgramOfThought(context: str, study_query: str -> findings: List[str], confidence_level: Literal['high', 'medium', 'low'], limitations: List[str], next_steps: List[str])" \
  --context "medical_research" \
  --study_query "Analyze the efficacy of different COVID-19 variants"

# Using background with a prompt
llm dspy "ChainOfThought(background, prompt, style -> response)" \
  --background "legal_precedents" \
  --prompt "Analyze the implications of this case for privacy law" \
  --style "professional legal analysis"

# Legal analysis with classification
llm dspy "ProgramOfThought(background, case_details -> jurisdiction: str, precedent_relevance: float, risk_factors: list[str], recommendation: str)" \
  --background "legal_precedents" \
  --case_details "Evaluate data privacy compliance for our new feature"

# Multiple RAG sources
llm dspy "ChainOfThought(context, documents, query -> analysis)" \
  --context "financial_reports" \
  --documents "market_research" \
  --query "What market trends suggest potential growth opportunities?"

# Financial analysis with metrics
llm dspy "ProgramOfThought(market_data: str, competitor_data: str, query: str -> growth_rate: float, risk_score: conint(ge=1, le=10), opportunities: List[str], threats: List[str])" \
  --market_data "financial_reports" \
  --competitor_data "market_research" \
  --query "Evaluate market position for Q3 planning"

# Complex research analysis requiring multiple lookups
llm dspy "ProgramOfThought(context, question -> answer)" \
  --context "research_papers" \
  --question "What are the environmental and economic trade-offs between different renewable energy sources for urban environments?"

# Detailed sustainability analysis
llm dspy "ProgramOfThought(context: str, analysis_request: str -> environmental_impact: confloat(ge=0, le=100), cost_efficiency: confloat(ge=0, le=100), implementation_challenges: List[str], recommendations: List[str])" \
  --context "research_papers" \
  --analysis_request "Compare solar vs wind power for metropolitan areas"

# Technical documentation analysis with historical context
llm dspy "ChainOfThought(context, question -> answer)" \
  --context "system_architecture" \
  --question "How has our authentication system evolved over the past year, and what security improvements were implemented?"

# Security audit with compliance check
llm dspy "ProgramOfThought(context: str, audit_scope: str -> compliance_status: Literal['compliant', 'partial', 'non_compliant'], vulnerabilities: List[str], risk_level: Literal['low', 'medium', 'high', 'critical'], action_items: List[str])" \
  --context "system_architecture" \
  --audit_scope "Evaluate OAuth2 implementation against OWASP standards"

# Multi-source analysis for business decisions
llm dspy "ProgramOfThought(market_data, competitor_analysis, question -> recommendation)" \
  --market_data "market_research" \
  --competitor_analysis "competitor_reports" \
  --question "Based on current market conditions and competitor strategies, what product features should we prioritize for Q3?"

# Strategic planning with metrics
llm dspy "ProgramOfThought(market_data, competitor_analysis, objectives -> priority_score: dict[str, float], timeline: dict[str, str], resource_requirements: list[str], expected_roi: float)" \
  --market_data "market_research" \
  --competitor_analysis "competitor_reports" \
  --objectives "Identify top 3 features for competitive advantage in Q3"
```

The plugin automatically detects which fields should trigger RAG functionality based on their names in the signature.

### RAG Support

The plugin supports Retrieval-Augmented Generation (RAG) using LLM's embeddings functionality. When using RAG, provide:
- `--context`: Name of the LLM collection to search
- `--question`: Question to use for searching the collection

Example:
```bash
# First, create and populate an LLM collection with technical documentation
llm embed technical_docs api_reference.md architecture.md deployment_guide.md

# Then use it with DSPy for technical queries
llm dspy "ChainOfThought(context, question -> answer)" \
  --context technical_docs \
  --question "How does our authentication system handle OAuth2 token refresh?"

# Combine with style guidance for specific audiences
llm dspy "ChainOfThought(context, question, style -> answer)" \
  --context technical_docs \
  --question "Explain our microservices architecture" \
  --style "suitable for non-technical stakeholders"
```

The plugin will:
1. Use the collection to search for relevant context using the question
2. Pass both the retrieved context and your question to the DSPy module
3. Any additional command-line arguments will be used for other input fields in the signature

### Enhanced RAG Support

The plugin includes advanced Retrieval-Augmented Generation (RAG) capabilities by default when using collections:

1. **Automatic Query Transformation**
   - Transforms user questions into optimized search queries
   - Breaks down complex questions into sub-questions for multi-hop reasoning

2. **Context Processing**
   - Retrieves and rewrites context to focus on the specific question
   - Preserves important details like dates, amounts, and technical terms
   - Maintains structure of lists and bullet points

3. **Intelligent Answer Generation**
   - Adapts response format based on question type
   - Ensures all relevant details from context are included
   - Maintains exact numbers, dates, and technical terms

Example:
```bash
# Use a collection with enhanced RAG features automatically
llm dspy "ChainOfThought(context, question -> answer)" \
  --context technical_docs \
  --question "What are our key financial metrics for Q2 2023?"

# The plugin will:
# 1. Transform the question into optimized search queries
# 2. Retrieve and process relevant context
# 3. Generate a comprehensive answer with all key metrics
```

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