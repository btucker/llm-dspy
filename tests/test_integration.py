import pytest
import tempfile
import os
import click
from click.testing import CliRunner
import dspy
import llm
from llm_dspy.cli.commands import register_commands
from sqlite_utils import Database
from tests.mocks.llm import Collection

@pytest.fixture(autouse=True)
def setup_dspy():
    """Configure DSPy with real language model."""
    lm = dspy.LM('openai/gpt-4')
    dspy.configure(lm=lm)
    yield
    # Clean up if needed

@pytest.fixture
def runner():
    """Click CLI runner for testing."""
    return CliRunner()

@pytest.fixture
def cli():
    """Create CLI with registered commands."""
    cli_group = click.Group(name="cli")
    register_commands(cli_group)
    return cli_group

@pytest.fixture
def test_files():
    """Create test files with realistic content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        api_docs_path = os.path.join(temp_dir, 'api_docs.md')
        financial_report_path = os.path.join(temp_dir, 'financial_report.md')
        
        with open(api_docs_path, 'w') as f:
            f.write('''
# Authentication API
Our OAuth2 implementation supports the following flows:
- Authorization Code
- Client Credentials
- Refresh Token

Token refresh is handled automatically by the client library. When a token expires, the library will:
1. Check if a refresh token is available
2. If available, make a request to the token endpoint
3. Update the stored tokens with the new access token
4. Retry the original request

## Security Considerations
- All tokens are encrypted at rest
- HTTPS is required for all endpoints
- Rate limiting is enforced
- Failed attempts are logged and monitored
            ''')
        
        with open(financial_report_path, 'w') as f:
            f.write('''
# Q2 Financial Report 2023

## Overview
Revenue: $1.2M
Growth Rate: 15% YoY
Profit Margin: 22%

## Key Transactions
- March 15: Client A, $50,000 (Enterprise License)
- April 2: Client B, $75,000 (Custom Development)
- May 20: Client C, $100,000 (Platform Subscription)

## Market Analysis
- Market share increased by 3%
- Two new competitors entered the space
- Customer retention rate at 92%

## Growth Opportunities
1. International expansion
2. New product features
3. Strategic partnerships
            ''')
        
        yield {
            'dir': temp_dir,
            'api_docs': api_docs_path,
            'financial_report': financial_report_path
        }

@pytest.fixture
def collections(test_files):
    """Set up test collections with real components."""
    import logging
    import llm
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Create collections using llm directly
    technical_docs = llm.Collection("technical_docs", model_id="ada-002")
    financial_records = llm.Collection("financial_records", model_id="ada-002")
    logger.debug(f"Created collections: technical_docs={technical_docs}, financial_records={financial_records}")
    
    # Register collections with llm
    llm.collections = {
        'technical_docs': technical_docs,
        'financial_records': financial_records
    }
    
    # Embed documents
    with open(test_files['api_docs'], 'r') as f:
        content = f.read()
        logger.debug(f"Embedding api_docs content: {content[:100]}...")
        technical_docs.embed("api_docs", content, metadata={"filename": "api_docs.md"}, store=True)
    
    with open(test_files['financial_report'], 'r') as f:
        content = f.read()
        logger.debug(f"Embedding financial_report content: {content[:100]}...")
        financial_records.embed("financial_report", content, metadata={"filename": "financial_report.md"}, store=True)
    
    yield {
        'technical_docs': technical_docs,
        'financial_records': financial_records,
        'dir': test_files['dir']
    }

def test_basic_collection_retrieval(collections):
    """Basic test to verify collection retrieval works."""
    collection = collections['financial_records']
    
    # Test direct retrieval
    results = collection.similar(value="revenue", number=1)
    assert len(results) > 0, "Should find at least one result"
    assert "revenue" in results[0].content.lower(), "Result should contain 'revenue'"
    
    # Test with more specific query
    results = collection.similar(value="key transactions", number=1)
    assert len(results) > 0, "Should find at least one result"
    assert "client a" in results[0].content.lower(), "Result should contain client information"

def test_rag_pipeline_integration(collections):
    """Test the complete RAG pipeline with real components."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    
    # Initialize RAG module with financial collection
    rag = EnhancedRAGModule(collection_name="financial_records", k=10)
    
    # Ask a question that requires understanding the financial report
    result = rag.forward(question="What were the key transactions in Q2 2023?")
    
    # Verify the response contains relevant information from the actual document
    answer = result.answer.lower()
    assert "client a" in answer and "client b" in answer and "client c" in answer, \
        "Response should mention all three clients from the document"
    assert "50,000" in answer and "75,000" in answer and "100,000" in answer, \
        "Response should include the actual transaction amounts from the document"
    assert "enterprise license" in answer and "custom development" in answer and "platform subscription" in answer, \
        "Response should mention the actual transaction types from the document"

def test_security_audit_integration(collections):
    """Test RAG pipeline with technical documentation."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    
    # Initialize RAG module with technical docs
    rag = EnhancedRAGModule(collection_name="technical_docs", k=10)
    
    # Ask about security features
    result = rag.forward(question="What security measures are in place for token handling?")
    
    # Verify the response contains actual security information from the document
    answer = result.answer.lower()
    assert "encrypt" in answer and "rest" in answer, "Response should mention token encryption from the document"
    assert "https" in answer and ("endpoint" in answer or "transmission" in answer), "Response should mention HTTPS requirement from the document"
    assert "rate limit" in answer, "Response should mention rate limiting from the document"
    assert ("failed" in answer or "fail" in answer) and "monitor" in answer, "Response should mention monitoring from the document"

def test_multi_hop_reasoning_integration(collections):
    """Test multi-hop reasoning in RAG pipeline."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    
    # Initialize RAG module with financial collection
    rag = EnhancedRAGModule(collection_name="financial_records", max_hops=2)
    
    # Ask a complex question requiring multiple hops
    result = rag.forward(
        question="Given the growth rate and market share increase, what's the relationship between our growth and market expansion?"
    )
    
    # Verify the response contains actual metrics and analysis from the document
    answer = result.answer.lower()
    assert "15%" in answer and "growth" in answer, "Response should mention the actual growth rate"
    assert "3%" in answer and "market" in answer, "Response should mention the actual market share increase"
    assert "92%" in answer and "retention" in answer, "Response should mention the actual retention rate"
    assert "competitors" in answer or "competition" in answer, "Response should mention market competition"

def test_cli_rag_integration(runner, cli, collections):
    """Test RAG integration through CLI interface."""
    result = runner.invoke(cli, [
        'dspy',
        'EnhancedRAGModule(question -> answer)',
        '--collection_name', 'financial_records',
        '--k', '10',
        '--question', 'What was the total revenue for Q2 2023?'
    ])
    
    assert result.exit_code == 0
    output = result.output.lower()
    assert any(value in output for value in ["$225,000", "$1.2m", "$1.2 million"]), \
        "Response should include either the total transaction amount or the overview revenue"

def test_basic_embedding_retrieval(collections):
    """Test that we can retrieve content we just embedded."""
    import logging
    logger = logging.getLogger(__name__)
    
    collection = collections['financial_records']
    
    # Try to retrieve with exact text from the document
    query = "Revenue: $1.2M"
    logger.debug(f"Searching for: {query}")
    results = collection.similar(value=query, number=1)
    logger.debug(f"Got results: {results}")
    
    assert len(results) > 0, "Should find at least one result"
    assert results[0].content is not None, "Result should have content"
    assert "$1.2M" in results[0].content, "Result should contain the exact text we searched for"

def test_database_contents(collections):
    """Test that content is properly stored in the database."""
    import logging
    logger = logging.getLogger(__name__)
    
    collection = collections['financial_records']
    db = collection.db
    
    # List all tables
    tables = db.tables
    logger.debug(f"Database tables: {tables}")
    
    # Check embeddings table
    embeddings = list(db.query("SELECT * FROM embeddings"))
    logger.debug(f"Embeddings table contents: {embeddings}")
    
    assert len(embeddings) > 0, "Should have embeddings stored in database"
    assert any("$1.2M" in str(row) for row in embeddings), "Should find our test content in embeddings"