import pytest
import tempfile
import os
import click
from click.testing import CliRunner
import dspy
import llm
from llm_dspy.cli.commands import register_commands
from sqlite_utils import Database

@pytest.fixture(autouse=True)
def setup_dspy():
    """Configure DSPy with real language model."""
    lm = dspy.LM('openai/gpt-4')
    dspy.configure(lm=lm)
    yield

@pytest.fixture
def test_files():
    """Create test files with realistic content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Technical documentation
        api_docs_path = os.path.join(temp_dir, 'api_docs.md')
        with open(api_docs_path, 'w') as f:
            f.write('''
# Authentication API Documentation

## Overview
Our authentication system provides secure OAuth2 implementation with automatic token refresh handling.

## OAuth2 Flows
We support the following standard OAuth2 flows:
- Authorization Code Flow (for web applications)
- Client Credentials Flow (for server-to-server)
- Refresh Token Flow (for long-term access)

## Token Management
Token refresh is handled automatically by our client library:
1. When a token expires, the library detects the expiration
2. If a refresh token is available, it's used to request a new access token
3. The new access token is stored securely
4. The original request is retried automatically

## Security Measures
- All tokens are encrypted at rest using AES-256
- HTTPS is required for all authentication endpoints
- Rate limiting is enforced (100 requests per minute)
- Failed authentication attempts are logged and monitored
- IP-based blocking after 5 failed attempts
- Regular security audits are performed

## Evolution and Improvements
Recent security enhancements (2023):
1. Added support for PKCE in Authorization Code flow
2. Implemented JWE for token encryption
3. Enhanced monitoring with real-time alerts
4. Added support for hardware security keys
            ''')
        
        # Financial documentation
        financial_report_path = os.path.join(temp_dir, 'financial_report.md')
        with open(financial_report_path, 'w') as f:
            f.write('''
# Q2 2023 Financial Report

## Executive Summary
- Total Revenue: $1.2M (15% YoY growth)
- Profit Margin: 22%
- Customer Growth: 25%

## Key Transactions
- March 15, 2023: Client A signed Enterprise License agreement ($50,000)
  Details: Annual license for 100 users, includes premium support
  
- April 2, 2023: Client B contracted Custom Development ($75,000)
  Details: Integration with their ERP system, 3-month project
  
- May 20, 2023: Client C purchased Platform Subscription ($100,000)
  Details: 2-year commitment, enterprise-wide deployment

## Market Analysis
- Market share increased to 12% (up 3% from Q1)
- Two new competitors entered the space
- Customer retention rate: 92%

## Growth Strategy
1. International Expansion
   - Target: APAC region
   - Timeline: Q3 2023
   - Expected investment: $200K

2. Product Development
   - AI/ML features
   - Mobile application
   - API marketplace

3. Strategic Partnerships
   - Cloud providers
   - System integrators
   - Technology partners
            ''')
        
        yield {
            'dir': temp_dir,
            'api_docs': api_docs_path,
            'financial_report': financial_report_path
        }

@pytest.fixture
def collections(test_files):
    """Set up real collections with LLM's vector search."""
    # Create collections
    technical_docs = llm.Collection("technical_docs", model_id="3-small")
    financial_records = llm.Collection("financial_records", model_id="3-small")
    
    # Register collections
    llm.collections = {
        'technical_docs': technical_docs,
        'financial_records': financial_records
    }
    
    # Embed documents with content and metadata
    with open(test_files['api_docs'], 'r') as f:
        technical_docs.embed(
            "auth_docs",
            f.read(),
            metadata={"type": "documentation", "topic": "authentication"},
            store=True
        )
    
    with open(test_files['financial_report'], 'r') as f:
        financial_records.embed(
            "q2_2023_report",
            f.read(),
            metadata={"type": "report", "period": "Q2 2023"},
            store=True
        )
    
    yield {
        'technical_docs': technical_docs,
        'financial_records': financial_records
    }
    
    # Cleanup
    technical_docs.delete()
    financial_records.delete()

def test_oauth_token_refresh(collections):
    """Test the OAuth2 token refresh example from README."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    
    rag = EnhancedRAGModule(collection_name="technical_docs", k=5)
    result = rag.forward(
        question="What are the specific steps in our token refresh process, including what happens to the original request?"
    )
    answer = result.answer.lower()
    
    # Verify key aspects of token refresh are mentioned
    assert "automatic" in answer, "Should mention automatic refresh"
    assert any(word in answer for word in ["expire", "expiration"]), "Should mention token expiration"
    assert "refresh token" in answer, "Should mention refresh token"
    assert "new access token" in answer, "Should mention new access token"
    assert any(word in answer for word in ["retry", "retried", "retrying"]), "Should mention request retry"

def test_security_audit(collections):
    """Test the security audit example from README."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    
    rag = EnhancedRAGModule(collection_name="technical_docs", k=5)
    result = rag.forward(
        question="Evaluate our OAuth2 implementation against security best practices. What security measures are in place?"
    )
    answer = result.answer.lower()
    
    # Verify all security measures are covered
    assert "encrypt" in answer, "Should mention encryption"
    assert "aes-256" in answer, "Should mention encryption standard"
    assert "https" in answer, "Should mention HTTPS requirement"
    assert "rate limit" in answer, "Should mention rate limiting"
    assert "monitor" in answer, "Should mention monitoring"
    assert "fail" in answer and "attempt" in answer, "Should mention failed attempts"
    assert any(word in answer for word in ["audit", "security measures", "security evaluation"]), "Should mention security evaluation"

def test_financial_analysis(collections):
    """Test the financial analysis example from README."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    
    # First test: Overview metrics
    rag = EnhancedRAGModule(collection_name="financial_records", k=5)
    result = rag.forward(
        question="What are our key financial metrics for Q2 2023, including revenue, growth rate, and profit margin?"
    )
    answer = result.answer.lower()
    
    # Verify financial metrics
    assert "$1.2m" in answer or "$1.2 million" in answer, "Should mention total revenue"
    assert "15%" in answer and "growth" in answer, "Should mention growth rate"
    assert "22%" in answer and "margin" in answer, "Should mention profit margin"
    
    # Second test: Transactions and market position
    result = rag.forward(
        question="List all key transactions from Q2 2023 in chronological order and describe our market position."
    )
    answer = result.answer.lower()
    
    # Verify key transactions
    assert "march 15" in answer and "$50,000" in answer, "Should mention March transaction"
    assert "april 2" in answer and "$75,000" in answer, "Should mention April transaction"
    assert "may 20" in answer and "$100,000" in answer, "Should mention May transaction"
    
    # Verify market analysis
    assert "market share" in answer and "12%" in answer, "Should mention market share"
    assert "92%" in answer and "retention" in answer, "Should mention retention rate"

def test_multi_hop_reasoning(collections):
    """Test multi-hop reasoning with complex questions."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    
    rag = EnhancedRAGModule(collection_name="technical_docs", k=5)
    result = rag.forward(
        question="How has our authentication system evolved in terms of security, and what specific improvements were made in 2023?"
    )
    answer = result.answer.lower()
    
    # Verify recent improvements
    assert "pkce" in answer, "Should mention PKCE addition"
    assert "jwe" in answer, "Should mention JWE implementation"
    assert "monitor" in answer and "alert" in answer, "Should mention enhanced monitoring"
    assert "hardware" in answer and "key" in answer, "Should mention hardware security keys"

def test_error_handling(collections):
    """Test error handling without mocks."""
    from llm_dspy.rag.enhanced import EnhancedRAGModule
    import pytest
    
    # Test with non-existent collection
    with pytest.raises(KeyError, match=r"Collection 'non_existent' not found"):
        rag = EnhancedRAGModule(collection_name="non_existent", k=5)
    
    # Test with empty collection name
    with pytest.raises(ValueError, match="collection_name must be provided"):
        rag = EnhancedRAGModule(collection_name="", k=5)
    
    # Test with empty query
    rag = EnhancedRAGModule(collection_name="technical_docs", k=5)
    result = rag.forward(question="")
    assert result.answer.strip(), "Should handle empty query gracefully"