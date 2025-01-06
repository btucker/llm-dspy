"""Tests for ContextRewriter component."""
import pytest
from llm_dspy.rag.enhanced import ContextRewriter

class TestContextRewriter:
    """Tests for the ContextRewriter class."""
    
    def test_preserves_entities(self):
        """Test that ContextRewriter preserves specific entity names and numbers."""
        rewriter = ContextRewriter()
        context = """
        Key transactions:
        - Client A paid $50,000 for Enterprise License
        - Client B invested $75,000 in Custom Development
        """
        question = "What were the key transactions?"
        
        result = rewriter.forward(context=context, question=question)
        answer = result.focused_context.lower()
        
        # Check that specific entities and numbers are preserved
        assert "client a" in answer, "Should preserve client names"
        assert "client b" in answer, "Should preserve client names"
        assert "50,000" in answer, "Should preserve exact amounts"
        assert "75,000" in answer, "Should preserve exact amounts"
        assert "enterprise license" in answer, "Should preserve transaction types"
        assert "custom development" in answer, "Should preserve transaction types" 