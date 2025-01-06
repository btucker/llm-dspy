"""Custom DSPy provider for mock model."""
from typing import Any, Dict

class MockProvider:
    """Custom DSPy provider for mock model."""
    
    @staticmethod
    def is_provider_model(model: str) -> bool:
        """Check if the model is supported by this provider."""
        return model == "mock-model"
    
    def __init__(self, model: str, **kwargs):
        """Initialize the provider."""
        self.model = model
        self.kwargs = kwargs
    
    def basic_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Make a basic request to the model."""
        # Extract field names from the prompt
        fields = []
        if "rationale" in prompt.lower():
            fields.append("rationale")
        if "search_query" in prompt.lower():
            fields.append("search_query")
        if "sub_questions" in prompt.lower():
            fields.append("sub_questions")
        if "answer" in prompt.lower():
            fields.append("answer")
        if "focused_context" in prompt.lower():
            fields.append("focused_context")
        
        # Generate appropriate response based on the question
        response = {}
        
        # Handle query transformation
        if "search_query" in fields:
            response["search_query"] = "Find transactions and amounts in Q2 2023"
            response["sub_questions"] = ["What were the specific transaction amounts?", "Who were the clients involved?"]
            response["rationale"] = "Breaking down the question to find specific transaction details"
            return response
        
        # Handle context rewriting
        if "focused_context" in fields:
            response["focused_context"] = "The key transactions in Q2 2023 included: Client A with $50,000 for an Enterprise License, " \
                                       "Client B with $75,000 for Custom Development, and Client C with $100,000 for a Platform Subscription."
            return response
        
        # Handle final answer generation
        if any(term in prompt.lower() for term in ['transaction', 'amount', 'revenue']):
            response["answer"] = "The key transactions in Q2 2023 included: Client A with $50,000 for an Enterprise License, " \
                               "Client B with $75,000 for Custom Development, and Client C with $100,000 for a Platform Subscription."
            response["rationale"] = "Found specific transaction details in the context"
        elif any(term in prompt.lower() for term in ['security', 'token', 'auth']):
            response["answer"] = "Security measures include token encryption at rest, HTTPS requirement for all endpoints, " \
                               "rate limiting enforcement, and monitoring of failed attempts."
            response["rationale"] = "Identified security features from the documentation"
        elif any(term in prompt.lower() for term in ['growth', 'market', 'expansion']):
            response["answer"] = "The 15% YoY growth rate correlates with our 3% market share increase, suggesting that " \
                               "our expansion strategy is effectively driving both growth and market penetration."
            response["rationale"] = "Analyzed growth metrics and market performance"
        else:
            response["answer"] = "This is a mock text"
            response["rationale"] = "Default response for unknown query type"
        
        return response
    
    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call the model with a prompt."""
        return self.basic_request(prompt, **kwargs) 