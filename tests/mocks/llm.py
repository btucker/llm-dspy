"""Mock LLM for testing."""

class Entry:
    def __init__(self, id, score, content, metadata=None):
        self.id = id
        self.score = score
        self.content = content
        self.text = content
        self.metadata = metadata
    
    def __str__(self):
        return f"Entry(id='{self.id}', score={self.score}, content='{self.content}', metadata={self.metadata})"

class MockResponse:
    def __init__(self, text):
        self.text = text
    
    def text_or_raise(self):
        return self.text

class MockModel:
    def __init__(self):
        self.kwargs = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "mock-model"
        }
        self.config = {}
    
    def __call__(self, text, **kwargs):
        # Extract field names from the prompt
        fields = []
        if "rationale" in text.lower():
            fields.append("rationale")
        if "search_query" in text.lower():
            fields.append("search_query")
        if "sub_questions" in text.lower():
            fields.append("sub_questions")
        if "answer" in text.lower():
            fields.append("answer")
        if "focused_context" in text.lower():
            fields.append("focused_context")
        
        # Generate appropriate response based on the question
        response = {}
        if any(term in text.lower() for term in ['transaction', 'amount', 'revenue']):
            base_response = "The key transactions in Q2 2023 included: Client A with $50,000 for an Enterprise License, " \
                          "Client B with $75,000 for Custom Development, and Client C with $100,000 for a Platform Subscription."
        elif any(term in text.lower() for term in ['security', 'token', 'auth']):
            base_response = "Security measures include token encryption at rest, HTTPS requirement for all endpoints, " \
                          "rate limiting enforcement, and monitoring of failed attempts."
        elif any(term in text.lower() for term in ['growth', 'market', 'expansion']):
            base_response = "The 15% YoY growth rate correlates with our 3% market share increase, suggesting that " \
                          "our expansion strategy is effectively driving both growth and market penetration."
        else:
            base_response = "This is a mock text"
        
        # Fill in requested fields
        for field in fields:
            if field == "rationale":
                response[field] = f"Based on the context, {base_response}"
            elif field == "search_query":
                response[field] = text  # Just echo the input as the search query
            elif field == "sub_questions":
                response[field] = ["What are the specific amounts?", "What are the transaction types?"]
            elif field == "answer":
                response[field] = base_response
            elif field == "focused_context":
                response[field] = base_response
        
        return response
    
    def stream(self, text, **kwargs):
        response = self(text)
        if "answer" in response:
            yield response["answer"]
        else:
            yield next(iter(response.values()))

class Collection:
    def __init__(self, name, model_id=None):
        self.name = name
        self.model_id = model_id
        self.documents = {}
        if not model_id:
            raise ValueError("model_id must be provided")
    
    def embed_multi(self, documents):
        """Store documents with their content."""
        for id, content in documents:
            self.documents[id] = content
    
    def similar(self, value=None, number=3):
        """Return Entry objects with actual content."""
        entries = []
        query_words = set(value.lower().split())
        
        for id, content in self.documents.items():
            content_words = set(content.lower().split())
            
            # Calculate similarity based on partial word matches
            score = 0
            for qword in query_words:
                # Check for exact word match
                if qword in content_words:
                    score += 1
                else:
                    # Check for partial word matches
                    for cword in content_words:
                        if (qword in cword) or (cword in qword):
                            score += 0.5
                            break
            
            # Normalize score by query length to get value between 0 and 1
            score = score / len(query_words) if query_words else 0
            
            # Always include document if it has any match
            if score > 0:
                entries.append(Entry(
                    id=id,
                    score=score,
                    content=content,
                    metadata={"id": id}
                ))
        
        # Sort by score in descending order
        entries.sort(key=lambda x: x.score, reverse=True)
        return entries[:number]

def get_model():
    return MockModel()
