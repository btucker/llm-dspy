"""Mock LLM for testing."""

class Entry:
    def __init__(self, id, score, content, metadata=None):
        self.id = id
        self.score = score
        self.content = content
        self.metadata = metadata
    
    def __str__(self):
        return f"Entry(id='{self.id}', score={self.score}, content='{self.content}', metadata={self.metadata})"

class MockResponse:
    def __init__(self, text):
        self.text = text
    
    def text_or_raise(self):
        return self.text

class MockModel:
    def __call__(self, text, **kwargs):
        return {"answer": "This is a mock text"}
    
    def stream(self, text, **kwargs):
        yield "This is a mock text"

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
        for id, content in self.documents.items():
            entries.append(Entry(
                id=id,
                score=0.8,
                content=content,
                metadata={"id": id}
            ))
        return entries[:number]

def get_model():
    return MockModel()
