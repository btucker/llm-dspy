"""Mock LLM for testing."""

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
        if not model_id:
            raise ValueError("model_id must be provided")
    
    def similar(self, text=None, n=3):
        return [{"text": "This is a mock document"}]

def get_model():
    return MockModel()
