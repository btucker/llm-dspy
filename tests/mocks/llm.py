class MockResponse:
    def __init__(self, text):
        self.text = text
    
    def text_or_raise(self):
        return self.text

class MockModel:
    def prompt(self, text):
        return MockResponse("Here's a step by step solution...")

class Collection:
    def __init__(self, name, model_id=None):
        self.name = name
        self.model_id = model_id
        if not model_id:
            raise ValueError("Either model= or model_id= must be provided when creating a new collection")
    
    def similar(self, text=None, n=3):
        return ["This is some relevant context from the database"]

def get_model():
    return MockModel()

def get_models():
    return [MockModel()]

def get_collection(name):
    return Collection(name, model_id="ada-002")
