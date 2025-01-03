
class MockResponse:
    def __init__(self, text):
        self.text = text
    
    def text_or_raise(self):
        return self.text

class MockModel:
    def prompt(self, text):
        return MockResponse("Here's a step by step solution...")

def get_model():
    return MockModel()
