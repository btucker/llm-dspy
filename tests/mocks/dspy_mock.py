"""Mock DSPy module for testing."""

class MockPrediction:
    """Mock prediction from DSPy."""
    def __init__(self, answer="Mock answer", text=None, **kwargs):
        self.answer = answer
        self.text = text or answer
        for key, value in kwargs.items():
            setattr(self, key, value) 