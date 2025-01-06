class MockModel:
    """Mock model for testing."""
    def __init__(self, response="This is a mock text"):
        self.response = response
        self.kwargs = {"temperature": 0.7}

    def __call__(self, prompt, **kwargs):
        return {"answer": self.response}

    def stream(self, prompt, **kwargs):
        yield self.response

    def prompt(self, prompt, **kwargs):
        return self.response

class MockLLMAdapter:
    """Mock LLM adapter for testing."""
    def __init__(self, model: Optional[Any] = None, **kwargs):
        self.model = model or MockModel(**kwargs)
        self.kwargs = kwargs
        self.temperature = kwargs.get("temperature", 0.7)

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        return self.model(prompt, **kwargs)

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream a mock response."""
        for chunk in self.model.stream(prompt, **kwargs):
            yield chunk

    def embed(self, text: str) -> list[float]:
        """Generate mock embeddings."""
        return [0.1] * 1536  # Standard OpenAI embedding size

    def prompt(self, prompt: str, **kwargs) -> str:
        """Generate a mock response from a prompt."""
        return self.model.prompt(prompt, **kwargs) 