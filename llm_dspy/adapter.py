import llm

class LLMAdapter:
    """Adapter to convert LLM responses into DSPy-compatible format."""
    def __init__(self):
        try:
            self.llm = llm.get_model()
        except llm.UnknownModelError:
            # If no model is set, try to get the default model
            try:
                self.llm = llm.get_model("gpt-3.5-turbo")
            except llm.UnknownModelError:
                # If that fails too, use the first available model
                models = llm.get_models()
                if not models:
                    raise RuntimeError("No LLM models available")
                self.llm = models[0]
