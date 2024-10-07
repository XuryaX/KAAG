import anthropic
from typing import List
from .base import BaseLLM

class AnthropicLLM(BaseLLM):
    """
    Implementation of Anthropic's Claude model.
    """

    def __init__(self, api_key: str):
        self.client = anthropic.Client(api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.completion(
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            model="claude-2",
            max_tokens_to_sample=300,
            **kwargs
        )
        return response.completion

    def embed(self, text: str) -> List[float]:
        # Note: As of my last update, Anthropic doesn't provide a public embedding API
        # This is a placeholder implementation
        raise NotImplementedError("Embedding is not supported for Anthropic models")