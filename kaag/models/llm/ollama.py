import requests
from typing import List
from .base import BaseLLM

class OllamaLLM(BaseLLM):
    """
    Implementation for local Ollama models.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, **kwargs, "stream": False}
        )
        return response.json()["response"]

    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return response.json()["embedding"]