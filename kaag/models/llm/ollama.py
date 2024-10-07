# kaag/models/llm/ollama.py

import requests
from typing import List, Dict
from .base import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, **kwargs}
        )
        return response.json()["response"]

    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return response.json()["embedding"]

    def generate_with_history(self, prompt: str, history: List[Dict[str, str]], **kwargs) -> str:
        context = "\n".join([f"Human: {entry['user']}\nAI: {entry['ai']}" for entry in history])
        full_prompt = f"{context}\nHuman: {prompt}\nAI:"
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": full_prompt, "stream": False,**kwargs}
        )
        return response.json()["response"]