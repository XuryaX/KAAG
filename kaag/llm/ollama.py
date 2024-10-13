import requests
from typing import Dict, Any
from .base import LLMInterface

class OllamaLLM(LLMInterface):
    def __init__(self, model: str = "llama2", api_url: str = "http://localhost:11434"):
        self.model = model
        self.api_url = api_url

    def generate(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self.api_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()['response']

    def get_model_info(self) -> Dict[str, Any]:
        response = requests.get(f"{self.api_url}/api/show", params={"name": self.model})
        response.raise_for_status()
        return response.json()