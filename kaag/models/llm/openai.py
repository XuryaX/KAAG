import openai
from typing import List
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    """
    Implementation of OpenAI's GPT models.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    def embed(self, text: str) -> List[float]:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']