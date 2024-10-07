from typing import Dict, Any
import requests

class KnowledgeRetriever:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def retrieve(self, query: str) -> Dict[str, Any]:
        response = requests.get(f"{self.api_url}/search", params={"q": query})
        return response.json()