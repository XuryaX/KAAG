import unittest
from unittest.mock import Mock, patch
from kaag.models.llm.base import BaseLLM
from kaag.models.llm.openai import OpenAILLM
from kaag.models.llm.anthropic import AnthropicLLM
from kaag.models.llm.ollama import OllamaLLM
from kaag.models.retriever.vector_db import VectorDBRetriever
from kaag.models.retriever.sql_db import SQLDBRetriever

class TestLLMModels(unittest.TestCase):
    def test_openai_llm(self):
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value.choices[0].message.content = "Test response"
            llm = OpenAILLM("fake_api_key")
            response = llm.generate("Test prompt")
            self.assertEqual(response, "Test response")

    def test_anthropic_llm(self):
        with patch('anthropic.Client') as mock_client:
            mock_client.return_value.completion.return_value.completion = "Test response"
            llm = AnthropicLLM("fake_api_key")
            response = llm.generate("Test prompt")
            self.assertEqual(response, "Test response")

    def test_ollama_llm(self):
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {"response": "Test response"}
            llm = OllamaLLM()
            response = llm.generate("Test prompt")
            self.assertEqual(response, "Test response")

class TestRetrievers(unittest.TestCase):
    def test_vector_db_retriever(self):
        mock_vector_db = Mock()
        mock_vector_db.search.return_value = ["Result 1", "Result 2"]
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [0.1, 0.2, 0.3]
        
        retriever = VectorDBRetriever(mock_vector_db, mock_embedding_model)
        result = retriever.retrieve("Test query")
        self.assertEqual(result, {"results": ["Result 1", "Result 2"]})

    def test_sql_db_retriever(self):
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            mock_connection = Mock()
            mock_connection.execute.return_value = [{"id": 1, "name": "Test"}]
            mock_create_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
            
            retriever = SQLDBRetriever("fake_connection_string")
            result = retriever.retrieve("SELECT * FROM test")
            self.assertEqual(result, {"results": [{"id": 1, "name": "Test"}]})

if __name__ == '__main__':
    unittest.main()