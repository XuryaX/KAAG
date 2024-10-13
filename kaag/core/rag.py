from typing import Dict, Any
from ..llm.base import LLMInterface
from ..knowledge_retriever.base import BaseKnowledgeRetriever
from jinja2 import Template

class RAG:
    def __init__(self, llm: LLMInterface, config: Dict[str, Any], knowledge_retriever: BaseKnowledgeRetriever, template: Template):
        self.llm = llm
        self.config = config
        self.knowledge_retriever = knowledge_retriever
        self.template = template

    def process_turn(self, user_input: str, conversation_history: str = "") -> str:
        # Retrieve relevant information
        retrieved_information = self.knowledge_retriever.retrieve(user_input)

        # Generate response using LLM with the template
        prompt = self.template.render(
            persona=self.config['persona'],
            retrieved_information=retrieved_information,
            conversation_history=conversation_history,
            user_message=user_input
        )
        response = self.llm.generate(prompt)

        return response