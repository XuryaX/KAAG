from typing import Dict, Any
from ..llm.base import LLMInterface
from jinja2 import Template

class NoRAG:
    def __init__(self, llm: LLMInterface, config: Dict[str, Any], template: Template):
        self.llm = llm
        self.config = config
        self.template = template

    def process_turn(self, user_input: str, conversation_history: str = "") -> str:
        # Generate response using LLM with the template
        prompt = self.template.render(
            persona=self.config['persona'],
            conversation_history=conversation_history,
            user_message=user_input
        )
        response = self.llm.generate(prompt)

        return response