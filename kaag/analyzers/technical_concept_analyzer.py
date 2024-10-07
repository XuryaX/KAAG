from .base_analyzer import BaseAnalyzer
from typing import Dict, Any
import re

class TechnicalConceptAnalyzer(BaseAnalyzer):
    def __init__(self, technical_concepts: Dict[str, str]):
        self.technical_concepts = technical_concepts

    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        combined_text = user_input + " " + ai_response
        concept_scores = {}

        for concept, definition in self.technical_concepts.items():
            concept_score = self._calculate_concept_score(combined_text, concept, definition)
            concept_scores[f"{concept}_understanding"] = concept_score

        return concept_scores

    def _calculate_concept_score(self, text: str, concept: str, definition: str) -> float:
        concept_regex = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
        concept_mentions = len(concept_regex.findall(text))

        definition_keywords = set(definition.lower().split())
        text_words = set(text.lower().split())
        keyword_overlap = len(definition_keywords & text_words)

        if concept_mentions == 0:
            return 0.0

        return min(1.0, (concept_mentions * keyword_overlap) / (len(definition_keywords) * 2))