from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_sentiment, get_entities
import numpy as np

class ComplianceConfidenceAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.compliance_keywords = set(['compliant', 'regulation', 'law', 'policy', 'standard'])

    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_confidence = current_state.get('compliance_confidence', 40)
        
        sentiment = get_sentiment(last_message)
        entities = set(get_entities(last_message))
        
        compliance_relevance = len(self.compliance_keywords.intersection(entities)) / len(self.compliance_keywords)
        
        # Positive sentiment with compliance terms increases confidence
        confidence_change = compliance_relevance * (sentiment + 1) * 15
        
        new_confidence = np.clip(current_confidence + confidence_change, 0, 100)
        return 'compliance_confidence', new_confidence