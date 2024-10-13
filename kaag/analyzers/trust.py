from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_sentiment, get_subjectivity
import numpy as np

class TrustAnalyzer(BaseAnalyzer):
    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_trust = current_state.get('trust', 50)
        
        sentiment = get_sentiment(last_message)
        subjectivity = get_subjectivity(last_message)
        
        # Trust is higher when sentiment is positive and subjectivity is low
        trust_change = sentiment * (1 - subjectivity) * 10
        
        new_trust = np.clip(current_trust + trust_change, 0, 100)
        return 'trust', new_trust