from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_sentiment
import numpy as np

class SentimentAnalyzer(BaseAnalyzer):
    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_sentiment = current_state.get('sentiment', 50)
        
        new_sentiment = get_sentiment(last_message)
        
        # Convert sentiment from [-1, 1] to [0, 100] and blend with current sentiment
        sentiment_score = (new_sentiment + 1) * 50
        blended_sentiment = (current_sentiment + sentiment_score) / 2
        return 'sentiment', blended_sentiment