from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_sentiment, get_noun_phrases
import numpy as np

class TimeAvailabilityAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.availability_keywords = set(['available', 'free', 'schedule', 'time', 'meet'])

    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_availability = current_state.get('time_availability', 60)
        
        sentiment = get_sentiment(last_message)
        noun_phrases = set(get_noun_phrases(last_message))
        
        availability_relevance = len(self.availability_keywords.intersection(noun_phrases)) / len(self.availability_keywords)
        
        # Positive sentiment with availability terms increases availability
        availability_change = availability_relevance * (sentiment + 1) * 10
        
        new_availability = np.clip(current_availability + availability_change, 0, 100)
        return 'time_availability', new_availability