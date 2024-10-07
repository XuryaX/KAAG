# kaag/evaluation/metrics.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

nltk.download('punkt', quiet=True)

def calculate_context_score(generated_response: str, expected_response: str) -> float:
    generated_blob = TextBlob(generated_response)
    expected_blob = TextBlob(expected_response)
    
    generated_vector = Counter(generated_blob.words)
    expected_vector = Counter(expected_blob.words)
    
    all_words = set(generated_vector.keys()) | set(expected_vector.keys())
    generated_vec = [generated_vector.get(word, 0) for word in all_words]
    expected_vec = [expected_vector.get(word, 0) for word in all_words]
    
    return cosine_similarity([generated_vec], [expected_vec])[0][0]

def calculate_adaptation_score(previous_metrics: dict, current_metrics: dict) -> float:
    if not previous_metrics:
        return 0.0
    
    changes = [abs(current_metrics.get(k, 0) - previous_metrics.get(k, 0)) for k in current_metrics.keys()]
    return np.mean(changes)

def calculate_inference_score(generated_response: str, expected_response: str) -> float:
    generated_sentences = sent_tokenize(generated_response)
    expected_sentences = sent_tokenize(expected_response)
    
    generated_topics = [TextBlob(sent).noun_phrases for sent in generated_sentences]
    expected_topics = [TextBlob(sent).noun_phrases for sent in expected_sentences]
    
    topic_overlap = len(set(topic for topics in generated_topics for topic in topics) & 
                        set(topic for topics in expected_topics for topic in topics))
    total_topics = len(set(topic for topics in generated_topics + expected_topics for topic in topics))
    
    return topic_overlap / total_topics if total_topics > 0 else 0

def calculate_steerability_score(generated_response: str, user_input: str, expected_response: str) -> float:
    user_blob = TextBlob(user_input)
    generated_blob = TextBlob(generated_response)
    expected_blob = TextBlob(expected_response)
    
    user_topics = set(user_blob.noun_phrases)
    generated_topics = set(generated_blob.noun_phrases)
    expected_topics = set(expected_blob.noun_phrases)
    
    generated_alignment = len(user_topics & generated_topics) / len(user_topics) if user_topics else 0
    expected_alignment = len(user_topics & expected_topics) / len(user_topics) if user_topics else 0
    
    return 1 - abs(generated_alignment - expected_alignment)

def calculate_satisfaction_score(generated_responses: List[str], expected_responses: List[str]) -> float:
    generated_sentiments = [TextBlob(resp).sentiment.polarity for resp in generated_responses]
    expected_sentiments = [TextBlob(resp).sentiment.polarity for resp in expected_responses]
    
    sentiment_diff = np.mean([abs(g - e) for g, e in zip(generated_sentiments, expected_sentiments)])
    return 1 - sentiment_diff

def calculate_coherence_score(response: str) -> float:
    sentences = sent_tokenize(response)
    if len(sentences) < 2:
        return 1.0
    
    coherence_scores = []
    for i in range(len(sentences) - 1):
        similarity = cosine_similarity(
            [TextBlob(sentences[i]).words],
            [TextBlob(sentences[i+1]).words]
        )[0][0]
        coherence_scores.append(similarity)
    
    return np.mean(coherence_scores)

def calculate_information_density_score(response: str) -> float:
    words = TextBlob(response).words
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0