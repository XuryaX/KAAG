# kaag/evaluation/metrics.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from typing import List, Union, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

nltk.download('punkt', quiet=True)

def calculate_context_score(generated_response: str, expected_response: str, context: Dict[str, Any]) -> float:
    # Implement more sophisticated context scoring
    short_term_score = _calculate_short_term_context(generated_response, expected_response)
    long_term_score = _calculate_long_term_context(generated_response, context)
    return 0.4 * short_term_score + 0.6 * long_term_score

def _calculate_short_term_context(generated_response: str, expected_response: str) -> float:
    # Existing implementation
    generated_blob = TextBlob(generated_response)
    expected_blob = TextBlob(expected_response)
    
    generated_vector = Counter(generated_blob.words)
    expected_vector = Counter(expected_blob.words)
    
    all_words = set(generated_vector.keys()) | set(expected_vector.keys())
    generated_vec = [generated_vector.get(word, 0) for word in all_words]
    expected_vec = [expected_vector.get(word, 0) for word in all_words]
    
    return cosine_similarity([generated_vec], [expected_vec])[0][0]

def _calculate_long_term_context(generated_response: str, context: Dict[str, Any]) -> float:
    conversation_history = context.get('conversation_history', [])
    if not conversation_history:
        return 1.0
    
    history_text = ' '.join([f"{turn[0]} {turn[1]}" for turn in conversation_history])
    vectorizer = CountVectorizer().fit([history_text, generated_response])
    history_vector = vectorizer.transform([history_text])
    response_vector = vectorizer.transform([generated_response])
    
    return cosine_similarity(history_vector, response_vector)[0][0]

def calculate_adaptation_score(previous_metrics: dict, current_metrics: dict, context: Dict[str, Any]) -> float:
    if not previous_metrics:
        return 0.0
    
    metric_changes = [abs(float(current_metrics.get(k, 0)) - float(previous_metrics.get(k, 0))) for k in current_metrics.keys()]
    metric_adaptation = np.mean(metric_changes)
    
    stage_adaptation = 1.0 if context.get('current_stage') != context.get('previous_stage') else 0.0
    
    return 0.7 * metric_adaptation + 0.3 * stage_adaptation

def calculate_inference_score(generated_response: str, expected_response: str, context: Dict[str, Any]) -> float:
    generated_sentences = sent_tokenize(generated_response)
    expected_sentences = sent_tokenize(expected_response)
    
    generated_topics = [TextBlob(sent).noun_phrases for sent in generated_sentences]
    expected_topics = [TextBlob(sent).noun_phrases for sent in expected_sentences]
    
    topic_overlap = len(set(topic for topics in generated_topics for topic in topics) & 
                        set(topic for topics in expected_topics for topic in topics))
    total_topics = len(set(topic for topics in generated_topics + expected_topics for topic in topics))
    
    inference_quality = _evaluate_inference_quality(generated_response, context)
    
    return 0.6 * (topic_overlap / total_topics if total_topics > 0 else 0) + 0.4 * inference_quality

def _evaluate_inference_quality(generated_response: str, context: Dict[str, Any]) -> float:
    # Implement logic to evaluate the quality and relevance of inferences
    # This is a placeholder implementation
    return 0.5

def calculate_steerability_score(generated_response: str, user_input: str, expected_response: str, context: Dict[str, Any]) -> float:
    user_blob = TextBlob(user_input)
    generated_blob = TextBlob(generated_response)
    expected_blob = TextBlob(expected_response)
    
    user_topics = set(user_blob.noun_phrases)
    generated_topics = set(generated_blob.noun_phrases)
    expected_topics = set(expected_blob.noun_phrases)
    
    generated_alignment = len(user_topics & generated_topics) / len(user_topics) if user_topics else 0
    expected_alignment = len(user_topics & expected_topics) / len(user_topics) if user_topics else 0
    
    topic_steerability = 1 - abs(generated_alignment - expected_alignment)
    
    conversation_flow = _evaluate_conversation_flow(generated_response, context)
    
    return 0.6 * topic_steerability + 0.4 * conversation_flow

def _evaluate_conversation_flow(generated_response: str, context: Dict[str, Any]) -> float:
    # Implement logic to evaluate how well the model guides the conversation
    # This is a placeholder implementation
    return 0.5

def calculate_knowledge_integration_score(generated_response: str, context: Dict[str, Any]) -> float:
    integrated_knowledge = context.get('integrated_knowledge', '')
    vectorizer = CountVectorizer().fit([integrated_knowledge, generated_response])
    knowledge_vector = vectorizer.transform([integrated_knowledge])
    response_vector = vectorizer.transform([generated_response])
    
    similarity = cosine_similarity(knowledge_vector, response_vector)[0][0]
    
    relevance = _evaluate_knowledge_relevance(generated_response, context)
    
    return 0.7 * similarity + 0.3 * relevance

def _evaluate_knowledge_relevance(generated_response: str, context: Dict[str, Any]) -> float:
    # Implement logic to evaluate the relevance of integrated knowledge
    # This is a placeholder implementation
    return 0.5

# Update the EvaluationFramework to use these new metrics