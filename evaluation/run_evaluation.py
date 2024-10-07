import os
import sys
import csv
from typing import List, Dict
import nltk
from textblob import download_corpora

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaag.core.metrics import MetricsManager
from kaag.core.stages import StageManager
from kaag.core.dbn import DynamicBayesianNetwork
from kaag.models.llm.ollama import OllamaLLM
from kaag.analyzers.sentiment_analyzer import SentimentAnalyzer
from kaag.analyzers.topic_analyzer import TopicAnalyzer
from kaag.analyzers.custom_metric_analyzer import CustomMetricAnalyzer
from kaag.utils.config import load_config
from framework import EvaluationFramework


def load_transcript(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as f:
        return [{'speaker': line.split(':')[0].strip(), 'utterance': ':'.join(line.split(':')[1:]).strip()} for line in f]

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt')

def ensure_textblob_corpora():
    try:
        from textblob import TextBlob
        TextBlob("Test").words
    except textblob.exceptions.MissingCorpusError:
        print("Downloading required TextBlob corpora...")
        download_corpora()

def run_evaluation():
    try:
        # Ensure required data is available
        ensure_nltk_data()
        ensure_textblob_corpora()

        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'sales_client_config.yaml')
        config = load_config(config_path)

        # Load transcript
        transcript_path = os.path.join(os.path.dirname(__file__), '..', 'conversation_transcript.txt')
        transcript = load_transcript(transcript_path)

        # Initialize components
        metrics_manager = MetricsManager(config['metrics'])
        stage_manager = StageManager(config['stages'])
        dbn = DynamicBayesianNetwork(metrics_manager, stage_manager)
        llm = OllamaLLM()

        # Define analyzers for each approach
        base_analyzers = []
        rag_analyzers = [SentimentAnalyzer(), TopicAnalyzer()]
        kaag_analyzers = [SentimentAnalyzer(), TopicAnalyzer(), CustomMetricAnalyzer(config['custom_metrics'])]

        # Create evaluation frameworks
        base_framework = EvaluationFramework(dbn, llm, config, base_analyzers)
        rag_framework = EvaluationFramework(dbn, llm, config, rag_analyzers)
        kaag_framework = EvaluationFramework(dbn, llm, config, kaag_analyzers)

       # Run evaluations
        base_results = base_framework.evaluate_conversation(transcript)
        rag_results = rag_framework.evaluate_conversation(transcript)
        kaag_results = kaag_framework.evaluate_conversation(transcript)

        # Save results
        results_path = os.path.join(os.path.dirname(__file__), '..', 'evaluation_results.csv')
        with open(results_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Turn', 'Metric', 'Base LLM', 'RAG Enhanced', 'KAAG Enhanced'])
            for turn, (base, rag, kaag) in enumerate(zip(base_results, rag_results, kaag_results), 1):
                for metric in base.keys():
                    if metric != 'generated_output':
                        writer.writerow([turn, metric, base[metric], rag[metric], kaag[metric]])

        # Generate file for post-run manual evaluation
        manual_eval_path = os.path.join(os.path.dirname(__file__), '..', 'manual_evaluation.txt')
        with open(manual_eval_path, 'w') as f:
            for i, (base, rag, kaag) in enumerate(zip(base_results, rag_results, kaag_results)):
                user_input = transcript[i*2]['utterance']
                expected_output = transcript[i*2+1]['utterance'] if i*2+1 < len(transcript) else ""
                f.write(f"Turn {i+1}\n")
                f.write(f"User Input: {user_input}\n")
                f.write(f"Expected Output: {expected_output}\n")
                f.write(f"Base LLM Output: {base['generated_output']}\n")
                f.write(f"RAG Enhanced Output: {rag['generated_output']}\n")
                f.write(f"KAAG Enhanced Output: {kaag['generated_output']}\n")
                f.write("\n---\n\n")

        print(f"Evaluation complete. Results saved in {results_path} and {manual_eval_path}")

    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()