import json
import os
import csv
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from kaag.core.kaag import KAAG
from kaag.core.rag import RAG
from kaag.core.norag import NoRAG
from kaag.llm.ollama import OllamaLLM
from kaag.knowledge_retriever.text_file import TextFileKnowledgeRetriever
from kaag.utils.config import load_config

from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from jinja2 import Environment, FileSystemLoader

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class SteerabilityMetric(BaseMetric):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        """
        This method calculates steerability by checking if the AI's responses
        are aligning with the user's conversation goal.
        """
        try:
            user_input = test_case.input.lower()
            generated_response = test_case.actual_output.lower()
            expected_goal = test_case.expected_output.lower()

            # Example basic steerability logic: Checking for keyword matching
            steerability_score = 0
            if expected_goal in generated_response:
                steerability_score = 1.0  # AI successfully steered towards goal

            self.score = steerability_score
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    @property
    def __name__(self):
        return "Steerability Metric"


def load_test_cases(file_path: str) -> Dict[str, Any]:
    print(f"Loading test cases from {file_path}...")
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_response(generated_response: str, expected_response: str, user_input: str) -> Dict[str, float]:
    print(f"Evaluating response:\nGenerated: {generated_response}\nExpected: {expected_response}")
    
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_response,
        expected_output=expected_response
    )
    
    # Evaluate with various metrics including steerability
    relevancy_metric = AnswerRelevancyMetric()
    hallucination_metric = HallucinationMetric()
    steerability_metric = SteerabilityMetric()

    results = {}
    results['relevancy'] = relevancy_metric.measure(test_case)
    results['hallucination'] = hallucination_metric.measure(test_case)
    results['steerability'] = steerability_metric.measure(test_case)

    return results

def process_turn(agent, user_input: str, expected_response: str) -> Dict[str, Any]:
    print(f"Processing turn for agent {agent.__class__.__name__} with input: {user_input}")
    response = agent.process_turn(user_input)
    evaluation = evaluate_response(response, expected_response, user_input)
    print(f"Completed processing turn for {agent.__class__.__name__}. Response: {response}, Evaluations: {evaluation}")
    return {
        'response': response,
        'evaluation': evaluation
    }

def run_scenario(kaag: KAAG, rag: RAG, norag: NoRAG, scenario: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Running scenario: {scenario['name']}")
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        for i, turn in enumerate(scenario['turns'], 1):
            user_input = turn['user_input']
            expected_response = turn['expected_ai_response']
            
            print(f"Processing turn {i} with input: {user_input}")
            futures = {
                executor.submit(process_turn, agent, user_input, expected_response): agent_name
                for agent_name, agent in [('KAAG', kaag), ('RAG', rag), ('NoRAG', norag)]
            }
            
            turn_results = {
                'user_input': user_input,
                'expected_response': expected_response,
                'agent_responses': {}
            }
            
            for future in as_completed(futures):
                agent_name = futures[future]
                print(f"Agent {agent_name} has completed turn {i}")
                turn_results['agent_responses'][agent_name] = future.result()
            
            results.append(turn_results)
    
    return results

def save_results(scenario_name: str, results: List[Dict[str, Any]], output_dir: str, csv_file: str, scenario_count: int):
    print(f"Saving results for scenario: {scenario_name} to {output_dir}")
    
    # Save detailed output
    detailed_output_file = os.path.join(output_dir, 'detailed_output.txt')
    os.makedirs(output_dir, exist_ok=True)

    with open(detailed_output_file, 'a') as f:
        f.write(f"Test Case: {scenario_name}\n\n")
        for i, turn in enumerate(results, 1):
            f.write(f"Turn {i}\n")
            f.write(f"User Input: {turn['user_input']}\n")
            f.write(f"Expected Response: {turn['expected_response']}\n")
            for agent, response in turn['agent_responses'].items():
                f.write(f"{agent} Response: {response['response']}\n")
                for metric, score in response['scores'].items():
                    f.write(f"{agent} {metric}: {score:.2f}\n")
            f.write("\n")
    
    # Save CSV metrics
    csv_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['Scenario Name', 'Turn', 'User Input', 'Expected Response', 'Agent', 
                      'Contextual Relevance', 'Adaptation Effectiveness', 'Convergence Rate', 
                      'Probabilistic Inference Accuracy', 'Controlled Steerability', 'User Satisfaction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not csv_exists:  # Write headers only if file doesn't exist
            writer.writeheader()

        for i, turn in enumerate(results, 1):
            for agent, response in turn['agent_responses'].items():
                writer.writerow({
                    'Scenario Name': scenario_name,
                    'Turn': i,
                    'User Input': turn['user_input'],
                    'Expected Response': turn['expected_response'],
                    'Agent': agent,
                    'Contextual Relevance': response['scores']['contextual_relevance'],
                    'Adaptation Effectiveness': response['scores']['adaptation_effectiveness'],
                    'Convergence Rate': response['scores']['convergence_rate'],
                    'Probabilistic Inference Accuracy': response['scores']['probabilistic_inference_accuracy'],
                    'Controlled Steerability': response['scores']['controlled_steerability'],
                    'User Satisfaction': response['scores']['user_satisfaction']
                })

        # Append average scores per scenario
        avg_scores = {agent: {metric: sum(turn['agent_responses'][agent]['scores'][metric] 
                          for turn in results) / len(results) 
                          for metric in ['contextual_relevance', 'adaptation_effectiveness', 'convergence_rate', 
                                         'probabilistic_inference_accuracy', 'controlled_steerability', 
                                         'user_satisfaction']}
                      for agent in ['KAAG', 'RAG', 'NoRAG']}
        
        for agent, scores in avg_scores.items():
            writer.writerow({
                'Scenario Name': scenario_name,
                'Turn': 'Average',
                'User Input': '',
                'Expected Response': '',
                'Agent': agent,
                'Contextual Relevance': scores['contextual_relevance'],
                'Adaptation Effectiveness': scores['adaptation_effectiveness'],
                'Convergence Rate': scores['convergence_rate'],
                'Probabilistic Inference Accuracy': scores['probabilistic_inference_accuracy'],
                'Controlled Steerability': scores['controlled_steerability'],
                'User Satisfaction': scores['user_satisfaction']
            })
    
    if scenario_count == 1:  # Calculate total average when processing the last scenario
        print(f"Calculating total averages across all scenarios...")
        calculate_total_averages(csv_file)

def calculate_total_averages(csv_file: str):
    # Load CSV data into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Filter out rows with 'Turn' marked as 'Average' or actual turns for agents
    filtered_df = df[df['Turn'] != 'Average']
    
    # Calculate average scores across all agents and turns
    total_avg_scores = filtered_df.groupby('Agent').mean()
    
    # Append the total averages to the CSV file
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['Scenario Name', 'Turn', 'User Input', 'Expected Response', 'Agent', 
                      'Contextual Relevance', 'Adaptation Effectiveness', 'Convergence Rate', 
                      'Probabilistic Inference Accuracy', 'Controlled Steerability', 'User Satisfaction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for agent, row in total_avg_scores.iterrows():
            writer.writerow({
                'Scenario Name': 'Total Average',
                'Turn': '',
                'User Input': '',
                'Expected Response': '',
                'Agent': agent,
                'Contextual Relevance': row['Contextual Relevance'],
                'Adaptation Effectiveness': row['Adaptation Effectiveness'],
                'Convergence Rate': row['Convergence Rate'],
                'Probabilistic Inference Accuracy': row['Probabilistic Inference Accuracy'],
                'Controlled Steerability': row['Controlled Steerability'],
                'User Satisfaction': row['User Satisfaction']
            })

def main():
    print("Starting main execution...")
    
    # Update paths to be relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    config_path = os.path.join(project_root, 'evaluation_data', 'sales_client_config.yaml')
    test_cases_path = os.path.join(project_root, 'evaluation_data', 'test_cases.json')
    knowledge_path = os.path.join(project_root, 'evaluation_data', 'knowledge.txt')
    template_dir = os.path.join(project_root, 'evaluation_data', 'prompt_templates')
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    # Load test cases
    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_path)

    # Initialize LLM
    print("Initializing LLM...")
    llm = OllamaLLM(model=config['llm']['model'], api_url=config['llm']['api_url'])

    # Load templates
    print("Loading templates...")
    env = Environment(loader=FileSystemLoader(template_dir))
    kaag_template = env.get_template('kaag.jinja')
    rag_template = env.get_template('rag.jinja')
    norag_template = env.get_template('norag.jinja')

    # Initialize knowledge retriever
    print("Initializing knowledge retriever...")
    knowledge_retriever = TextFileKnowledgeRetriever(knowledge_path, top_k=config['knowledge_retriever']['top_k'])

    # Initialize KAAG, RAG, and NoRAG
    print("Initializing KAAG, RAG, and NoRAG...")
    kaag = KAAG(llm, config, kaag_template)
    rag = RAG(llm, config, knowledge_retriever, rag_template)
    norag = NoRAG(llm, config, norag_template)

    # Run evaluations
    print("Starting scenario evaluations...")
    output_dir = 'evaluation_results'
    csv_file = os.path.join(output_dir, 'evaluation_metrics.csv')
    
    scenario_count = len(test_cases['test_scenarios'])
    
    for i, scenario in enumerate(test_cases['test_scenarios'], 1):
        print(f"Evaluating scenario: {scenario['name']}")
        results = run_scenario(kaag, rag, norag, scenario)
        save_results(scenario['name'], results, output_dir, csv_file, scenario_count - i)

if __name__ == "__main__":
    main()
