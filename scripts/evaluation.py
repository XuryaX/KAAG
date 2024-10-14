import json
import os
import csv
from typing import Dict, Any, List, NamedTuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from kaag.core.kaag import KAAG
from kaag.core.rag import RAG
from kaag.core.norag import NoRAG
from kaag.llm.ollama import OllamaLLM
from kaag.knowledge_retriever.text_file import TextFileKnowledgeRetriever
from kaag.utils.config import load_config

class Turn(NamedTuple):
    user_input: str
    generated_output: str
    expected_output: str
    metrics: Dict[str, float]

class Metric:
    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold

    def calculate(self, generated_output: str, expected_output: str) -> float:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([generated_output, expected_output])
        return cosine_similarity(vectors[0], vectors[1])[0][0]

class ControlledSteerabilityMetric(Metric):
    def __init__(self, threshold=0.7):
        super().__init__("Controlled Steerability", threshold)

class ConvergenceRateMetric(Metric):
    def __init__(self, threshold=0.7, convergence_threshold=0.9):
        super().__init__("Convergence Rate", threshold)
        self.convergence_threshold = convergence_threshold
        self.cumulative_generated = ""
        self.cumulative_expected = ""
        self.converged_turn = None

    def calculate(self, generated_output: str, expected_output: str) -> float:
        self.cumulative_generated += " " + generated_output
        self.cumulative_expected += " " + expected_output
        
        similarity = super().calculate(self.cumulative_generated, self.cumulative_expected)
        
        if similarity >= self.convergence_threshold and self.converged_turn is None:
            self.converged_turn = len(self.cumulative_generated.split())
        
        return 1 / self.converged_turn if self.converged_turn else 0.0

def load_test_cases(file_path: str) -> Dict[str, Any]:
    print(f"Loading test cases from {file_path}...")
    with open(file_path, 'r') as f:
        return json.load(f)

def process_turn(agent, user_input: str, expected_output: str, metrics: List[Metric]) -> Turn:
    print(f"Processing turn for agent {agent.__class__.__name__} with input: {user_input}")
    generated_output = agent.process_turn(user_input)
    turn_metrics = {metric.name: metric.calculate(generated_output, expected_output) for metric in metrics}
    return Turn(user_input, generated_output, expected_output, turn_metrics)

def run_scenario(kaag: KAAG, rag: RAG, norag: NoRAG, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    print(f"Running scenario: {scenario['name']}")
    results = []
    metrics = [ControlledSteerabilityMetric(), ConvergenceRateMetric()]
    static_analyzer = kaag.get_analyzers()[0]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        for turn in scenario['turns']:
            user_input = turn['user_input']
            expected_output = turn['expected_ai_response']
            static_analyzer.update_properties(turn['metrics'])
            
            futures = {
                executor.submit(process_turn, agent, user_input, expected_output, metrics): name
                for name, agent in [('KAAG', kaag), ('RAG', rag), ('NoRAG', norag)]
            }
            
            turn_results = {
                'user_input': user_input,
                'expected_output': expected_output,
                'agent_responses': {}
            }
            
            for future in as_completed(futures):
                agent_name = futures[future]
                turn_data = future.result()
                turn_results['agent_responses'][agent_name] = turn_data
                
                if agent_name == 'KAAG':
                    turn_results['kaag_state'] = kaag.get_current_state()
            
            results.append(turn_results)
    
    return results

def save_results(scenario_name: str, results: List[Dict[str, Any]], output_dir: str, csv_file: str):

    # Custom handler for non-serializable types
    def default(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


    print(f"Saving results for scenario: {scenario_name} to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    detailed_output_file = os.path.join(output_dir, f"{scenario_name}_detailed_output.txt")

    with open(detailed_output_file, 'w') as f:
        f.write(f"Test Case: {scenario_name}\n\n")
        for i, turn in enumerate(results, 1):
            f.write(f"Turn {i}\n")
            f.write(f"User Input: {turn['user_input']}\n")
            f.write(f"Expected Output: {turn['expected_output']}\n")
            for agent, response_data in turn['agent_responses'].items():
                f.write(f"{agent} Output: {response_data.generated_output}\n")
                for metric, score in response_data.metrics.items():
                    f.write(f"{agent} {metric}: {score:.2f}\n")
            if 'kaag_state' in turn:
                f.write("KAAG State:\n")
                f.write(json.dumps(turn['kaag_state'], indent=2, default=default))
            f.write("\n")
    
    csv_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['Scenario Name', 'Turn', 'User Input', 'Expected Output', 'Agent', 'Generated Output', 'Controlled Steerability', 'Convergence Rate', 'KAAG Current Node', 'KAAG Metrics']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not csv_exists:
            writer.writeheader()

        for i, turn in enumerate(results, 1):
            for agent, response_data in turn['agent_responses'].items():
                row = {
                    'Scenario Name': scenario_name,
                    'Turn': i,
                    'User Input': turn['user_input'],
                    'Expected Output': turn['expected_output'],
                    'Agent': agent,
                    'Generated Output': response_data.generated_output,
                    'Controlled Steerability': response_data.metrics['Controlled Steerability'],
                    'Convergence Rate': response_data.metrics['Convergence Rate'],
                    'KAAG Current Node': '',
                    'KAAG Metrics': ''
                }
                if agent == 'KAAG' and 'kaag_state' in turn:
                    row['KAAG Current Node'] = turn['kaag_state']['current_node']
                    row['KAAG Metrics'] = json.dumps(turn['kaag_state']['interaction_state'], indent=2, default=default)
                writer.writerow(row)

def calculate_total_averages(csv_file: str):
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    total_avg_scores = df.groupby('Agent')[['Controlled Steerability', 'Convergence Rate']].mean()
    
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])
        writer.writerow(['Total Averages'])
        writer.writerow(['Agent', 'Controlled Steerability', 'Convergence Rate'])
        for agent, row in total_avg_scores.iterrows():
            writer.writerow([agent, row['Controlled Steerability'], row['Convergence Rate']])

def main():
    print("Starting main execution...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    config_path = os.path.join(project_root, 'evaluation_data', 'sales_client_config.yaml')
    test_cases_path = os.path.join(project_root, 'evaluation_data', 'test_cases.json')
    knowledge_path = os.path.join(project_root, 'evaluation_data', 'knowledge.txt')
    template_dir = os.path.join(project_root, 'evaluation_data', 'prompt_templates')
    
    print("Loading configuration...")
    config = load_config(config_path)

    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_path)

    print("Initializing LLM...")
    llm = OllamaLLM(model=config['llm']['model'], api_url=config['llm']['api_url'])

    print("Loading templates...")
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(template_dir))
    kaag_template = env.get_template('kaag.jinja')
    rag_template = env.get_template('rag.jinja')
    norag_template = env.get_template('norag.jinja')

    print("Initializing knowledge retriever...")
    knowledge_retriever = TextFileKnowledgeRetriever(knowledge_path, top_k=config['knowledge_retriever']['top_k'])

    print("Initializing KAAG, RAG, and NoRAG...")
    kaag = KAAG(llm, config, kaag_template)
    rag = RAG(llm, config, knowledge_retriever, rag_template)
    norag = NoRAG(llm, config, norag_template)

    print("Starting scenario evaluations...")
    output_dir = 'evaluation_results'
    csv_file = os.path.join(output_dir, 'evaluation_metrics.csv')
    
    for scenario in test_cases['test_scenarios']:
        print(f"Evaluating scenario: {scenario['name']}")
        results = run_scenario(kaag, rag, norag, scenario)
        save_results(scenario['name'], results, output_dir, csv_file)

    calculate_total_averages(csv_file)
    print("Evaluation complete. Results saved to:", output_dir)

if __name__ == "__main__":
    main()