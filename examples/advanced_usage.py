from kaag import (
    MetricsManager, StageManager, DynamicBayesianNetwork,
    OpenAILLM, SentimentAnalyzer, TopicAnalyzer, CustomMetricAnalyzer,
    Simulator
)
from kaag.utils.config import load_config

def create_simulator_from_config(config_path: str):
    config = load_config(config_path)
    
    metrics_manager = MetricsManager(config['metrics'])
    stage_manager = StageManager(config['stages'])
    dbn = DynamicBayesianNetwork(metrics_manager, stage_manager)
    
    llm = OpenAILLM(api_key=config['llm']['api_key'], **config['llm']['config'])
    
    analyzers = [
        SentimentAnalyzer(),
        TopicAnalyzer(),
        CustomMetricAnalyzer(config['custom_metrics'])
    ]
    
    return Simulator(dbn, llm, config, analyzers)

def run_sales_call_simulation(config_path: str):
    simulator = create_simulator_from_config(config_path)

    print("Sales Call Simulation Started")
    print("You are a salesperson pitching Learning & Development services to an AI-powered tech startup client.")
    print("Type 'exit' to end the simulation.\n")

    while True:
        user_input = input("Salesperson: ")
        if user_input.lower() == 'exit':
            break

        response = simulator.run_interaction(user_input)
        print(f"AI Client: {response}")

        if "Conversation ended" in response:
            break

    print("Simulation ended.")

if __name__ == "__main__":
    run_sales_call_simulation("sales_client_config.yaml")