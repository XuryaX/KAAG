import os
from kaag import (
    MetricsManager, DynamicBayesianNetwork, Stage,
    OpenAILLM, VectorDBRetriever, Simulator
)
from kaag.utils.config import load_config

def create_dbn_from_config(config):
    metrics_manager = MetricsManager(config['metrics'])
    stages = [Stage(s['id'], s['conditions'], s['instructions']) for s in config['stages']]
    return DynamicBayesianNetwork(stages, metrics_manager)

def run_sales_call_simulation():
    config = load_config("sales_client_config.yaml")
    
    dbn = create_dbn_from_config(config)
    llm = OpenAILLM(api_key=os.environ.get("OPENAI_API_KEY"), **config["llm"]["config"])
    
    simulator = Simulator(dbn, llm, config)

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
    run_sales_call_simulation()