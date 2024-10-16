import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def create_line_graph(input_file, output_file):
    # Read the CSV file
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Calculate averages per turn for each agent
    turn_avg = {}
    for row in rows:
        agent = row['Agent']
        turn = int(row['Turn'])
        steerability = float(row['Controlled Steerability'])
        
        if agent not in turn_avg:
            turn_avg[agent] = {}
        
        if turn not in turn_avg[agent]:
            turn_avg[agent][turn] = []

        turn_avg[agent][turn].append(steerability)

    # Calculate the average for each turn
    for agent in turn_avg:
        for turn in turn_avg[agent]:
            turn_avg[agent][turn] = sum(turn_avg[agent][turn]) / len(turn_avg[agent][turn])

    # Create the line plot
    plt.figure(figsize=(12, 6))

    for agent in turn_avg:
        turns = sorted(turn_avg[agent].keys())
        averages = [turn_avg[agent][turn] for turn in turns]
        plt.plot(turns, averages, label=agent, marker='o')

    plt.title('Average Steerability Metric Scores by Turn')
    plt.xlabel('Turn')
    plt.ylabel('Average Controlled Steerability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_table_graph(input_file, output_file):
    # Read the CSV file
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Calculate averages per scenario for each agent
    scenario_avg = {}
    for row in rows:
        agent = row['Agent']
        scenario = row['Scenario Name']
        steerability = float(row['Controlled Steerability'])
        
        if agent not in scenario_avg:
            scenario_avg[agent] = {}
        
        if scenario not in scenario_avg[agent]:
            scenario_avg[agent][scenario] = []

        scenario_avg[agent][scenario].append(steerability)

    # Calculate the average for each scenario
    for agent in scenario_avg:
        for scenario in scenario_avg[agent]:
            scenario_avg[agent][scenario] = sum(scenario_avg[agent][scenario]) / len(scenario_avg[agent][scenario])

    # Prepare data for the table
    scenarios = sorted(set(scenario for agent in scenario_avg for scenario in scenario_avg[agent]))
    agents = sorted(scenario_avg.keys())

    data = [[round(scenario_avg[agent].get(scenario, 0), 3) for agent in agents] for scenario in scenarios]

    # Create the figure and axes
    fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 6), gridspec_kw={"width_ratios":[20, 1]})
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data,
                     rowLabels=scenarios,
                     colLabels=agents,
                     cellLoc='center',
                     loc='center')

    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color code the cells based on their values
    min_val = min(min(row) for row in data)
    max_val = max(max(row) for row in data)
    norm = plt.Normalize(min_val, max_val)

    for i in range(len(scenarios)):
        for j in range(len(agents)):
            cell = table[i+1, j]
            value = data[i][j]
            cell.set_facecolor(plt.cm.RdYlGn(norm(value)))

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax)
    cax.set_ylabel('Average Controlled Steerability')

    plt.suptitle('Average Controlled Steerability by Scenario and Agent', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

def create_average_line_graph(input_files, output_file):
    all_data = []
    for input_file in input_files:
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            all_data.extend(list(reader))

    turn_avg = {}
    for row in all_data:
        agent = row['Agent']
        turn = int(row['Turn'])
        steerability = float(row['Controlled Steerability'])
        
        if agent not in turn_avg:
            turn_avg[agent] = {}
        
        if turn not in turn_avg[agent]:
            turn_avg[agent][turn] = []

        turn_avg[agent][turn].append(steerability)

    for agent in turn_avg:
        for turn in turn_avg[agent]:
            turn_avg[agent][turn] = sum(turn_avg[agent][turn]) / len(turn_avg[agent][turn])

    plt.figure(figsize=(12, 6))

    for agent in turn_avg:
        turns = sorted(turn_avg[agent].keys())
        averages = [turn_avg[agent][turn] for turn in turns]
        plt.plot(turns, averages, label=agent, marker='o')

    plt.title('Average Steerability Metric Scores by Turn (All Results)')
    plt.xlabel('Turn')
    plt.ylabel('Average Controlled Steerability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_average_table_graph(input_files, output_file):
    all_data = []
    for input_file in input_files:
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            all_data.extend(list(reader))

    scenario_avg = {}
    for row in all_data:
        agent = row['Agent']
        scenario = row['Scenario Name']
        steerability = float(row['Controlled Steerability'])
        
        if agent not in scenario_avg:
            scenario_avg[agent] = {}
        
        if scenario not in scenario_avg[agent]:
            scenario_avg[agent][scenario] = []

        scenario_avg[agent][scenario].append(steerability)

    for agent in scenario_avg:
        for scenario in scenario_avg[agent]:
            scenario_avg[agent][scenario] = sum(scenario_avg[agent][scenario]) / len(scenario_avg[agent][scenario])

    scenarios = sorted(set(scenario for agent in scenario_avg for scenario in scenario_avg[agent]))
    agents = sorted(scenario_avg.keys())

    data = [[round(scenario_avg[agent].get(scenario, 0), 3) for agent in agents] for scenario in scenarios]

    fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 6), gridspec_kw={"width_ratios":[20, 1]})
    ax.axis('off')

    table = ax.table(cellText=data,
                     rowLabels=scenarios,
                     colLabels=agents,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    min_val = min(min(row) for row in data)
    max_val = max(max(row) for row in data)
    norm = plt.Normalize(min_val, max_val)

    for i in range(len(scenarios)):
        for j in range(len(agents)):
            cell = table[i+1, j]
            value = data[i][j]
            cell.set_facecolor(plt.cm.RdYlGn(norm(value)))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax)
    cax.set_ylabel('Average Controlled Steerability')

    plt.suptitle('Average Controlled Steerability by Scenario and Agent (All Results)', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()

def process_results(base_folder):
    results_folder = os.path.join(base_folder, "results")
    result_graphs_folder = os.path.join(base_folder, "result_graphs")
    
    os.makedirs(result_graphs_folder, exist_ok=True)

    all_input_files = []

    for result_subfolder in os.listdir(results_folder):
        result_path = os.path.join(results_folder, result_subfolder)
        if os.path.isdir(result_path):
            input_file = os.path.join(result_path, 'evaluation_metrics.csv')
            output_subfolder = os.path.join(result_graphs_folder, result_subfolder)
            
            if os.path.exists(input_file):
                os.makedirs(output_subfolder, exist_ok=True)
                
                line_graph_output = os.path.join(output_subfolder, 'line_graph.png')
                table_graph_output = os.path.join(output_subfolder, 'table_graph.png')
                
                create_line_graph(input_file, line_graph_output)
                create_table_graph(input_file, table_graph_output)
                print(f"Graphs generated for {result_subfolder}")

                all_input_files.append(input_file)
            else:
                print(f"No evaluation_metrics.csv found in {result_subfolder}")

    # Create average graphs
    if all_input_files:
        avg_line_graph_output = os.path.join(result_graphs_folder, 'average_line_graph.png')
        avg_table_graph_output = os.path.join(result_graphs_folder, 'average_table_graph.png')
        
        create_average_line_graph(all_input_files, avg_line_graph_output)
        create_average_table_graph(all_input_files, avg_table_graph_output)
        print("Average graphs generated in result_graphs directory")

if __name__ == "__main__":
    base_folder = ""  # Assuming the script is run from the KAAG directory
    process_results(base_folder)