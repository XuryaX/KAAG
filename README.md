# KAAG: Knowledge and Aptitude Augmented Generation

KAAG is a framework for creating adaptive AI agents that can engage in dynamic, context-aware conversations. It integrates knowledge retrieval, aptitude modeling, and large language models to provide a flexible and powerful system for simulating complex interactions.

## Features

- Dynamic Bayesian Network for modeling conversation states
- Integration with various LLM providers (OpenAI, Anthropic, Ollama)
- Customizable analyzers for sentiment, topic coherence, and more
- Flexible configuration system for defining conversation stages and metrics
- Simulation capabilities for testing and evaluating AI agents

## Installation

```bash
pip install kaag
```

## Quick Start

```python
from kaag import (
    MetricsManager, StageManager, DynamicBayesianNetwork,
    OpenAILLM, SentimentAnalyzer, TopicAnalyzer, CustomMetricAnalyzer,
    Simulator
)
from kaag.utils.config import load_config

# Load configuration
config = load_config("sales_client_config.yaml")

# Initialize components
metrics_manager = MetricsManager(config['metrics'])
stage_manager = StageManager(config['stages'])
dbn = DynamicBayesianNetwork(metrics_manager, stage_manager)

llm = OpenAILLM(api_key="your_api_key_here", **config['llm']['config'])

analyzers = [
    SentimentAnalyzer(),
    TopicAnalyzer(),
    CustomMetricAnalyzer(config['custom_metrics'])
]

simulator = Simulator(dbn, llm, config, analyzers)

# Run a simulation
response = simulator.run_interaction("Hello, I'm interested in your product.")
print(response)
```

## Documentation

For full documentation, visit [docs.kaag.io](https://docs.kaag.io).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.