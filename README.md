# KAAG: Knowledge and Aptitude Augmented Generation

KAAG is a framework for creating adaptive AI agents that engage in dynamic, context-aware conversations. It integrates knowledge retrieval, aptitude modeling, and large language models to provide a flexible system for complex interactions.

[![PyPI version](https://badge.fury.io/py/kaag.svg)](https://badge.fury.io/py/kaag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Dynamic Bayesian Networks (DBNs) for modeling conversation states
- Gamified Interaction Model (GIM) for managing interaction context
- Integration with various LLM providers (OpenAI, Anthropic, Ollama)
- Customizable analyzers for interaction state analysis
- Flexible configuration system for conversation stages and metrics
- Simulation capabilities for testing and evaluating AI agents

## Installation

```bash
pip install kaag
```

For development:

```bash
git clone https://github.com/aroundAI/kaag.git
cd kaag
pip install -e ".[dev]"
```

## Quick Start

```python
from kaag import KAAG, RAG, NoRAG
from kaag.llm.ollama import OllamaLLM
from kaag.knowledge_retriever.text_file import TextFileKnowledgeRetriever
from kaag.utils.config import load_config
from jinja2 import Environment, FileSystemLoader

# Load configuration and initialize components
config = load_config("config.yaml")
llm = OllamaLLM(model="llama2", api_url="http://localhost:11434")
knowledge_retriever = TextFileKnowledgeRetriever("knowledge.txt", top_k=3)

# Load templates
env = Environment(loader=FileSystemLoader("templates"))
kaag_template = env.get_template('kaag.jinja')
rag_template = env.get_template('rag.jinja')
norag_template = env.get_template('norag.jinja')

# Initialize agents
kaag_agent = KAAG(llm, config, kaag_template)
rag_agent = RAG(llm, config, knowledge_retriever, rag_template)
norag_agent = NoRAG(llm, config, norag_template)

# Process user input
user_input = "Hello, I'm interested in your product."
kaag_response = kaag_agent.process_turn(user_input)
rag_response = rag_agent.process_turn(user_input)
norag_response = norag_agent.process_turn(user_input)

print("KAAG response:", kaag_response)
print("RAG response:", rag_response)
print("NoRAG response:", norag_response)
```

## Documentation

For full documentation, visit [docs.kaag.io](https://docs.kaag.io).

## Evaluation

To run evaluations:

```bash
python -m scripts.evaluation <num_runs>
```

Results will be saved in the `results` directory.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use KAAG in your research, please cite:

```bibtex
@article{chaudhuri2023kaag,
  title={Knowledge and Aptitude Augmented Generation: Adaptive Multi-Turn Interaction in LLM Systems},
  author={Chaudhuri, Shauryadeep},
  journal={AroundAI},
  year={2023}
}
```

## Contact

For questions and support, please open an issue on the GitHub repository or contact shaurya@aroundai.co.