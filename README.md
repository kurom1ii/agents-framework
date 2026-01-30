# Agents Framework

A lightweight Python multi-agent framework with team orchestration, MCP integration, and hybrid memory system.

## Features

- **Async-First Design**: Built from the ground up with async/await for high-performance concurrent operations
- **Protocol-Based Architecture**: Clean interfaces using Python Protocols for flexibility and testability
- **Pydantic v2**: Modern data validation and settings management
- **Team Orchestration**: Coordinate multiple agents with different roles and capabilities
- **MCP Integration**: Model Context Protocol support for tool integration
- **Hybrid Memory System**: Short-term, long-term, and entity memory for context management

## Installation

```bash
# Basic installation
pip install agents-framework

# With development tools
pip install agents-framework[dev]

# With specific LLM providers
pip install agents-framework[anthropic]
pip install agents-framework[openai]

# All extras
pip install agents-framework[all]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/kurom1ii/agents-framework.git
cd agents-framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from agents_framework.core.config import AgentConfig
from agents_framework.core.message import Message, MessageRole

# Create an agent configuration
config = AgentConfig(
    name="research_agent",
    role="Research Analyst",
    goal="Find and analyze information on given topics",
    backstory="You are an expert research analyst with deep expertise in data analysis.",
)

# Create a message
message = Message(
    role=MessageRole.USER,
    content="What are the latest trends in AI?",
)
```

## Project Structure

```
src/agents_framework/
    __init__.py
    core/
        __init__.py
        base.py          # Protocols (Agent, Tool, Memory, LLM)
        message.py       # Message types
        state.py         # AgentState, TeamState
        config.py        # Configuration classes
        exceptions.py    # Custom exceptions
```

## Requirements

- Python 3.10+
- Pydantic v2

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/agents_framework --cov-report=html

# Type checking
mypy src/agents_framework

# Linting and formatting
ruff check src tests
ruff format src tests
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.
