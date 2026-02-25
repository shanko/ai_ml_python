# CrewAI Task Description Generator

A Python agent built with CrewAI that transforms one-line task explanations into detailed, comprehensive short descriptions.

## Overview

This project demonstrates how to build an AI agent using the CrewAI framework. The agent takes simple, one-line task descriptions and expands them into well-structured, professional descriptions that include context, scope, and key considerations.

### Features

- **Two Implementation Approaches**:
  1. `task_description_agent.py` - Simple, straightforward implementation
  2. `task_description_agent_advanced.py` - Advanced YAML configuration-based approach

- **Intelligent Expansion**: Transforms brief task explanations into comprehensive descriptions
- **Flexible Configuration**: Easily customize the agent's behavior
- **Professional Output**: Generated descriptions are clear, concise, and actionable

## Requirements

- Python 3.10+
- OpenAI API key

## Installation

### 1. Clone or Download the Project

```bash
cd task-description-generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your-openai-api-key-here
```

Alternatively, set the environment variable directly:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Usage

### Simple Version

Run the basic task description generator:

```bash
python task_description_agent.py
```

**Example Output:**
```
📋 One-line Task: Build a machine learning model to predict customer churn
------
📝 Generated Description:
This task involves developing a predictive machine learning model that can identify 
customers at risk of leaving. The model will analyze historical customer data and 
behavioral patterns to provide early warnings, enabling proactive retention strategies. 
This is crucial for reducing customer acquisition costs and maintaining revenue stability.
```

### Advanced Version (YAML Configuration)

Run the advanced generator with YAML configuration:

```bash
python task_description_agent_advanced.py
```

The advanced version automatically creates `config/` directory with `agents.yaml` and `tasks.yaml` files.

## Project Structure

```
.
├── task_description_agent.py          # Simple implementation
├── task_description_agent_advanced.py # Advanced YAML-based implementation
├── requirements.txt                   # Python dependencies
├── .env.example                       # Example environment variables
└── README.md                          # This file
```

## Code Explanation

### Simple Version (`task_description_agent.py`)

The `TaskDescriptionGenerator` class:

1. **Creates an Agent** with:
   - Role: Task Description Specialist
   - Goal: Transform task explanations into detailed descriptions
   - Tools: Custom `format_description` tool
   - LLM: OpenAI (gpt-4o-mini or gpt-4)

2. **Defines a Task** that:
   - Takes the one-line explanation as input
   - Generates a 2-4 sentence comprehensive description
   - Includes context, scope, and key considerations

3. **Forms a Crew** that:
   - Orchestrates the agent and task
   - Executes the workflow
   - Returns the generated description

### Usage Example

```python
from task_description_agent import TaskDescriptionGenerator

# Initialize the generator
generator = TaskDescriptionGenerator(model="gpt-4o-mini")

# Generate a description
task = "Optimize database query performance"
description = generator.generate_description(task)
print(description)
```

## Advanced Version with YAML Configuration

The advanced version follows CrewAI best practices by separating configuration from code:

### agents.yaml
Defines agent roles, goals, and backstories

### tasks.yaml
Defines task descriptions and expected outputs

This approach makes it easier to:
- Modify agent behavior without changing code
- Maintain separation of concerns
- Scale to multiple agents and tasks

## Customization

### Change the LLM Model

```python
# Use a different model
generator = TaskDescriptionGenerator(model="gpt-4-turbo")
```

### Modify Agent Behavior

In `task_description_agent.py`, modify the agent creation:

```python
def _create_agent(self) -> Agent:
    return Agent(
        role="Your Custom Role",
        goal="Your Custom Goal",
        backstory="Your Custom Backstory",
        # ... other configurations
    )
```

### Add More Tools

Create custom tools using the `@tool` decorator:

```python
from crewai_tools import tool

@tool
def custom_tool(input: str) -> str:
    """Description of what the tool does."""
    return f"Processed: {input}"
```

Then add it to the agent:

```python
return Agent(
    # ...
    tools=[format_description, custom_tool],
    # ...
)
```

## Supported Models

The agent works with any LLM supported by CrewAI, including:

- **OpenAI**: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Open Source**: Via Ollama, LM Studio, etc.

To use a different provider, set appropriate environment variables and modify the model parameter.

## Example Tasks

Here are some example one-line tasks you can try:

```python
tasks = [
    "Build a machine learning model to predict customer churn",
    "Create a REST API for user authentication",
    "Optimize database query performance for reporting",
    "Implement automated testing for the payment module",
    "Design a real-time notification system",
    "Develop a mobile app for inventory management",
    "Create a data pipeline for ETL processes"
]
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"
**Solution**: Make sure you've set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Issue: "Configuration files not found"
**Solution**: The advanced version automatically creates config files. Make sure you have write permissions in the directory.

### Issue: "Module not found"
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

## Contributing

Feel free to extend this agent with:
- Additional tools for task validation
- Support for multiple output formats
- Integration with project management tools
- Custom prompting strategies

## License

This project is provided as-is for educational purposes.

## Resources

- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI GitHub Repository](https://github.com/crewAIInc/crewAI)
- [CrewAI Examples](https://github.com/crewAIInc/crewAI-examples)

## Support

For issues with:
- **CrewAI Framework**: Visit [CrewAI GitHub Issues](https://github.com/crewAIInc/crewAI/issues)
- **OpenAI API**: Visit [OpenAI Help](https://help.openai.com)
