#!/usr/bin/env python3
"""
Advanced CrewAI Agent with YAML configuration for generating task descriptions.
This follows the CrewAI best practices pattern.
"""

import os
import argparse
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, LLM


class AdvancedTaskDescriptionGenerator:
    """Advanced crew using YAML configuration files."""
    
    def __init__(self, config_dir: str = "config", model: str = "perplexity/sonar"):
        """
        Initialize with YAML configuration files.
        
        Args:
            config_dir: Directory containing agents.yaml and tasks.yaml
            model: LLM model to use (default: perplexity/sonar)
        """
        self.config_dir = Path(config_dir)
        self.model = model
        
        # Load configurations
        self.agents_config = self._load_yaml("agents.yaml")
        self.tasks_config = self._load_yaml("tasks.yaml")
        
        # Create agents and tasks
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()
    
    def _load_yaml(self, filename: str) -> dict:
        """Load YAML configuration file."""
        filepath = self.config_dir / filename
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {filepath}")
            return {}
    
    def _create_agents(self) -> dict:
        """Create agents from configuration."""
        agents = {}
        
        for agent_name, agent_config in self.agents_config.items():
            # Create LLM configuration based on provider
            if self.model.startswith("ollama/"):
                # Ollama local configuration
                llm = LLM(
                    model=self.model,
                    base_url="http://localhost:11434"
                )
                agents[agent_name] = Agent(
                    role=agent_config.get("role", "AI Agent"),
                    goal=agent_config.get("goal", "Help with tasks"),
                    backstory=agent_config.get("backstory", ""),
                    llm=llm,
                    verbose=True,
                    max_iter=1
                )
            elif self.model.startswith("perplexity/"):
                # Perplexity configuration
                llm = LLM(
                    model=self.model,
                    api_key=os.getenv("PERPLEXITY_API_KEY"),
                    base_url="https://api.perplexity.ai"
                )
                agents[agent_name] = Agent(
                    role=agent_config.get("role", "AI Agent"),
                    goal=agent_config.get("goal", "Help with tasks"),
                    backstory=agent_config.get("backstory", ""),
                    llm=llm,
                    verbose=True,
                    max_iter=1
                )
            else:
                # OpenAI or other providers - use model parameter directly
                agents[agent_name] = Agent(
                    role=agent_config.get("role", "AI Agent"),
                    goal=agent_config.get("goal", "Help with tasks"),
                    backstory=agent_config.get("backstory", ""),
                    model=self.model,
                    verbose=True,
                    max_iter=1
                )
        
        return agents
    
    def _create_tasks(self) -> dict:
        """Create tasks from configuration."""
        tasks = {}
        
        for task_name, task_config in self.tasks_config.items():
            agent_name = task_config.get("agent", "description_specialist")
            agent = self.agents.get(agent_name)
            
            if agent:
                tasks[task_name] = Task(
                    description=task_config.get("description", ""),
                    expected_output=task_config.get("expected_output", ""),
                    agent=agent
                )
        
        return tasks
    
    def _create_crew(self) -> Crew:
        """Create the crew."""
        return Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            verbose=True,
            max_retry_limit=0
        )
    
    def generate_description(self, one_line_task: str) -> str:
        """Generate description from one-line task."""
        # Recreate tasks with fresh input to avoid state reuse
        tasks = {}
        for task_name, task_config in self.tasks_config.items():
            agent_name = task_config.get("agent", "description_specialist")
            agent = self.agents.get(agent_name)
            
            if agent:
                # Replace {task} placeholder with actual task
                description = task_config.get("description", "").format(task=one_line_task)
                tasks[task_name] = Task(
                    description=description,
                    expected_output=task_config.get("expected_output", ""),
                    agent=agent
                )
        
        # Create fresh crew for each generation
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(tasks.values()),
            verbose=True,
            max_retry_limit=0
        )
        
        result = crew.kickoff(
            inputs={"task": one_line_task}
        )
        return result.raw


def create_example_configs():
    """Create example YAML configuration files."""
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create agents.yaml
    agents_config = {
        "description_specialist": {
            "role": "Task Description Specialist",
            "goal": "Transform one-line task explanations into clear, comprehensive short descriptions",
            "backstory": """You are an expert at taking brief task explanations and expanding them 
into well-structured, clear descriptions. You excel at understanding the intent 
behind a task and providing context, scope, and actionable details. Your descriptions 
are clear, professional, and easy to understand."""
        }
    }
    
    with open(config_dir / "agents.yaml", 'w') as f:
        yaml.dump(agents_config, f, default_flow_style=False, sort_keys=False)
    
    # Create tasks.yaml
    tasks_config = {
        "description_generation_task": {
            "description": """Generate a comprehensive short description based on the one-line task explanation provided.
The description should include: what needs to be done, why it matters, and any key considerations.
Keep it to 2-4 sentences maximum.
Task: {task}""",
            "expected_output": "A well-structured short description (2-4 sentences) that expands on the one-line explanation",
            "agent": "description_specialist"
        }
    }
    
    with open(config_dir / "tasks.yaml", 'w') as f:
        yaml.dump(tasks_config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Example configuration files created in 'config/' directory")


def main():
    """Main function for the advanced generator."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Advanced CrewAI Task Description Generator")
    parser.add_argument(
        "--ollama",
        type=str,
        help="Use Ollama with specified model (e.g., llama3.2)",
        default=None
    )
    parser.add_argument(
        "--cloud",
        type=str,
        help="Use cloud provider: perplexity (default), openai, or specify model with provider=model (e.g., perplexity=sonar-pro)",
        default="perplexity"
    )
    args = parser.parse_args()
    
    # Determine which model to use
    if args.ollama:
        model = f"ollama/{args.ollama}"
        print(f"🦙 Using Ollama with model: {args.ollama}")
    else:
        # Parse cloud provider and optional model
        cloud_arg = args.cloud.lower()
        if "=" in cloud_arg:
            # Format: provider=model (command-line override)
            provider, model_name = cloud_arg.split("=", 1)
            provider = provider.strip()
            model_name = model_name.strip()
        else:
            # Just provider name, check ENV vars or use defaults
            provider = cloud_arg.strip()
            model_name = None
        
        # Set model based on provider
        if provider == "openai":
            # Priority: command-line arg > ENV var > hardcoded default
            if model_name:
                model = model_name
            elif os.getenv("OPENAI_MODEL_NAME"):
                model = os.getenv("OPENAI_MODEL_NAME")
            else:
                model = "gpt-4o-mini"
            print(f"🤖 Using OpenAI with model: {model}")
        else:  # Default to Perplexity
            # Priority: command-line arg > ENV var > hardcoded default
            if model_name:
                model = f"perplexity/{model_name}"
            elif os.getenv("PERPLEXITY_MODEL_NAME"):
                model = f"perplexity/{os.getenv('PERPLEXITY_MODEL_NAME')}"
            else:
                model = "perplexity/sonar"
            # Extract just the model name for display
            display_model = model.split("/", 1)[1] if "/" in model else model
            print(f"🔮 Using Perplexity with model: {display_model}")
    
    # Create example configs if they don't exist
    if not Path("config/agents.yaml").exists():
        create_example_configs()
    
    # Initialize the generator
    generator = AdvancedTaskDescriptionGenerator(config_dir="config", model=model)
    
    # Example tasks
    example_tasks = [
        "Build a machine learning model to predict customer churn",
        "Create a REST API for user authentication",
    ]
    
    print("=" * 70)
    print("Advanced CrewAI Task Description Generator")
    print("=" * 70)
    
    for task in example_tasks:
        print(f"\n📋 One-line Task: {task}")
        print("-" * 70)
        
        description = generator.generate_description(task)
        print(f"📝 Generated Description:\n{description}")
        print("=" * 70)


if __name__ == "__main__":
    # Parse arguments first to check provider
    parser = argparse.ArgumentParser(description="Advanced CrewAI Task Description Generator")
    parser.add_argument(
        "--ollama",
        type=str,
        help="Use Ollama with specified model (e.g., llama3.2)",
        default=None
    )
    parser.add_argument(
        "--cloud",
        type=str,
        help="Use cloud provider: perplexity (default), openai, or specify model with provider=model (e.g., perplexity=sonar-pro)",
        default="perplexity"
    )
    args = parser.parse_args()
    
    # Check API keys based on provider if not using Ollama
    if not args.ollama:
        cloud_arg = args.cloud.lower()
        # Extract provider name (before '=' if present)
        provider = cloud_arg.split("=")[0].strip()
        
        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                print("⚠️  Please set your OPENAI_API_KEY environment variable")
                print("Example: export OPENAI_API_KEY='your-key-here'")
                print("\nOr use Ollama with: --ollama llama3.2")
                exit(1)
        else:  # Perplexity (default)
            if not os.getenv("PERPLEXITY_API_KEY"):
                print("⚠️  Please set your PERPLEXITY_API_KEY environment variable")
                print("Example: export PERPLEXITY_API_KEY='your-key-here'")
                print("\nOr use Ollama with: --ollama llama3.2")
                print("Or use OpenAI with: --cloud openai")
                exit(1)
    
    main()
