#!/usr/bin/env python3
"""
CrewAI Agent for generating short descriptions from one-line task explanations.
"""

import os
import argparse
from crewai import Agent, Task, Crew, LLM


class TaskDescriptionGenerator:
    """A crew that generates detailed descriptions from one-line task explanations."""
    
    def __init__(self, model: str = "perplexity/sonar"):
        """
        Initialize the task description generator.
        
        Args:
            model: The LLM model to use (default: perplexity/sonar)
        """
        self.model = model
        self.agent = self._create_agent()
        self.task = self._create_task()
        self.crew = self._create_crew()
    
    def _create_agent(self) -> Agent:
        """Create the description generator agent."""
        # Create LLM configuration based on provider
        if self.model.startswith("ollama/"):
            # Ollama local configuration
            llm = LLM(
                model=self.model,
                base_url="http://localhost:11434"
            )
        elif self.model.startswith("perplexity/"):
            # Perplexity configuration
            llm = LLM(
                model=self.model,
                api_key=os.getenv("PERPLEXITY_API_KEY"),
                base_url="https://api.perplexity.ai"
            )
        else:
            # OpenAI or other providers - use model parameter directly
            return Agent(
                role="Task Description Specialist",
                goal="Transform one-line task explanations into clear, comprehensive short descriptions",
                backstory="""You are an expert at taking brief task explanations and expanding them 
                into well-structured, clear descriptions. You excel at understanding the intent 
                behind a task and providing context, scope, and actionable details in a concise manner. 
                Your descriptions are clear, professional, and easy to understand.""",
                model=self.model,
                verbose=True,
                max_iter=1
            )
        
        # Return agent with LLM configuration for Ollama and Perplexity
        return Agent(
            role="Task Description Specialist",
            goal="Transform one-line task explanations into clear, comprehensive short descriptions",
            backstory="""You are an expert at taking brief task explanations and expanding them 
            into well-structured, clear descriptions. You excel at understanding the intent 
            behind a task and providing context, scope, and actionable details in a concise manner. 
            Your descriptions are clear, professional, and easy to understand.""",
            llm=llm,
            verbose=True,
            max_iter=1
        )
    
    def _create_task(self) -> Task:
        """Create the description generation task."""
        return Task(
            description="Generate a comprehensive short description based on the one-line task explanation provided. "
                       "The description should include: what needs to be done, why it matters, and any key considerations. "
                       "Keep it to 2-4 sentences maximum.",
            expected_output="A well-structured short description (2-4 sentences) that expands on the one-line explanation",
            agent=self.agent
        )
    
    def _create_crew(self) -> Crew:
        """Create the crew with the agent and task."""
        return Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=True,
            max_retry_limit=0
        )
    
    def generate_description(self, one_line_task: str) -> str:
        """
        Generate a description from a one-line task explanation.
        
        Args:
            one_line_task: A one-line explanation of the task
            
        Returns:
            A comprehensive short description
        """
        # Create a fresh task for each generation to avoid state reuse
        task = Task(
            description=f"""Generate a comprehensive short description for the following one-line task: 
            "{one_line_task}"
            
            The description should include: what needs to be done, why it matters, and any key considerations. 
            Keep it to 2-4 sentences maximum.""",
            expected_output="A well-structured short description (2-4 sentences) that expands on the one-line explanation",
            agent=self.agent
        )
        
        # Create a fresh crew for each generation
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,
            max_retry_limit=0
        )
        
        # Kickoff the crew with the input
        result = crew.kickoff(
            inputs={"task": one_line_task}
        )
        
        return result.raw


def main():
    """Main function to demonstrate the task description generator."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CrewAI Task Description Generator")
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
    
    # Example one-line tasks
    example_tasks = [
        "Build a machine learning model to predict customer churn",
        "Create a REST API for user authentication",
        "Optimize database query performance for reporting",
        "Implement automated testing for the payment module"
    ]
    
    print("=" * 70)
    print("CrewAI Task Description Generator")
    print("=" * 70)
    
    # Initialize the generator
    generator = TaskDescriptionGenerator(model=model)
    
    # Generate descriptions for each example task
    for task in example_tasks:
        print(f"\n📋 One-line Task: {task}")
        print("-" * 70)
        
        description = generator.generate_description(task)
        print(f"📝 Generated Description:\n{description}")
        print("=" * 70)


if __name__ == "__main__":
    # Parse arguments first to check provider
    parser = argparse.ArgumentParser(description="CrewAI Task Description Generator")
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
