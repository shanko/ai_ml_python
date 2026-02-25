# Quick Start Guide

## 30-Second Setup

### Step 1: Get Your API Key
Get an OpenAI API key from [platform.openai.com](https://platform.openai.com)

### Step 2: Install
```bash
pip install -r requirements.txt
```

### Step 3: Configure
```bash
export OPENAI_API_KEY="your-key-here"
```

### Step 4: Run
```bash
python task_description_agent.py
```

---

## What Does It Do?

It takes a one-line task like:
```
"Build a machine learning model to predict customer churn"
```

And generates:
```
This task involves developing a predictive machine learning model that can 
identify customers at risk of leaving by analyzing historical data and behavioral 
patterns. The model will enable proactive retention strategies and help reduce 
customer acquisition costs while maintaining revenue stability.
```

---

## Use It in Your Code

```python
from task_description_agent import TaskDescriptionGenerator

# Create the generator
generator = TaskDescriptionGenerator()

# Generate a description
task = "Create a REST API for user authentication"
description = generator.generate_description(task)
print(description)
```

---

## Next Steps

1. **Explore the Code**: Check out `task_description_agent.py`
2. **Try the Advanced Version**: Run `task_description_agent_advanced.py`
3. **Customize**: Modify the agent's role, goal, or backstory
4. **Add Tools**: Extend with custom tools for validation or formatting
5. **Read Full Docs**: See `README.md` for complete documentation

---

## Troubleshooting

**Error: "OPENAI_API_KEY not found"**
→ Run: `export OPENAI_API_KEY="your-key"`

**Error: "Module crewai not found"**
→ Run: `pip install -r requirements.txt`

**Error: "Configuration file not found"**
→ The advanced version creates config files automatically

---

## Key Concepts

### Agent
An AI character with a role, goal, and backstory that performs tasks

### Task
A specific job the agent needs to complete with clear inputs and outputs

### Crew
A team of agents working together to accomplish goals

### Tools
Functions the agent can use to accomplish its tasks

---

## Model Options

Use any of these by passing to the generator:

```python
# OpenAI Models
TaskDescriptionGenerator(model="gpt-4-turbo")
TaskDescriptionGenerator(model="gpt-4o")
TaskDescriptionGenerator(model="gpt-4o-mini")  # Fastest & cheapest

# Other Providers (Requires additional setup)
TaskDescriptionGenerator(model="claude-3-sonnet-20240229")
```

---

## Example Tasks to Try

```python
tasks = [
    "Build a recommendation engine for e-commerce",
    "Create a CI/CD pipeline for microservices",
    "Develop a real-time chat application",
    "Implement zero-downtime database migration",
    "Design a scalable caching strategy"
]

generator = TaskDescriptionGenerator()
for task in tasks:
    print(f"Task: {task}")
    print(f"Description: {generator.generate_description(task)}")
    print("-" * 50)
```

---

## Performance Tips

1. **Use gpt-4o-mini for speed and cost** (default)
2. **Use gpt-4-turbo for better quality** (slower, more expensive)
3. **Set `verbose=False`** in Agent for cleaner output in production
4. **Batch requests** if processing multiple tasks

---

## Need Help?

- Read the full `README.md`
- Check [CrewAI Docs](https://docs.crewai.com)
- Visit [GitHub Issues](https://github.com/crewAIInc/crewAI/issues)

Happy building! 🚀
