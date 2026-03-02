# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is an experimentation and learning repository for AI/ML Python scripts. Scripts are standalone proof-of-concepts demonstrating various LLM providers, agent frameworks, and orchestration patterns. There is no single application — each file is independent.

## Setup

```bash
# Copy and fill in API keys
cp .env.example .env

# Install dependencies (all scripts share this environment)
pip install -r requirements.txt
```

Required API keys (set in `.env`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`. Most scripts use `python-dotenv` to load them automatically.

For Ollama-based scripts (e.g. `multi_agents_agentic.py`, `tst_ollama_*.py`): requires a local Ollama server running at `http://localhost:11434`.

## Running Scripts

```bash
# Run any standalone script
python <script_name>.py

# Run the reporting agent (CSV-based LangGraph agent)
python reporting_agent.py

# Run the refactored version with CLI
python -m reporting_agent_refactored.cli data.csv feed --entity-id F1
```

## Running Tests

Tests only exist for the refactored reporting agent:

```bash
# Run all tests
pytest reporting_agent_refactored/test_reporting_agent.py -v

# With coverage
pytest reporting_agent_refactored/test_reporting_agent.py --cov=reporting_agent_refactored

# Filter specific tests
pytest reporting_agent_refactored/test_reporting_agent.py -k "test_calculate" -v
```

## Architecture Overview

### Script Categories

| Pattern | Example Files | Framework |
|---------|--------------|-----------|
| LangGraph agents with tools | `tst_lang_graph.py`, `tst_lg_*.py` | LangGraph + LangChain |
| Data reporting agent | `reporting_agent.py` | LangGraph + pandas + matplotlib |
| Multi-agent orchestration | `multi_agents_agentic.py` | LangGraph + Ollama |
| CrewAI agents | `crewai/` directory | CrewAI |
| MCP server/client | `mcp_server.py`, `mcp_client.py` | FastMCP / mcp library |
| Direct API calls | `tst_groq_1.py`, `openai_2.py`, `tst_nvidia.py` | Provider SDKs |
| Local LLM inference | `tst_ollama_*.py`, `ollama_web_rag.py` | Ollama |
| RAG pipelines | `tst_rag.py`, `tst_rag_2.py`, `tst_llama_index.py` | LlamaIndex / LangChain |

### Key LangGraph Pattern (used throughout)

All LangGraph agents follow the same structure:
1. Define a typed `AgentState` (TypedDict)
2. Create node functions that receive and return state
3. Build a `StateGraph`, add nodes and edges
4. Use `tools_condition` for conditional tool-use routing
5. Compile and invoke or stream

```python
# Typical pattern
from langgraph.graph import StateGraph, END
graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("chatbot", tools_condition)
app = graph.compile()
for event in app.stream({"messages": [HumanMessage(content=query)]}):
    ...
```

### Reporting Agent (`reporting_agent.py`)

LangGraph graph for CSV-based activity reporting. Flow: `load_data → route_report_type → (canned_report | custom_report)`. Uses pandas for aggregation, matplotlib/seaborn for charts, OpenAI GPT-4 for custom report text. Outputs to `./reports/`.

### Reporting Agent Refactored (`reporting_agent_refactored/`)

Production-quality refactor of the above demonstrating dependency injection. All external dependencies (LLM, filesystem, time, visualization renderer) are injected via a `Dependencies` dataclass. Mock implementations in `test_mocks.py` enable 70+ tests with zero API calls and zero disk I/O. Same public API (`run_report()`).

Key files:
- `interfaces.py` — Abstract base classes (`TimeProvider`, `FileSystemInterface`, `LLMInterface`, `VisualizationRenderer`, `ConfigProvider`)
- `implementations.py` — Real implementations using OpenAI, matplotlib, filesystem
- `test_mocks.py` — In-memory mocks for testing
- `test_reporting_agent.py` — 70+ tests, ~95% coverage

### CrewAI (`crewai/`)

Two agent implementations:
- `task_description_agent.py` — Single agent, dynamic LLM provider selection by model prefix (`ollama/`, `openai/`, `perplexity/`)
- `task_description_agent_advanced.py` — YAML-driven config (`config/agents.yaml`, `config/tasks.yaml`)

### MCP (`mcp_server.py` / `mcp_client.py`)

`mcp_server.py` uses `FastMCP` to expose a prompt, a resource, and a tool via decorator pattern. `mcp_client.py` connects via SSE to `localhost:6277` and calls tools by name.

## LLM Provider Reference

| Provider | Import / Class | Key env var |
|----------|---------------|-------------|
| OpenAI | `ChatOpenAI`, `openai.OpenAI` | `OPENAI_API_KEY` |
| Anthropic | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| Groq | `ChatGroq`, `groq.Groq` | `GROQ_API_KEY` |
| Google | `google.genai` | `GOOGLE_API_KEY` |
| Perplexity | `openai.OpenAI(base_url=...)` | `PERPLEXITY_API_KEY` |
| Ollama | `ChatOllama`, `ollama` | none (local) |
| NVIDIA | `openai.OpenAI(base_url=...)` | `NVIDIA_API_KEY` |
