# AI/ML Python

A collection of AI/ML tools and examples using Python.

## File Descriptions

Alphabetical listing of Python files in this repository:

- **1_llm_example.py**: Minimal LLM demo — Claude generates a stakeholder summary as pure text, no reasoning or tools.

- **2_lrm_example.py**: Reasoning-model (LRM) demo — Claude's extended-thinking block deliberates a go/no-go launch decision before answering.

- **3_lam_example.py**: Large Action Model (LAM) demo — Claude uses tools to fetch test/bug data and email the team lead in an agentic loop.

- **a2a_researcher.py**: A2A protocol client — researches a topic via OpenAI, then delegates summarisation to the peer SummarizerAgent.

- **a2a_summarizer.py**: A2A protocol HTTP server (port 5001) — receives tasks from peer agents and returns a 3-bullet OpenAI summary.

- **ai_assist.py**: Scaffold combining Google Calendar event lookup with Twilio reminders (placeholder credentials — not runnable as-is).

- **cloud_llm_benchmark.py**: Benchmarks cloud LLM providers (OpenAI, Anthropic, Groq, Gemini, ...) using whichever API keys are present in the environment.

- **eleven_labs.py**: ElevenLabs TTS and speech-to-text demo — synthesizes audio from text and transcribes a remote MP3.

- **finetune_sft.py**: Minimalist supervised fine-tuning demo using LoRA adapters on a HuggingFace causal LM, with before/after sample prompts.

- **gen_images.py**: Stable Diffusion v1.5 image generation via the `diffusers` library, with per-step timing instrumentation.

- **img2txt_openai.py**: Script that converts images to structured text using OpenAI's GPT-4o model. Processes images into JSON format based on a predefined schema.

- **mcp_client.py**: Client implementation for MCP (Model Control Protocol) to connect to a proxy server.

- **mcp_server.py**: Server implementation for MCP (Model Control Protocol) with demo tools and resources.

- **min_autogluon.py**: Minimal AutoGluon tabular example — trains a `TabularPredictor` on a local CSV and prints the leaderboard.

- **multi_agents_agentic.py**: Two-agent (researcher + writer) LangGraph workflow that collaborates via a local Ollama model.

- **multi_model_llms.py**: Side-by-side helpers showing the same chat call against OpenAI, Anthropic, and Ollama.

- **ollama_benchmark.py**: Benchmarks every locally installed Ollama chat model with a standard prompt and tabulates the results.

- **ollama_cloud.py**: Streams a chat completion from an Ollama-hosted cloud model (e.g. `nemotron-3-nano:30b-cloud`).

- **ollama_web_rag.py**: Simple RAG pipeline — retrieves Wikipedia context and answers questions using a local Ollama model.

- **openai_2.py**: OpenAI API walkthrough (from a Colab assignment) covering model listing and basic text generation.

- **perplexity_api.py**: Uses the Perplexity Sonar API (via OpenAI-compatible client) to extract structured event data from a web page.

- **predictive_models.py**: Classical scikit-learn classification demo — Random Forest vs Gradient Boosting on the breast cancer dataset, with feature-importance plot.

- **reporting_agent.py**: LangGraph CSV-based activity reporting agent — loads data, routes to canned/custom reports, generates charts and text via OpenAI.

- **test_cloudflare_mcp_server.py**: Connects to Cloudflare's demo MCP server over SSE and lists the tools it exposes.

- **translate_2.py**: Translation demo using Meta's NLLB-200 distilled model via HuggingFace `transformers`.

- **tst_autogluon.py**: AutoGluon tabular classifier with CLI args for training/test CSVs and timing output.

- **tst_chat.py**: Example code demonstrating OpenAI chat completion API parameters and usage.

- **tst_code_gen.py**: Code generation example solving a mathematical problem using brute force and algebraic methods.

- **tst_google_adk.py**: Test of the `google-adk` package — Gemini 2.x agent with the Google Search tool, querying current events.

- **tst_granite4_hybrid.py**: Compares an IBM Granite 4.0 Mamba2+Attention+MoE hybrid model against the dense Granite 4.0 variant.

- **tst_granite_guardian.py**: Test script for Ollama's Granite Guardian model with system prompts.

- **tst_groq_1.py**: Example of using Groq API to interact with LLaMA models.

- **tst_helloworld_lg.py**: Hello world example using LangGraph and LangChain frameworks.

- **tst_lang_chain.py**: Demonstrates LangChain usage with translation examples using Groq.

- **tst_lang_graph.py**: Example of building a conversational agent using LangGraph with Anthropic's Claude model.

- **tst_lg_1.py**: LangGraph example #1.

- **tst_lg_2.py**: LangGraph example #2.

- **tst_lg_3.py**: LangGraph example #3.

- **tst_lg_4.py**: LangGraph example #4.

- **tst_lg_weather.py**: LangGraph example for weather-related tasks.

- **tst_lg_wttr.py**: LangGraph example using wttr.in weather service.

- **tst_llama32.py**: Testing script for LLaMA 3.2 models.

- **tst_llama_index.py**: Example using LlamaIndex for retrieval-augmented generation.

- **tst_lottery_lg.py**: LangGraph example for lottery-related processing.

- **tst_marathi_gen.py**: Script for generating text in Marathi language.

- **tst_nvidia.py**: Example of using NVIDIA's API to interact with the Llama 3.1 Nemotron 70B model for text generation.

- **tst_ollama_1.py**: Ollama API usage example #1 with prompt engineering.

- **tst_ollama_2.py**: Ollama API usage example #2.

- **tst_ollama_3.py**: Ollama API usage example #3.

- **tst_ollama_lg.py**: Integration of Ollama with LangGraph.

- **tst_openrouter_api.py**: Minimal OpenRouter API call routing to `openai/gpt-5` via the chat completions endpoint.

- **tst_perplex.py**: Example of using Perplexity AI's API for image analysis.

- **tst_rag.py**: Implementation of Retrieval-Augmented Generation using Google's DataGemma model.

- **tst_rag_2.py**: RAG pipeline using ChromaDB + sentence-transformers for embeddings and Ollama (`tinyllama`) for generation.

- **tst_rag_web.py**: RAG demo that scrapes Google search results with BeautifulSoup and feeds them to Ollama as context.

- **tst_code_gen.py**: Code generation example solving a mathematical problem using brute force and algebraic methods.

- **tst_granite_guardian.py**: Test script for Ollama's Granite Guardian model with system prompts.

- **tst_groq_1.py**: Example of using Groq API to interact with LLaMA models.

- **tst_helloworld_lg.py**: Hello world example using LangGraph and LangChain frameworks.

- **tst_lang_chain.py**: Demonstrates LangChain usage with translation examples using Groq.

- **tst_lang_graph.py**: Example of building a conversational agent using LangGraph with Anthropic's Claude model.

- **tst_lg_1.py**: LangGraph example #1.

- **tst_lg_2.py**: LangGraph example #2.

- **tst_lg_3.py**: LangGraph example #3.

- **tst_lg_4.py**: LangGraph example #4.

- **tst_lg_weather.py**: LangGraph example for weather-related tasks.

- **tst_lg_wttr.py**: LangGraph example using wttr.in weather service.

- **tst_llama32.py**: Testing script for LLaMA 3.2 models.

- **tst_llama_index.py**: Example using LlamaIndex for retrieval-augmented generation.

- **tst_lottery_lg.py**: LangGraph example for lottery-related processing.

- **tst_marathi_gen.py**: Script for generating text in Marathi language.

- **tst_nvidia.py**: Example of using NVIDIA's API to interact with the Llama 3.1 Nemotron 70B model for text generation.

- **tst_ollama_1.py**: Ollama API usage example #1 with prompt engineering.

- **tst_ollama_2.py**: Ollama API usage example #2.

- **tst_ollama_3.py**: Ollama API usage example #3.

- **tst_ollama_lg.py**: Integration of Ollama with LangGraph.

- **tst_perplex.py**: Example of using Perplexity AI's API for image analysis.

- **tst_rag.py**: Implementation of Retrieval-Augmented Generation using Google's DataGemma model.

## TODO

- [ ] Credit where credit is due (i.e. add links to original sources)
- [ ] Fix broken examples (e.g. tst_rag.py)
- [ ] Cleanup code
- [ ] Cleanup requirements.txt
- [ ] Add more examples