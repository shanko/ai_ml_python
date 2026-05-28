"""
A2A Summarizer Agent — HTTP Server (port 5001)

A real A2A-protocol agent built with the official a2a-sdk.
Receives tasks from peer agents, calls OpenAI to produce a 3-bullet summary,
and returns the result via the A2A event queue.

  Protocol flow
  ─────────────
  Client → POST /  (JSON-RPC message/send)
      → DefaultRequestHandler → SummarizerExecutor.execute()
          → enqueue TaskStatusUpdateEvent(working)
          → OpenAI API call
          → enqueue TaskArtifactUpdateEvent  (the summary text)
          → enqueue TaskStatusUpdateEvent(completed)
      → A2AStarletteApplication serialises events → HTTP response

  AgentCard (service discovery)
  ─────────────────────────────
  GET /.well-known/agent-card.json  →  agent metadata + skills

Usage
─────
  source ~/code/lang/py/bin/activate
  export OPENAI_API_KEY=sk-...
  python a2a_summarizer.py

Then in a second terminal:
  python a2a_researcher.py
"""

import os

import uvicorn
from openai import AsyncOpenAI

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

# ── Config ─────────────────────────────────────────────────────────────────────

PORT = 5001
oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Core agent logic ───────────────────────────────────────────────────────────

class SummarizerExecutor(AgentExecutor):
    """
    Implements the AgentExecutor interface required by a2a-sdk.

    execute() is called by DefaultRequestHandler for every incoming task.
    Results are published to the EventQueue (not returned directly) —
    the SDK drains the queue and serialises events into the HTTP response.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Register the task so the client can track it by ID
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        # Tell the client we are actively processing
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=new_agent_text_message("Summarising..."),
                ),
                final=False,
            )
        )

        # Extract the first text part from the incoming message
        user_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                # Part is a RootModel — the actual content is in .root
                root = getattr(part, "root", part)
                text = getattr(root, "text", None)
                if text:
                    user_text = text
                    break

        print(f"\n[SummarizerAgent] Task {context.task_id[:8]}")
        print(f"[SummarizerAgent] Input ({len(user_text)} chars):\n{user_text[:300]}...")

        # Call OpenAI
        response = await oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise summariser. "
                        "Given raw research notes, produce exactly 3 bullet points. "
                        "Each bullet is one sentence and starts with '•'."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            temperature=0,
        )
        summary = response.choices[0].message.content.strip()
        print(f"[SummarizerAgent] Summary:\n{summary}")

        # Publish the result as a named artifact
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=new_text_artifact(name="summary", text=summary),
                last_chunk=True,
            )
        )

        # Signal completion — client unblocks after this event
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.completed),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


# ── Server bootstrap ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    skill = AgentSkill(
        id="summarisation",
        name="Bullet-point summariser",
        description=(
            "Condenses raw research notes into a 3-bullet executive summary. "
            "Input: plain-text research notes. Output: 3 '•' bullet points."
        ),
        tags=["summarisation", "nlp", "research"],
        examples=["Summarise these notes on transformers: ..."],
    )

    agent_card = AgentCard(
        name="SummarizerAgent",
        description="Condenses research notes into 3 concise bullet points using GPT-4o-mini.",
        url=f"http://localhost:{PORT}",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    handler = DefaultRequestHandler(
        agent_executor=SummarizerExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    print(f"[SummarizerAgent] Starting on http://localhost:{PORT}")
    print(f"[SummarizerAgent] AgentCard → http://localhost:{PORT}/.well-known/agent-card.json\n")
    uvicorn.run(app.build(), host="127.0.0.1", port=PORT)
