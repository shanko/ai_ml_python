"""
A2A Researcher Agent — Client

Gathers raw facts on a topic (using OpenAI), then delegates summarisation
to the SummarizerAgent via the real A2A protocol.

  Protocol flow
  ─────────────
  1. A2ACardResolver  →  GET /.well-known/agent-card.json  (service discovery)
  2. A2AClient        →  POST / (JSON-RPC message/send)     (send task)
  3. Parse response   →  extract artifact text              (read output)

Usage
─────
  # Terminal 1 — start the server first:
  python a2a_summarizer.py

  # Terminal 2 — run the client:
  python a2a_researcher.py
  python a2a_researcher.py "Quantum computing"   # any topic
"""

import asyncio
import os
import sys
from uuid import uuid4

import httpx
from openai import AsyncOpenAI

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TextPart,
)

# ── Config ─────────────────────────────────────────────────────────────────────

SUMMARIZER_URL = "http://localhost:5001"
oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Researcher logic ───────────────────────────────────────────────────────────

async def gather_facts(topic: str) -> str:
    """Call OpenAI to produce raw research notes on a topic."""
    response = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. "
                    "Given a topic, write 5–6 specific facts or key developments, "
                    "one per line. Include numbers and dates where possible."
                ),
            },
            {"role": "user", "content": f"Research topic: {topic}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_summary(task: Task) -> str | None:
    """Pull text from the first artifact in a completed Task."""
    if not task.artifacts:
        return None
    for artifact in task.artifacts:
        if not artifact.parts:
            continue
        for part in artifact.parts:
            # Part is a RootModel — actual content is in part.root
            root = getattr(part, "root", part)
            text = getattr(root, "text", None)
            if text:
                return text
    return None


async def run(topic: str) -> None:
    print(f"\n{'='*60}")
    print(f"A2A RESEARCHER  →  SUMMARIZER DEMO")
    print(f"Topic: '{topic}'")
    print(f"{'='*60}")

    # ── Step 1: Gather raw facts locally ──────────────────────────────────────
    print("\n[ResearcherAgent] Gathering raw facts with OpenAI...")
    raw_facts = await gather_facts(topic)
    print(f"[ResearcherAgent] Facts:\n{raw_facts}")

    async with httpx.AsyncClient() as http_client:
        # ── Step 2: Discover SummarizerAgent via A2A service discovery ────────
        # Fetches /.well-known/agent-card.json and validates against AgentCard schema
        print(f"\n[ResearcherAgent] Discovering agent at {SUMMARIZER_URL}...")
        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url=SUMMARIZER_URL,
        )
        card = await resolver.get_agent_card()
        print(f"[ResearcherAgent] Found: '{card.name}'")
        print(f"  Skills: {[s.id for s in card.skills]}")
        print(f"  Version: {card.version}")

        # ── Step 3: Build and send an A2A Task ────────────────────────────────
        # A2AClient uses the url from the AgentCard to route requests
        client = A2AClient(httpx_client=http_client, agent_card=card)

        message = Message(
            role=Role.user,
            message_id=str(uuid4()),
            parts=[
                TextPart(
                    text=(
                        f"Summarise these research notes on '{topic}':\n\n"
                        f"{raw_facts}"
                    )
                )
            ],
        )
        params = MessageSendParams(
            message=message,
            configuration=MessageSendConfiguration(
                accepted_output_modes=["text"]
            ),
        )
        request = SendMessageRequest(id=str(uuid4()), params=params)

        print(f"\n[ResearcherAgent] Sending A2A task to '{card.name}'...")
        response = await client.send_message(request)

        # ── Step 4: Read the TaskResult ───────────────────────────────────────
        # SendMessageResponse.root is a discriminated union:
        #   SendMessageSuccessResponse  →  .result is Task or Message
        #   JSONRPCErrorResponse        →  .error describes the failure
        root = response.root

        if not isinstance(root, SendMessageSuccessResponse):
            print(f"\n[ResearcherAgent] RPC error: {root.error}")
            return

        result = root.result   # Task or Message

        if isinstance(result, Task):
            summary = extract_summary(result)
            if summary:
                print(f"\n{'='*60}")
                print(f"FINAL SUMMARY  (from {card.name})")
                print(f"{'='*60}")
                print(summary)
            else:
                print(f"\n[ResearcherAgent] Task completed but no artifact found.")
                print(f"  State: {result.status.state}")
        else:
            # Inline Message response (no task created)
            text = getattr(result, "content", str(result))
            print(f"\n[ResearcherAgent] Inline response:\n{text}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "Transformer neural networks"
    asyncio.run(run(topic))
