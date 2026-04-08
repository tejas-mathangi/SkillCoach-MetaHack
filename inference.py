"""Baseline inference script for the SkillCoach environment.

Mandatory stdout format
-----------------------
[START] task=<task_name> env=skillcoach model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables
---------------------
API_BASE_URL   Base URL for the OpenAI-compatible LLM endpoint
               Default: https://api.openai.com/v1
MODEL_NAME     Model identifier passed to the API
               Default: gpt-4o-mini
HF_TOKEN       API key / HuggingFace token used for authentication
               Default: reads OPENAI_API_KEY, then empty string

Notes
-----
* All LLM calls use the OpenAI client exclusively (never anthropic / requests).
* [END] is always printed even when an exception occurs (try/finally).
* Runtime target: < 20 minutes on vcpu=2, memory=8 GB.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import OpenAI

from skillcoach_env import SkillCoachAction, SkillCoachEnv, SkillCoachObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

# Max steps per task (tasks 1 & 2 always finish in 1 step; task 3 needs ≤ 5)
MAX_STEPS_EASY_MEDIUM: int = 8
MAX_STEPS_HARD: int = 5

TASKS: list[str] = ["identify-error", "hint-without-answer", "guided-debugging"]

SYSTEM_PROMPT: str = (
    "You are a Socratic coding tutor. Your ONLY job is to help students find bugs themselves.\n"
    "Rules:\n"
    "1. NEVER give the direct answer or fix\n"
    "2. Ask ONE guiding question per turn\n"
    "3. Point to the area of the code that's relevant without saying what's wrong\n"
    "4. If asked for the answer directly, redirect with a question\n"
    "5. Keep responses under 100 words"
)

# ---------------------------------------------------------------------------
# OpenAI client (must not be swapped for any other client)
# ---------------------------------------------------------------------------

_client: OpenAI = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict[str, str]]) -> str:
    """
    Send *messages* to the configured LLM endpoint via the OpenAI client.

    Returns the assistant message content as a plain string.
    """
    completion = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=200,
        temperature=0.7,
    )
    return (completion.choices[0].message.content or "").strip()


def build_messages(obs: SkillCoachObservation) -> list[dict[str, str]]:
    """
    Construct the full message list for the LLM from *obs*.

    History is included as a summary in the user turn so the agent
    remembers what it has already asked.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    history_section = ""
    if obs.history:
        lines = "\n".join(
            f"  Turn {i + 1}: {msg}" for i, msg in enumerate(obs.history)
        )
        history_section = f"\n\n**Your previous responses:**\n{lines}"

    user_content = (
        f"**Task:** {obs.task_name}{history_section}\n\n"
        f"**Student says:** {obs.student_message}\n\n"
        f"**Student's code:**\n```python\n{obs.buggy_code}\n```\n\n"
        f"**Error message:** {obs.error_message or '(none provided)'}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(task_name: str) -> None:
    """
    Run a complete episode for *task_name*, printing required log lines.

    The [END] line is always printed via a try/finally block.
    """
    max_steps = MAX_STEPS_HARD if task_name == "guided-debugging" else MAX_STEPS_EASY_MEDIUM

    print(
        f"[START] task={task_name} env=skillcoach model={MODEL_NAME}",
        flush=True,
    )

    env = SkillCoachEnv(task_name=task_name)
    rewards: list[float] = []
    steps: int = 0
    success: bool = False
    final_score: float = 0.0

    try:
        obs = await env.reset()

        for step_num in range(1, max_steps + 1):
            error_str: str | None = None
            reward: float = 0.0
            done: bool = False
            action_str: str = ""

            try:
                messages = build_messages(obs)
                action_str = call_llm(messages)
                action = SkillCoachAction(response=action_str)
                obs, reward, done, _info = await env.step(action)
                rewards.append(reward)
                steps = step_num

                if done:
                    # reward at done=True is the final episode score
                    final_score = reward
                    success = True

            except Exception as exc:  # noqa: BLE001
                error_str = str(exc)
                done = True

            # Truncate action for log line readability
            action_display = repr(action_str[:80])

            print(
                f"[STEP]  step={step_num} "
                f"action={action_display} "
                f"reward={reward:.2f} "
                f"done={str(done).lower()} "
                f"error={error_str if error_str is not None else 'null'}",
                flush=True,
            )

            if done:
                break

    finally:
        await env.close()

        if not rewards:
            final_score = 0.0
        elif not success:
            # Episode ended due to an error — use last recorded reward
            final_score = rewards[-1] if rewards else 0.0

        rewards_str = (
            ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        )

        print(
            f"[END]   success={str(success).lower()} "
            f"steps={steps} "
            f"score={final_score:.3f} "
            f"rewards={rewards_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all three tasks sequentially and emit structured logs."""
    for task_name in TASKS:
        await run_episode(task_name)
        print(flush=True)  # blank separator between tasks


if __name__ == "__main__":
    asyncio.run(main())
