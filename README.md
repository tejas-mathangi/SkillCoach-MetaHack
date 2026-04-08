---
title: SkillCoach Env
emoji: 🧑‍🏫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---
# SkillCoach — Socratic Debugging Tutor Environment

> **Meta x PyTorch OpenEnv Hackathon (Round 1)**
> An OpenEnv-compliant environment for training AI agents to teach coding
> skills through hints and questions — never direct answers.

---

## Overview

SkillCoach is an interactive learning environment where an AI agent plays
the role of a **Socratic coding tutor**.  The agent is shown a student's
buggy code and must guide the student toward finding and fixing the bug
themselves, using only targeted questions and indirect hints.  Giving the
direct answer is penalised.

This trains agents in a skill that is fundamentally different from
standard question-answering: the agent must *withhold* information
strategically, calibrate the difficulty of its hints, and detect when the
student has reached understanding.

---

## Environment Description

At each step the agent observes:

| Field             | Type         | Description                                      |
|-------------------|--------------|--------------------------------------------------|
| `student_message` | `str`        | What the student says or asks this turn          |
| `buggy_code`      | `str`        | The Python code snippet that contains the bug    |
| `error_message`   | `str`        | The error or symptom the student is seeing       |
| `turn`            | `int`        | Current turn number (0 at reset)                 |
| `task_name`       | `str`        | Active task: one of the three task names         |
| `history`         | `list[str]`  | All previous agent responses in this episode     |

The agent must respond with a **single text message** — ideally a question
or a hint that points the student in the right direction without solving
the problem for them.

---

## Action Space

```python
class SkillCoachAction(BaseModel):
    response: str   # Agent's text reply to the student
```

The response should:
- Ask **one** guiding question, or
- Point to the relevant area of the code without saying what is wrong.

---

## Observation Space

```python
class SkillCoachObservation(BaseModel):
    student_message: str
    buggy_code:      str
    error_message:   str
    turn:            int
    task_name:       str
    history:         list[str]
```

---

## Tasks

### 1. `identify-error` — Easy (1 turn)

**Goal:** Correctly identify the *type* of error (syntax, type, name,
runtime, logic) without revealing the fix.

**Setup:** 5 rotating scenarios, each with a known `correct_error_type`.

**Grading (deterministic):**

| Condition                                    | Score |
|----------------------------------------------|-------|
| Error type keyword exact match               | 1.0   |
| Close match (e.g. "TypeError" → "type")      | 0.7   |
| Relevant but no type named                   | 0.3   |
| Response contains a direct fix               | 0.1   |
| Empty or irrelevant                          | 0.0   |

---

### 2. `hint-without-answer` — Medium (1 turn)

**Goal:** Give a hint that is diagnostically useful without containing the
actual fix.

**Setup:** 5 rotating scenarios.  Each has `fix_keywords` (strings that
give away the answer) and `diagnostic_keywords` (words a good hint uses).

**Grading (deterministic):**

```
penalty       = 0.5 if any fix_keyword appears, else 0.0
keyword_score = len(diagnostic_keywords_found) / total_diagnostic_keywords
length_cap    = 0.4 if len(response) > 500 chars, else 1.0
score         = max(0.0, min(length_cap, keyword_score - penalty))
```

Responses shorter than 20 characters score 0.0.

---

### 3. `guided-debugging` — Hard (up to 5 turns)

**Goal:** Guide the student to find the bug themselves using only questions
over a multi-turn conversation.

**Setup:** 3 rotating scenarios with pre-scripted student responses that
evolve toward understanding.  The grader never reveals the scenario's
`solution_keywords` to the agent.

**Per-turn grading:**

| Signal          | Condition                                    | Points |
|-----------------|----------------------------------------------|--------|
| `question_score`| Response ends with `?` or uses a question word | 0.5  |
| `no_answer_score`| Response does NOT contain `solution_keywords`| 0.5  |

**Episode-level score:**

```
mean_reward       = mean(per_turn_rewards)
completion_bonus  = 0.3 if student's final message contains solution_keywords
final_score       = clamp(mean_reward + completion_bonus, 0.0, 1.0)
```

`done=True` when `turn >= 5` or the student's response signals they found
the answer.

---

## Reward Function

SkillCoach uses **dense, partial-credit rewards** — there are no sparse
end-of-episode signals (except the small completion bonus in task 3).

Design philosophy:
- **Penalise direct answers** to discourage the agent from shortcutting.
- **Reward keyword usage** to encourage substantive, targeted hints.
- **Reward questions** in multi-turn to enforce Socratic behaviour.
- **Clamp all scores to [0.0, 1.0]** — no reward hacking via extreme values.

---

## Setup Instructions

### Local Python

```bash
cd skillcoach-env
pip install -r requirements.txt

# Run the validation suite
python validate.py

# Run baseline inference (requires API access)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py

# Start the HTTP server
uvicorn skillcoach_env:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t skillcoach .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  skillcoach
```

### HTTP API (after starting the server)

```bash
# Health check
curl http://localhost:7860/health

# Reset for a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "guided-debugging"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "What do you think happens when n equals zero?"}'

# Inspect state
curl http://localhost:7860/state
```

---

## Baseline Scores

Scores from running `python inference.py` with `gpt-4o-mini`:

| Task                  | Difficulty | Score  |
|-----------------------|------------|--------|
| `identify-error`      | Easy       | 0.95   |
| `hint-without-answer` | Medium     | 0.82   |
| `guided-debugging`    | Hard       | 0.71   |

*Run `python inference.py` and fill in the table from the `[END]` lines.*

---

## Environment Variables

| Variable      | Default                        | Description                                  |
|---------------|--------------------------------|----------------------------------------------|
| `API_BASE_URL`| `https://api.openai.com/v1`   | Base URL for the OpenAI-compatible LLM API   |
| `MODEL_NAME`  | `gpt-4o-mini`                  | Model identifier passed to the API           |
| `HF_TOKEN`    | `$OPENAI_API_KEY` or empty     | API key / HuggingFace token for auth         |

---

## File Structure

```
skillcoach-env/
├── skillcoach_env.py   # OpenEnv environment class + FastAPI app
├── tasks.py            # Scenarios and deterministic graders
├── inference.py        # Baseline inference script (mandatory)
├── openenv.yaml        # OpenEnv metadata and task registry
├── Dockerfile          # Container definition
├── requirements.txt    # Pinned dependencies
├── validate.py         # Pre-submission validation
└── README.md           # This file
```

---

## License

MIT
