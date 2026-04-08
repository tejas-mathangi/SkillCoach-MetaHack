"""Task definitions, scenarios, and deterministic graders for SkillCoach.

Three tasks of increasing difficulty:
  identify-error      — name the error type without giving the fix (1 turn)
  hint-without-answer — give a useful hint but not the solution (1 turn)
  guided-debugging    — guide student via questions over up to 5 turns
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Grading result (used by skillcoach_env.py — defined here to avoid circular
# imports since tasks.py must NOT import from skillcoach_env.py)
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    """Result returned by every grader function."""

    score: float            # Final score for this step, always in [0.0, 1.0]
    grader_feedback: str    # Human-readable explanation of the score
    hint_too_direct: bool   # True if agent gave away the answer
    keywords_found: list[str]  # Diagnostic/fix keywords detected in response
    task_complete: bool     # True when the episode should end


# ---------------------------------------------------------------------------
# TaskState — mutable runtime state for one episode
# ---------------------------------------------------------------------------

@dataclass
class TaskState:
    """Holds the chosen scenario and accumulates episode state."""

    task_name: str
    scenario: dict[str, Any]
    per_turn_rewards: list[float] = field(default_factory=list)
    student_found_answer: bool = False

    # -- scenario property shortcuts -----------------------------------------

    @property
    def student_message(self) -> str:
        """Initial student message shown at reset."""
        return self.scenario["student_message"]

    @property
    def buggy_code(self) -> str:
        """The buggy code snippet."""
        return self.scenario["buggy_code"]

    @property
    def error_message(self) -> str:
        """Error message visible to the student (empty string if none)."""
        return self.scenario.get("error_message", "")

    # -- turn helpers ---------------------------------------------------------

    def get_next_student_message(self, turn: int) -> str:
        """
        Return the student's message that follows the agent's *turn*-th response.

        For single-turn tasks the initial message is repeated (it won't be read
        again since done=True).  For guided-debugging the messages are
        pre-scripted: index 0 follows agent turn 1, index 1 follows turn 2, etc.
        """
        if self.task_name in ("identify-error", "hint-without-answer"):
            return self.scenario["student_message"]

        msgs: list[str] = self.scenario.get("student_turn_messages", [])
        idx = turn - 1          # turn is 1-indexed after increment in step()
        if 0 <= idx < len(msgs):
            return msgs[idx]
        return msgs[-1] if msgs else self.scenario["student_message"]

    def check_and_flag_student_found_answer(self, turn: int) -> bool:
        """
        Check whether the student's next message (after agent turn *turn*)
        contains any solution keyword.  Sets self.student_found_answer if so.
        """
        if self.task_name != "guided-debugging":
            return False
        msg = self.get_next_student_message(turn)
        for kw in self.scenario.get("solution_keywords", []):
            if kw.lower() in msg.lower():
                self.student_found_answer = True
                return True
        return False

    def is_done(self, turn: int) -> bool:
        """Return True when the episode should terminate after *turn* steps."""
        if self.task_name in ("identify-error", "hint-without-answer"):
            return turn >= 1
        # guided-debugging
        max_turns: int = self.scenario.get("max_turns", 5)
        return turn >= max_turns or self.student_found_answer

    def to_dict(self) -> dict[str, Any]:
        """Serialise state (omits raw scenario to keep secrets hidden)."""
        return {
            "task_name": self.task_name,
            "buggy_code": self.scenario["buggy_code"],
            "per_turn_rewards": list(self.per_turn_rewards),
            "student_found_answer": self.student_found_answer,
        }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

IDENTIFY_ERROR_SCENARIOS: list[dict[str, Any]] = [
    {
        "buggy_code": "def add(a, b):\n    return a + b\n\nresult = add(5, '3')",
        "error_message": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "correct_error_type": "type",
        "student_message": "I'm getting an error and I don't know why, can you help?",
    },
    {
        "buggy_code": "def greet():\n    print(mesage)\n\ngreet()",
        "error_message": "NameError: name 'mesage' is not defined",
        "correct_error_type": "name",
        "student_message": "My greet function crashes every time I run it.",
    },
    {
        "buggy_code": "def factorial(n)\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
        "error_message": "SyntaxError: expected ':'",
        "correct_error_type": "syntax",
        "student_message": "My code won't even run — I get an error immediately.",
    },
    {
        "buggy_code": "arr = [1, 2, 3]\nprint(arr[5])",
        "error_message": "IndexError: list index out of range",
        "correct_error_type": "runtime",
        "student_message": "I'm trying to access an element but my program crashes.",
    },
    {
        "buggy_code": "def is_even(n):\n    return n % 2 == 1\n\nprint(is_even(4))  # Expected: True, Got: False",
        "error_message": "AssertionError: is_even(4) returned False, expected True",
        "correct_error_type": "logic",
        "student_message": "My function gives the wrong answer but there is no Python exception.",
    },
]

HINT_WITHOUT_ANSWER_SCENARIOS: list[dict[str, Any]] = [
    {
        "buggy_code": "arr = [1, 2, 3]\nfor i in range(len(arr) + 1):\n    print(arr[i])",
        "error_message": "IndexError: list index out of range",
        "correct_fix": "range(len(arr))",
        "fix_keywords": ["range(len(arr))", "len(arr))", "remove the +1", "range(3)"],
        "diagnostic_keywords": ["boundary", "index", "range", "off by one", "last", "exceed"],
        "student_message": "My code keeps crashing, what's wrong?",
    },
    {
        "buggy_code": "def double(x):\n    result = x * 2\n\nprint(double(5))  # prints None",
        "error_message": "Output is None instead of 10",
        "correct_fix": "return result",
        "fix_keywords": ["return result", "return x * 2", "return x*2"],
        "diagnostic_keywords": ["return", "value", "function", "missing", "none", "result"],
        "student_message": "My function prints None instead of the right number.",
    },
    {
        "buggy_code": "age = input('Enter age: ')\nif age > 18:\n    print('Adult')",
        "error_message": "TypeError: '>' not supported between instances of 'str' and 'int'",
        "correct_fix": "int(age)",
        "fix_keywords": ["int(age)", "int(", "convert to int", "cast to int"],
        "diagnostic_keywords": ["type", "string", "integer", "compare", "input", "convert"],
        "student_message": "My age check crashes whenever I enter a number.",
    },
    {
        "buggy_code": (
            "numbers = [3, 1, 4, 1, 5]\n"
            "sorted_numbers = numbers\n"
            "sorted_numbers.sort()\n"
            "print(numbers)  # also sorted!"
        ),
        "error_message": "Original list was unexpectedly modified",
        "correct_fix": "numbers.copy()",
        "fix_keywords": ["copy()", ".copy()", "list(numbers)", "sorted(numbers)"],
        "diagnostic_keywords": ["reference", "copy", "original", "mutable", "same list", "alias"],
        "student_message": "My original list keeps changing even though I only modified the copy.",
    },
    {
        "buggy_code": (
            "def find_max(lst):\n"
            "    max_val = lst[0]\n"
            "    for x in lst:\n"
            "        if x > max_val:\n"
            "            max_val = x\n"
            "        return max_val  # wrong indent — returns after first iteration"
        ),
        "error_message": "Returns first element regardless of list contents",
        "correct_fix": "unindent the return statement",
        "fix_keywords": ["unindent", "return max_val", "outside the for", "dedent"],
        "diagnostic_keywords": ["indent", "loop", "iteration", "early return", "complete", "all elements"],
        "student_message": "My find_max function always returns the first element of the list.",
    },
]

GUIDED_DEBUGGING_SCENARIOS: list[dict[str, Any]] = [
    {
        "buggy_code": (
            "def factorial(n):\n"
            "    if n == 0:\n"
            "        return 0  # bug: should return 1\n"
            "    return n * factorial(n-1)"
        ),
        "error_message": "factorial(5) returns 0 instead of 120",
        "solution": "return 1 when n == 0",
        "solution_keywords": ["return 1", "base case", "0! equals 1", "zero factorial", "should be 1"],
        "max_turns": 5,
        "student_message": "My factorial function gives wrong answers for everything.",
        "student_turn_messages": [
            "When I call factorial(0), I get 0 instead of 1.",
            "Hmm, factorial(1) also gives 0. Is it something about the base case?",
            "Wait — the base case returns 0 but mathematically 0! = 1, so I should return 1 as the base case!",
            "Oh I see it now! I need to return 1 for n == 0!",
        ],
    },
    {
        "buggy_code": (
            "def fizzbuzz(n):\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 3 == 0:\n"
            "            print('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            print('Buzz')\n"
            "        elif i % 15 == 0:  # bug: never reached\n"
            "            print('FizzBuzz')\n"
            "        else:\n"
            "            print(i)"
        ),
        "error_message": "Numbers divisible by 15 print 'Fizz' instead of 'FizzBuzz'",
        "solution": "check divisible by 15 first",
        "solution_keywords": ["15 first", "order matters", "FizzBuzz before", "check 15 before", "move the 15"],
        "max_turns": 5,
        "student_message": "Numbers divisible by 15 just show 'Fizz', not 'FizzBuzz'.",
        "student_turn_messages": [
            "I see that 15 is divisible by both 3 and 5. Is the order of my checks important?",
            "So if 15 passes the i % 3 check first, it never reaches the i % 15 check?",
            "I get it! I need to check for 15 first — order matters here! Move the 15 check before 3 and 5.",
            "Yes! Moving the 15 check to the top fixes it — check 15 before 3 or 5!",
        ],
    },
    {
        "buggy_code": (
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr)  # bug: should be len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1"
        ),
        "error_message": "IndexError: list index out of range",
        "solution": "high = len(arr) - 1",
        "solution_keywords": ["len(arr) - 1", "len(arr)-1", "minus 1", "last valid index", "last index"],
        "max_turns": 5,
        "student_message": "My binary search crashes with an IndexError sometimes.",
        "student_turn_messages": [
            "The error happens when mid points past the end of the array.",
            "Is the problem with my 'high' variable? It's set to len(arr)...",
            "Oh! len(arr) is one past the last valid index. I need len(arr) - 1 as the last index!",
            "Yes! high should be len(arr) - 1, which is the last valid index!",
        ],
    },
]

_SCENARIO_MAP: dict[str, list[dict[str, Any]]] = {
    "identify-error": IDENTIFY_ERROR_SCENARIOS,
    "hint-without-answer": HINT_WITHOUT_ANSWER_SCENARIOS,
    "guided-debugging": GUIDED_DEBUGGING_SCENARIOS,
}


def get_task(task_name: str) -> TaskState:
    """Randomly select a scenario for *task_name* and return a fresh TaskState."""
    if task_name not in _SCENARIO_MAP:
        raise ValueError(
            f"Unknown task {task_name!r}. Valid names: {list(_SCENARIO_MAP)}"
        )
    scenario = random.choice(_SCENARIO_MAP[task_name])
    return TaskState(task_name=task_name, scenario=scenario)


# ---------------------------------------------------------------------------
# Grader — identify-error
# ---------------------------------------------------------------------------

# Aliases that count as "close match" for each error type
_ERROR_TYPE_ALIASES: dict[str, list[str]] = {
    "type":    ["typeerror", "type error", "type mismatch", "wrong type"],
    "syntax":  ["syntaxerror", "syntax error", "syntax issue"],
    "runtime": ["runtimeerror", "runtime error", "indexerror", "index error", "out of range"],
    "logic":   ["logic error", "logical error", "logic bug", "wrong output", "incorrect result"],
    "name":    ["nameerror", "name error", "undefined variable", "not defined"],
}

# Patterns that suggest the agent gave a direct fix (penalised)
_DIRECT_FIX_PATTERNS: list[str] = [
    r"\bint\s*\(",
    r"\bstr\s*\(",
    r"\bfloat\s*\(",
    r"should be",
    r"change it to",
    r"replace with",
    r"\bfix is\b",
    r"\bsolution is\b",
    r"\bthe answer is\b",
    r"\byou need to\b",
    r"\bjust add\b",
    r"\bjust change\b",
]


def _grade_identify_error(scenario: dict[str, Any], response: str) -> GradeResult:
    """
    Deterministic grader for the identify-error task.

    Scoring:
      1.0 — exact error category present in response
      0.7 — close match (e.g. "TypeError" when category is "type")
      0.3 — relevant response (contains generic diagnostic words)
      0.1 — response contains a direct fix pattern (penalty)
      0.0 — empty or completely irrelevant
    """
    resp_lower = response.lower().strip()

    if not resp_lower:
        return GradeResult(
            score=0.01,
            grader_feedback="Empty response.",
            hint_too_direct=False,
            keywords_found=[],
            task_complete=True,
        )

    # Penalty: direct fix detected
    for pattern in _DIRECT_FIX_PATTERNS:
        if re.search(pattern, resp_lower):
            return GradeResult(
                score=0.1,
                grader_feedback="Response contains a direct fix — penalised to 0.1.",
                hint_too_direct=True,
                keywords_found=[],
                task_complete=True,
            )

    correct = scenario["correct_error_type"].lower()

    # Exact match on error category keyword
    if correct in resp_lower:
        return GradeResult(
            score=0.99,
            grader_feedback=f"Correctly identified error type '{correct}'.",
            hint_too_direct=False,
            keywords_found=[correct],
            task_complete=True,
        )

    # Close match (e.g. "TypeError" for category "type")
    for alias in _ERROR_TYPE_ALIASES.get(correct, []):
        if alias in resp_lower:
            return GradeResult(
                score=0.7,
                grader_feedback=f"Close match: found '{alias}' for error type '{correct}'.",
                hint_too_direct=False,
                keywords_found=[alias],
                task_complete=True,
            )

    # Relevant but non-specific
    for word in ("error", "check", "look at", "examine", "problem", "issue", "wrong"):
        if word in resp_lower:
            return GradeResult(
                score=0.3,
                grader_feedback="Relevant response but did not identify the specific error type.",
                hint_too_direct=False,
                keywords_found=[word],
                task_complete=True,
            )

    return GradeResult(
        score=0.01,
        grader_feedback="Irrelevant or empty response.",
        hint_too_direct=False,
        keywords_found=[],
        task_complete=True,
    )


# ---------------------------------------------------------------------------
# Grader — hint-without-answer
# ---------------------------------------------------------------------------

def _grade_hint_without_answer(scenario: dict[str, Any], response: str) -> GradeResult:
    """
    Deterministic grader for the hint-without-answer task.

    Scoring algorithm:
      1. Too short (<20 chars)          → 0.01 immediately
      2. fix_keywords present           → penalty = 0.5
      3. diagnostic_keywords counted    → keyword_score = count / total (capped 0.99)
      4. Too long (>500 chars)          → score capped at 0.4
      5. final = max(0.01, min(cap, keyword_score - penalty))
    """
    stripped = response.strip()

    if len(stripped) < 20:
        return GradeResult(
            score=0.01,
            grader_feedback="Response too short (< 20 characters).",
            hint_too_direct=False,
            keywords_found=[],
            task_complete=True,
        )

    resp_lower = stripped.lower()
    fix_keywords: list[str] = scenario.get("fix_keywords", [])
    diag_keywords: list[str] = scenario.get("diagnostic_keywords", [])

    # Check for fix keywords (direct answer penalty)
    fix_found = [kw for kw in fix_keywords if kw.lower() in resp_lower]
    penalty = 0.5 if fix_found else 0.0

    # Length cap
    length_cap = 0.4 if len(stripped) > 500 else 1.0

    # Diagnostic keyword score
    diag_found = [kw for kw in diag_keywords if kw.lower() in resp_lower]
    keyword_score = min(0.99, len(diag_found) / max(len(diag_keywords), 1))

    raw = max(0.01, keyword_score - penalty)
    final_score = max(0.01, min(length_cap, raw))

    feedback_parts: list[str] = []
    if fix_found:
        feedback_parts.append(f"Gave away fix keywords {fix_found} — penalty -0.5.")
    if diag_found:
        feedback_parts.append(f"Good diagnostic keywords: {diag_found}.")
    else:
        feedback_parts.append("No diagnostic keywords found.")
    if len(stripped) > 500:
        feedback_parts.append("Response too long — score capped at 0.4.")

    return GradeResult(
        score=final_score,
        grader_feedback=" ".join(feedback_parts),
        hint_too_direct=bool(fix_found),
        keywords_found=diag_found,
        task_complete=True,
    )


# ---------------------------------------------------------------------------
# Grader — guided-debugging (per-turn + episode aggregation)
# ---------------------------------------------------------------------------

_QUESTION_WORDS: tuple[str, ...] = (
    "what", "why", "how", "which", "where",
    "have you", "did you", "can you", "do you", "could you",
)


def _grade_guided_turn(
    scenario: dict[str, Any], response: str
) -> tuple[float, bool, list[str]]:
    """
    Score a single agent turn in the guided-debugging task.

    Returns:
        per_turn_reward  — float in [0.01, 0.99]
        hint_too_direct  — bool
        keywords_found   — solution keywords found in response (bad if any)
    """
    resp_lower = response.lower().strip()

    # question_score: 0.5 if agent used a question, else 0.0
    is_question = resp_lower.endswith("?") or any(qw in resp_lower for qw in _QUESTION_WORDS)
    question_score = 0.5 if is_question else 0.0

    # no_answer_score: 0.5 if agent did NOT give the solution
    solution_keywords: list[str] = scenario.get("solution_keywords", [])
    answer_given = [kw for kw in solution_keywords if kw.lower() in resp_lower]
    no_answer_score = 0.5 if not answer_given else 0.0

    per_turn_reward = max(0.01, min(0.99, question_score + no_answer_score))
    return per_turn_reward, bool(answer_given), answer_given


# ---------------------------------------------------------------------------
# Public grading interface
# ---------------------------------------------------------------------------

def grade_response(
    task_name: str,
    task_state: TaskState,
    response: str,
    turn: int,
) -> GradeResult:
    """
    Grade an agent *response* at the given *turn* for *task_name*.

    For single-turn tasks (identify-error, hint-without-answer) this returns
    the final episode score.

    For guided-debugging, per-turn rewards are accumulated in *task_state*.
    When the episode ends (done=True), the returned score is the final episode
    score: ``clamp(mean(per_turn_rewards) + completion_bonus, 0.0, 1.0)``.
    On non-terminal turns the per-turn reward is returned for step logging.

    All returned scores are guaranteed to lie in [0.0, 1.0].
    """
    scenario = task_state.scenario

    if task_name == "identify-error":
        result = _grade_identify_error(scenario, response)
        result.score = max(0.01, min(0.99, result.score))
        task_state.per_turn_rewards.append(result.score)
        return result

    if task_name == "hint-without-answer":
        result = _grade_hint_without_answer(scenario, response)
        result.score = max(0.01, min(0.99, result.score))
        task_state.per_turn_rewards.append(result.score)
        return result

    if task_name == "guided-debugging":
        per_turn_reward, too_direct, kw_found = _grade_guided_turn(scenario, response)
        per_turn_reward = max(0.01, min(0.99, per_turn_reward))
        task_state.per_turn_rewards.append(per_turn_reward)

        # Check if student's next message indicates they found the answer
        task_state.check_and_flag_student_found_answer(turn)

        episode_done = task_state.is_done(turn)

        if episode_done:
            rewards = task_state.per_turn_rewards
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.01
            completion_bonus = 0.3 if task_state.student_found_answer else 0.0
            final_score = max(0.01, min(0.99, mean_reward + completion_bonus))

            parts = [f"Mean per-turn reward: {mean_reward:.2f}."]
            if task_state.student_found_answer:
                parts.append("Student found the answer — +0.3 completion bonus.")
            if too_direct:
                parts.append(f"Warning: agent gave solution keywords {kw_found}.")

            return GradeResult(
                score=final_score,
                grader_feedback=" ".join(parts),
                hint_too_direct=too_direct,
                keywords_found=kw_found,
                task_complete=True,
            )

        # Non-terminal turn
        feedback = f"Turn {turn}: per-turn reward {per_turn_reward:.2f}."
        if too_direct:
            feedback += f" Warning: solution keywords detected: {kw_found}."

        return GradeResult(
            score=per_turn_reward,
            grader_feedback=feedback,
            hint_too_direct=too_direct,
            keywords_found=kw_found,
            task_complete=False,
        )

    raise ValueError(f"Unknown task: {task_name!r}")
