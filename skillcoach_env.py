"""SkillCoach environment — OpenEnv-compliant Socratic debugging tutor.

Exposes both a Python API (SkillCoachEnv) and an HTTP API (FastAPI app)
so the environment can be hosted as a HuggingFace Space.

HTTP endpoints
--------------
POST /reset  — reset environment, returns observation JSON
POST /step   — advance one step, returns {observation, reward, done, info}
GET  /state  — current internal state
GET  /health — liveness probe for HF Space ping checks
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Body, FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict

from tasks import GradeResult, TaskState, get_task, grade_response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SkillCoachObservation(BaseModel):
    """Everything the agent sees at each step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    student_message: str    # What the student says / shows this turn
    buggy_code: str         # The code snippet with the bug
    error_message: str      # The error the student sees
    turn: int               # Current turn number (0 at reset)
    task_name: str          # Which task is active
    history: list[str]      # All previous agent responses this episode


class SkillCoachAction(BaseModel):
    """The agent's response to the student."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: str   # Text response — must be a hint or question, never the direct fix


class SkillCoachInfo(BaseModel):
    """Diagnostic information returned alongside the reward."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grader_feedback: str        # Explanation of the score awarded
    hint_too_direct: bool       # True if agent gave away the answer
    keywords_found: list[str]   # Diagnostic/solution keywords detected
    task_complete: bool         # True when the episode has ended


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SkillCoachEnv:
    """
    OpenEnv-compliant Socratic debugging tutor environment.

    The agent plays the role of a coding tutor.  It observes a student's
    description of their problem together with their buggy code and must
    guide them toward the fix using only questions and indirect hints —
    never by giving the direct answer.

    Supported tasks
    ---------------
    identify-error      — name the error type in one turn
    hint-without-answer — give a useful hint without the fix in one turn
    guided-debugging    — Socratic multi-turn conversation (max 5 turns)
    """

    VALID_TASKS: tuple[str, ...] = (
        "identify-error",
        "hint-without-answer",
        "guided-debugging",
    )

    def __init__(self, task_name: str = "identify-error") -> None:
        """
        Initialise the environment for *task_name*.

        Parameters
        ----------
        task_name:
            One of ``"identify-error"``, ``"hint-without-answer"``,
            ``"guided-debugging"``.
        """
        if task_name not in self.VALID_TASKS:
            raise ValueError(
                f"Unknown task {task_name!r}. Valid: {self.VALID_TASKS}"
            )
        self.task_name: str = task_name
        self._task_state: Optional[TaskState] = None
        self._turn: int = 0
        self._history: list[str] = []
        self._done: bool = False

    # ------------------------------------------------------------------
    # Core OpenEnv interface
    # ------------------------------------------------------------------

    async def reset(self) -> SkillCoachObservation:
        """
        Reset to a fresh episode and return the initial observation.

        A random scenario is selected for the active task on each call.
        """
        self._task_state = get_task(self.task_name)
        self._turn = 0
        self._history = []
        self._done = False
        return self._build_obs(self._task_state.student_message)

    async def step(
        self, action: SkillCoachAction
    ) -> tuple[SkillCoachObservation, float, bool, SkillCoachInfo]:
        """
        Apply the agent's action and advance the environment by one step.

        Parameters
        ----------
        action:
            The agent's text response to the student.

        Returns
        -------
        observation:
            Updated state including the student's next message.
        reward:
            Per-turn reward for ongoing steps; final episode score (already
            accounting for any completion bonus) when *done* is True.
        done:
            True when the episode has ended.
        info:
            Diagnostic grading details (see SkillCoachInfo).

        Raises
        ------
        RuntimeError
            If called before reset().
        """
        if self._task_state is None:
            raise RuntimeError("Call reset() before step().")

        self._turn += 1
        self._history.append(action.response)

        grade_result: GradeResult = grade_response(
            task_name=self.task_name,
            task_state=self._task_state,
            response=action.response,
            turn=self._turn,
        )

        # is_done may have been updated by grade_response (student found answer)
        self._done = self._task_state.is_done(self._turn)

        # Advance to the next student message for multi-turn tasks
        next_student_msg = self._task_state.get_next_student_message(self._turn)
        obs = self._build_obs(next_student_msg)

        reward = max(0.01, min(0.99, float(grade_result.score)))
        info = SkillCoachInfo(
            grader_feedback=grade_result.grader_feedback,
            hint_too_direct=grade_result.hint_too_direct,
            keywords_found=grade_result.keywords_found,
            task_complete=grade_result.task_complete,
        )
        return obs, reward, self._done, info

    def state(self) -> dict[str, Any]:
        """Return the current internal state as a plain dict."""
        return {
            "task_name": self.task_name,
            "turn": self._turn,
            "done": self._done,
            "history": list(self._history),
            "task_state": self._task_state.to_dict() if self._task_state else None,
        }

    async def close(self) -> None:
        """Release episode resources and reset internal counters."""
        self._task_state = None
        self._history = []
        self._turn = 0
        self._done = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, student_message: str) -> SkillCoachObservation:
        """Construct an observation from current env state."""
        assert self._task_state is not None, "task_state must be set before building observation"
        return SkillCoachObservation(
            student_message=student_message,
            buggy_code=self._task_state.buggy_code,
            error_message=self._task_state.error_message,
            turn=self._turn,
            task_name=self.task_name,
            history=list(self._history),
        )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SkillCoach Environment",
    description=(
        "Socratic debugging tutor — OpenEnv-compliant HTTP interface. "
        "The agent must teach coding by asking questions, never by giving answers."
    ),
    version="1.0.0",
)

# Global environment instance (suitable for a single-user HF Space)
_env: SkillCoachEnv = SkillCoachEnv(task_name="identify-error")


# -- Request models ----------------------------------------------------------

class _ResetRequest(BaseModel):
    """Body for POST /reset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_name: str = "identify-error"


class _StepRequest(BaseModel):
    """Body for POST /step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: str


# -- Endpoints ---------------------------------------------------------------

@app.get("/")
async def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/reset")
async def http_reset(
    request: Optional[_ResetRequest] = Body(default=None),
) -> dict[str, Any]:
    """
    Reset the environment for *task_name* and return the initial observation.

    Body (optional JSON):
        ``{"task_name": "identify-error"}``
    """
    global _env
    task_name = request.task_name if request is not None else "identify-error"
    _env = SkillCoachEnv(task_name=task_name)
    obs = await _env.reset()
    return obs.model_dump()


@app.post("/step")
async def http_step(request: _StepRequest) -> dict[str, Any]:
    """
    Advance the environment by one step.

    Body:
        ``{"response": "<agent text>"}``

    Returns:
        ``{observation, reward, done, info}``
    """
    action = SkillCoachAction(response=request.response)
    obs, reward, done, info = await _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info.model_dump(),
    }


@app.get("/state")
async def http_state() -> dict[str, Any]:
    """Return the current internal state of the active environment."""
    return _env.state()


@app.get("/health")
async def http_health() -> dict[str, str]:
    """Liveness probe — required for HF Space ping checks."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
