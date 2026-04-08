"""Local pre-submission validation script for SkillCoach.

Run with:
    python validate.py

Exits with code 0 if all checks pass, code 1 if any fail.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one PASS/FAIL result and print it immediately."""
    _results.append((label, condition, detail))
    symbol = "+" if condition else "x"
    status = "PASS" if condition else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  {symbol} [{status}] {label}{suffix}")


# ---------------------------------------------------------------------------
# Validation coroutine
# ---------------------------------------------------------------------------

async def run_checks() -> None:
    """Execute all validation checks."""
    print("\n=== SkillCoach Pre-submission Validation ===\n", flush=True)

    # -- 1. Required files exist ----------------------------------------------
    print("-- Required Files --")
    required_files = [
        "openenv.yaml",
        "inference.py",
        "Dockerfile",
        "README.md",
        "requirements.txt",
        "skillcoach_env.py",
        "tasks.py",
        "validate.py",
    ]
    for fname in required_files:
        check(f"{fname} exists", (ROOT / fname).exists())

    # -- 2. openenv.yaml content ----------------------------------------------
    print("\n-- openenv.yaml Fields --")
    yaml_ok = False
    try:
        import yaml  # type: ignore

        with open(ROOT / "openenv.yaml") as fh:
            cfg = yaml.safe_load(fh)

        for field in ("name", "version", "description", "author", "tags", "tasks"):
            check(f"openenv.yaml has '{field}'", field in cfg)

        task_names = {t["name"] for t in cfg.get("tasks", [])}
        for tname in ("identify-error", "hint-without-answer", "guided-debugging"):
            check(f"Task '{tname}' declared in yaml", tname in task_names)

        yaml_ok = True
    except ImportError:
        check("pyyaml importable", False, "pip install pyyaml")
    except Exception as exc:
        check("openenv.yaml parseable", False, str(exc))

    # -- 3. Environment instantiation ----------------------------------------─
    print("\n-- Environment Instantiation --")
    env_module_ok = False
    try:
        from skillcoach_env import (  # noqa: PLC0415
            SkillCoachAction,
            SkillCoachEnv,
            SkillCoachInfo,
            SkillCoachObservation,
        )

        env_module_ok = True
        check("skillcoach_env importable", True)
    except Exception as exc:
        check("skillcoach_env importable", False, str(exc))

    if env_module_ok:
        for tname in ("identify-error", "hint-without-answer", "guided-debugging"):
            try:
                SkillCoachEnv(task_name=tname)
                check(f"SkillCoachEnv('{tname}') instantiated", True)
            except Exception as exc:
                check(f"SkillCoachEnv('{tname}') instantiated", False, str(exc))

        # -- 4. reset() ------------------------------------------------------─
        print("\n-- reset() --")
        env_ie = SkillCoachEnv(task_name="identify-error")
        try:
            obs = await env_ie.reset()
            check("reset() returns SkillCoachObservation", isinstance(obs, SkillCoachObservation))
            check("obs.student_message is str", isinstance(obs.student_message, str) and bool(obs.student_message))
            check("obs.buggy_code is str", isinstance(obs.buggy_code, str) and bool(obs.buggy_code))
            check("obs.turn == 0", obs.turn == 0)
            check("obs.history == []", obs.history == [])
            check("obs.task_name correct", obs.task_name == "identify-error")
        except Exception as exc:
            check("reset() succeeds", False, str(exc))

        # -- 5. step() --------------------------------------------------------
        print("\n-- step() --")
        env_ie2 = SkillCoachEnv(task_name="identify-error")
        await env_ie2.reset()
        try:
            action = SkillCoachAction(response="This looks like a type error in the code.")
            result = await env_ie2.step(action)
            check("step() returns 4-tuple", isinstance(result, tuple) and len(result) == 4)
            _obs2, reward, done, info = result
            check(f"reward in [0.0, 1.0]", 0.0 < reward < 1.0, f"reward={reward:.3f}")
            check("done is bool", isinstance(done, bool))
            check("info is SkillCoachInfo", isinstance(info, SkillCoachInfo))
            check("info.grader_feedback is str", isinstance(info.grader_feedback, str))
        except Exception as exc:
            check("step() succeeds", False, str(exc))

        # -- 6. state() ------------------------------------------------------─
        print("\n-- state() --")
        env_s = SkillCoachEnv(task_name="hint-without-answer")
        await env_s.reset()
        try:
            s = env_s.state()
            check("state() returns dict", isinstance(s, dict))
            for key in ("task_name", "turn", "done", "history"):
                check(f"state has '{key}'", key in s)
        except Exception as exc:
            check("state() succeeds", False, str(exc))

        # -- 7. Graders return scores in [0, 1] ------------------------------─
        print("\n-- Graders --")
        try:
            from tasks import get_task, grade_response  # noqa: PLC0415

            grader_cases: list[tuple[str, str]] = [
                ("identify-error", "This looks like a type error in the code."),
                ("identify-error", "There is a syntax error here."),
                ("hint-without-answer", "Think carefully about the boundary of your loop range."),
                ("hint-without-answer", "x"),  # too short → 0.0
                ("guided-debugging", "What do you think happens when n equals zero?"),
                ("guided-debugging", "Have you traced through what each iteration does?"),
            ]
            for tname, resp in grader_cases:
                ts = get_task(tname)
                gr = grade_response(task_name=tname, task_state=ts, response=resp, turn=1)
                check(
                    f"grader '{tname}' score in [0,1]",
                    0.0 < gr.score < 1.0,
                    f"score={gr.score:.3f}",
                )
        except Exception as exc:
            check("graders importable and callable", False, str(exc))

    # -- 8. FastAPI /health endpoint ------------------------------------------
    print("\n-- FastAPI Endpoints --")
    try:
        import httpx  # noqa: PLC0415
        from skillcoach_env import app  # noqa: PLC0415

        # httpx >= 0.20 requires ASGITransport; fall back for older versions
        try:
            transport = httpx.ASGITransport(app=app)  # type: ignore[attr-defined]
            client_kwargs: dict[str, Any] = {"transport": transport, "base_url": "http://test"}
        except AttributeError:
            client_kwargs = {"app": app, "base_url": "http://test"}  # type: ignore[assignment]

        async with httpx.AsyncClient(**client_kwargs) as client:
            # /health
            resp = await client.get("/health")
            check("/health returns 200", resp.status_code == 200, f"status={resp.status_code}")
            check("/health returns {status: ok}", resp.json().get("status") == "ok")

            # /reset
            resp2 = await client.post(
                "/reset", json={"task_name": "identify-error"}
            )
            check("/reset returns 200", resp2.status_code == 200, f"status={resp2.status_code}")
            body2 = resp2.json()
            check("/reset returns observation", "student_message" in body2 and "buggy_code" in body2)

            # /step
            resp3 = await client.post(
                "/step", json={"response": "What type does the variable hold?"}
            )
            check("/step returns 200", resp3.status_code == 200, f"status={resp3.status_code}")
            body3 = resp3.json()
            check(
                "/step returns {observation, reward, done, info}",
                all(k in body3 for k in ("observation", "reward", "done", "info")),
            )
            check(
                "/step reward in [0,1]",
                0.0 <= body3.get("reward", -1) <= 1.0,
                f"reward={body3.get('reward')}",
            )

            # /state
            resp4 = await client.get("/state")
            check("/state returns 200", resp4.status_code == 200)
            check("/state returns dict", isinstance(resp4.json(), dict))

    except Exception as exc:
        check("FastAPI endpoint tests", False, str(exc))

    # -- 9. close() tidies up ------------------------------------------------─
    print("\n-- close() --")
    if env_module_ok:
        try:
            env_c = SkillCoachEnv(task_name="guided-debugging")
            await env_c.reset()
            await env_c.close()
            s = env_c.state()
            check("close() resets turn to 0", s["turn"] == 0)
            check("close() clears history", s["history"] == [])
        except Exception as exc:
            check("close() succeeds", False, str(exc))

    # -- Summary --------------------------------------------------------------
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'='*44}")
    print(f"Summary: {passed}/{total} passed, {failed} failed")
    print(f"{'='*44}\n")

    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_checks())
