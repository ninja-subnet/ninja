"""v13 orchestration: reserved finisher wall + salvage on every empty exit."""

from __future__ import annotations

import dataclasses
import time

from agent.agent_loop import AgentOutcome, AgentRunConfig, run_agent_loop
from agent.finisher import run_finisher
from agent.repo_diff import collect_repo_patch
from agent.verify import patch_passes_runtime, sanitize_patch


_FINISHER_RESERVE_SECONDS = 15.0
_FINISHER_MIN_REMAINING = 8.0
_END_MARGIN = 3.0


def run_solve(
    *,
    started: float,
    repo_path: str,
    issue: str,
    model_name: str,
    base_url: str,
    proxy_token: str,
    max_steps: int,
    command_timeout: int,
    max_tokens: int,
    max_observation_chars: int,
    max_log_chars: int,
    max_message_chars: int,
    wall_clock_limit: float,
) -> AgentOutcome:
    main_wall = wall_clock_limit
    if wall_clock_limit > 0:
        main_wall = max(75.0, wall_clock_limit - _FINISHER_RESERVE_SECONDS)

    run_config = AgentRunConfig(
        repo_dir=repo_path,
        model_name=model_name,
        base_url=base_url,
        auth_token=proxy_token,
        max_steps=max_steps,
        command_timeout=command_timeout,
        max_tokens=max_tokens,
        max_observation_chars=max_observation_chars,
        max_log_chars=max_log_chars,
        max_message_chars=max_message_chars,
        wall_clock_limit=main_wall,
        issue_text=issue,
    )
    try:
        from agent.reroll import run_best_of_two
    except Exception:
        run_best_of_two = None

    remaining_before = _remaining_seconds(started, wall_clock_limit)
    use_reroll = run_best_of_two is not None and remaining_before >= 90.0
    if use_reroll:
        outcome = run_best_of_two(run_config, "", issue)
    else:
        outcome = run_agent_loop(config=run_config, task="")
    outcome = _finalize_outcome(outcome, repo_path, issue)

    if not collect_repo_patch(repo_path).strip():
        remaining = _remaining_seconds(started, wall_clock_limit)
        if remaining >= _FINISHER_MIN_REMAINING:
            deadline = time.monotonic() + remaining
            try:
                run_finisher(run_config, repo_path, issue, deadline)
            except Exception:
                pass
            outcome = _finalize_outcome(outcome, repo_path, issue)
    return outcome


def _remaining_seconds(started: float, wall_clock_limit: float) -> float:
    if wall_clock_limit <= 0:
        return 9999.0
    return max(0.0, wall_clock_limit - (time.monotonic() - started) - _END_MARGIN)


def _finalize_outcome(outcome: AgentOutcome, repo_path: str, issue_text: str = "") -> AgentOutcome:
    patch = sanitize_patch(collect_repo_patch(repo_path))
    if (
        patch.strip()
        and issue_text
        and outcome.exit_status != "Submitted"
        and not patch_passes_runtime(repo_path, patch, issue_text)
    ):
        patch = ""
    success = bool(patch.strip())
    exit_status = outcome.exit_status
    message = outcome.message
    if success and exit_status != "Submitted":
        exit_status = "SubmittedOnDisk"
        message = f"patch on disk after {outcome.exit_status}: {outcome.message}"
    return dataclasses.replace(outcome, patch=patch, success=success, exit_status=exit_status, message=message)
