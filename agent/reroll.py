"""Primary-draw wrapper: rebuilds the initial prompt with a fuller repository
summary and a shortlist of likely edit targets, runs the base loop once for the
full wall, and falls open to the on-disk diff on any error.

The loop opens on the deterministic shortlist (instant) while ``agent.prefetch``
re-ranks the same candidates with a model call on a daemon thread; when that
lands, the loop swaps the sharper list in. Both the thread and the swap are
best-effort: if either fails, the run is exactly what it was without them."""

from __future__ import annotations

import dataclasses

from agent.agent_loop import AgentOutcome, run_agent_loop
from agent.context import build_context_task
from agent.repo_diff import collect_repo_patch


def run_best_of_two(base_config, task, issue_text) -> AgentOutcome:
    repo = getattr(base_config, "repo_dir", "") or ""
    run_task = task
    try:
        rebuilt = build_context_task(issue_text, repo)
        if rebuilt and "<task>" in rebuilt:
            run_task = rebuilt
    except Exception:
        run_task = task
    try:
        outcome = run_agent_loop(
            config=base_config, task=run_task, task_updater=_start_prefetch(base_config, repo, issue_text)
        )
    except Exception:
        outcome = _floor_outcome(repo)
    return _outcome_on_disk(outcome, repo)


def _start_prefetch(base_config, repo, issue_text):
    """The loop's task_updater, or None if the re-rank can't even be started."""
    try:
        from agent.prefetch import start

        return start(
            issue=issue_text,
            repo_dir=repo,
            model_name=base_config.model_name,
            base_url=base_config.base_url,
            auth_token=base_config.auth_token,
        )
    except Exception:  # noqa: BLE001 - the loop must run with or without a re-rank
        return None


def _outcome_on_disk(outcome, repo):
    try:
        patch = collect_repo_patch(repo)
    except Exception:
        return outcome
    return dataclasses.replace(outcome, patch=patch, success=bool(patch.strip()))


def _floor_outcome(repo):
    try:
        patch = collect_repo_patch(repo)
    except Exception:
        patch = ""
    return AgentOutcome(
        success=bool(patch.strip()),
        patch=patch,
        logs="",
        steps=0,
        cost=None,
        message="wrapper fell open to the on-disk repository diff",
        exit_status="Submitted",
    )
