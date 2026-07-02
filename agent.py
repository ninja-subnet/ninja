#!/usr/bin/env python3
"""
Multi-file SWE coding agent for the tau subnet.

Contract (unchanged from the public single-file base agent):
    The validator imports this file and calls:

        solve(
            repo_path="/tmp/task_repo",
            issue="Fix the bug...",
            model="validator-managed-model",
            api_base="http://validator-proxy/v1",
            api_key="per-run-proxy-token"
        )

    It returns a dict with patch, logs, steps, cost, and success.

Layout:
    agent.py             validator-owned contract + thin solve() wiring
    agent/prompts.py     system/instance templates tuned for diff-match scoring
    agent/model.py       stdlib OpenAI-compatible chat client with retries
    agent/environment.py fresh-subshell bash executor
    agent/agent_loop.py  the query -> act -> observe step loop
    agent/repo_diff.py   harness-compatible patch collection

All inference uses only the validator-provided api_base/api_key; there are no
third-party dependencies and no sampling overrides (the validator proxy owns
sampling).
"""

from __future__ import annotations

import os
import re
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from agent.agent_loop import AgentRunConfig, run_agent_loop
from agent.prompts import build_task_prompt
from agent.repo_diff import collect_repo_patch

# -----------------------------
# Config
# -----------------------------

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "50"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "15"))

# VALIDATOR CONTRACT: These defaults are only fallbacks for local testing and
# validator wiring. During real validation the validator passes model, api_base,
# and api_key into solve(). Keep this code compatible with that path.
DEFAULT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("NINJA_MODEL", "")
DEFAULT_API_BASE = (
    os.environ.get("AGENT_API_BASE")
    or os.environ.get("NINJA_INFERENCE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL", "")
)
DEFAULT_API_KEY = (
    os.environ.get("AGENT_API_KEY")
    or os.environ.get("NINJA_INFERENCE_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "8000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))
MAX_MESSAGE_CHARS = int(os.environ.get("AGENT_MAX_MESSAGE_CHARS", "90000"))


# Stay under the validator's per-round budget so the loop can finish gracefully
# and report its own patch instead of relying on the kill path. The validator
# now exports its real per-round budget as TAU_AGENT_TIMEOUT_SECONDS; honor it
# (leaving a margin for diff collection) so a looser budget actually lets the
# agent keep working. Falls back to the conservative 280s when unset.
def _wall_clock_limit_seconds() -> float:
    budget = os.environ.get("TAU_AGENT_TIMEOUT_SECONDS")
    if budget:
        try:
            return max(60.0, float(int(budget)) - 20.0)
        except ValueError:
            pass
    return 280.0


WALL_CLOCK_LIMIT_SECONDS = _wall_clock_limit_seconds()
_REPO_SUMMARY_ITEM_LIMIT = 80
_PRELOAD_MAX_CHARS = 8000
_SKIP_DIR_NAMES = frozenset({
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    ".next",
    "dist",
    "build",
    "target",
    "vendor",
    "coverage",
    ".gradle",
})
_FILE_IN_ISSUE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|json|yaml|yml|md|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
    re.I,
)


def _normalize_api_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _resolve_inference_config(
    model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, str]:
    model_name = (model or DEFAULT_MODEL).strip()
    base = (api_base or DEFAULT_API_BASE).strip()
    key = (api_key if api_key is not None else DEFAULT_API_KEY).strip()

    if not model_name:
        raise ValueError("model is required; validators must pass the centrally managed model id")
    if not base:
        raise ValueError("api_base is required; validators must pass the managed inference proxy URL")
    if not key:
        raise ValueError("api_key is required; validators must pass the per-run proxy token")

    return model_name, _normalize_api_base(base), key


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    return build_task_prompt(task_text=issue, repo_summary=repo_summary, preloaded_context=preloaded_context)


def build_repo_summary(repo_dir: str) -> str:
    if not repo_dir or not os.path.isdir(repo_dir):
        return ""
    paths = _repo_paths(repo_dir)
    shown = paths[:_REPO_SUMMARY_ITEM_LIMIT]
    note = "" if len(paths) <= len(shown) else f"\n... ({len(paths) - len(shown)} more items)"
    return "\n".join(shown) + note


def _repo_paths(repo_dir: str) -> list[str]:
    root_dir = os.path.abspath(repo_dir)
    paths: list[str] = []
    for root, dir_names, file_names in os.walk(root_dir, topdown=True, followlinks=False):
        dir_names[:] = sorted(name for name in dir_names if name not in _SKIP_DIR_NAMES)
        rel_root = os.path.relpath(root, root_dir)
        rel_prefix = "" if rel_root == "." else rel_root.replace("\\", "/")
        for name in dir_names:
            rel = f"{rel_prefix}/{name}" if rel_prefix else name
            paths.append(rel + "/")
        for name in sorted(file_names):
            rel = f"{rel_prefix}/{name}" if rel_prefix else name
            paths.append(rel)
        if len(paths) >= _REPO_SUMMARY_ITEM_LIMIT * 3:
            break
    return sorted(paths)


def issue_named_context(issue: str, repo_dir: str) -> str:
    blocks: list[str] = []
    used = 0
    for path in _existing_issue_files(issue, repo_dir, limit=3):
        content = _read_repo_file(repo_dir, path)
        if not content:
            continue
        budget = _PRELOAD_MAX_CHARS - used
        if budget <= 200:
            break
        clipped = content[:budget]
        suffix = "\n... (truncated)" if len(content) > len(clipped) else ""
        block = (
            f"-----\nFILE NAME: {path}\n"
            "NOTE: current content of a file named by the task; use it as context.\n"
            f"FILE CONTENT:\n```\n{clipped}{suffix}\n```\n-----"
        )
        blocks.append(block)
        used += len(clipped)
    return "\n".join(blocks)


def _existing_issue_files(issue: str, repo_dir: str, *, limit: int) -> list[str]:
    out: list[str] = []
    for match in _FILE_IN_ISSUE_RE.finditer(issue or ""):
        rel = match.group(1).strip().lstrip("./")
        if rel and rel not in out and os.path.isfile(os.path.join(repo_dir, rel)):
            out.append(rel)
        if len(out) >= limit:
            break
    return out


def _read_repo_file(repo_dir: str, relative_path: str) -> str:
    try:
        with open(os.path.join(repo_dir, relative_path), "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        model_name, base_url, proxy_token = _resolve_inference_config(model, api_base, api_key)
        repo_summary = build_repo_summary(repo_path)
        preloaded_context = issue_named_context(issue, repo_path)
        run_config = AgentRunConfig(
            repo_dir=repo_path,
            model_name=model_name,
            base_url=base_url,
            auth_token=proxy_token,
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_tokens=max_tokens,
            max_observation_chars=MAX_OBSERVATION_CHARS,
            max_log_chars=MAX_TOTAL_LOG_CHARS,
            max_message_chars=MAX_MESSAGE_CHARS,
            wall_clock_limit=WALL_CLOCK_LIMIT_SECONDS,
        )
        outcome = run_agent_loop(
            config=run_config,
            task=build_initial_user_prompt(issue, repo_summary, preloaded_context),
        )
        elapsed = time.monotonic() - started
        return {
            "patch": outcome.patch,
            "logs": outcome.logs,
            "steps": outcome.steps,
            "cost": outcome.cost,
            "success": outcome.success,
            "message": f"{outcome.exit_status}: {outcome.message} in {elapsed:.1f}s",
        }
    except Exception:
        fallback_patch = collect_repo_patch(repo_path)
        return {
            "patch": fallback_patch,
            "logs": traceback.format_exc()[-8000:],
            "steps": 0,
            "cost": None,
            "success": bool(fallback_patch.strip()),
            "message": "agent crashed; returning the on-disk repository diff",
        }
