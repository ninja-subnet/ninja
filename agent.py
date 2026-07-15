#!/usr/bin/env python3
"""
Invest SN66 agent v13.1.1 — runtime smoke + quality vs king.

v13.1.1: auto runtime smoke before submit, softened early-submit (4 steps until
verified), allow runtime verify commands post-patch, multifile context preload.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from agent.pipeline import run_solve
from agent.repo_diff import collect_repo_patch
from agent.verify import sanitize_patch

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "50"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "15"))
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

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "16000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))
MAX_MESSAGE_CHARS = int(os.environ.get("AGENT_MAX_MESSAGE_CHARS", "90000"))


def _wall_clock_limit_seconds() -> float:
    budget = os.environ.get("TAU_AGENT_TIMEOUT_SECONDS")
    if budget:
        try:
            return max(60.0, float(int(budget)) - 20.0)
        except ValueError:
            pass
    return 280.0


WALL_CLOCK_LIMIT_SECONDS = _wall_clock_limit_seconds()


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
        outcome = run_solve(
            started=started,
            repo_path=repo_path,
            issue=issue,
            model_name=model_name,
            base_url=base_url,
            proxy_token=proxy_token,
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_tokens=max_tokens,
            max_observation_chars=MAX_OBSERVATION_CHARS,
            max_log_chars=MAX_TOTAL_LOG_CHARS,
            max_message_chars=MAX_MESSAGE_CHARS,
            wall_clock_limit=WALL_CLOCK_LIMIT_SECONDS,
        )
        patch = sanitize_patch(outcome.patch or collect_repo_patch(repo_path))
        elapsed = time.monotonic() - started
        return {
            "patch": patch,
            "logs": outcome.logs,
            "steps": outcome.steps,
            "cost": outcome.cost,
            "success": bool(patch.strip()),
            "message": f"{outcome.exit_status}: {outcome.message} in {elapsed:.1f}s",
        }
    except Exception:
        fallback_patch = sanitize_patch(collect_repo_patch(repo_path))
        return {
            "patch": fallback_patch,
            "logs": traceback.format_exc()[-8000:],
            "steps": 0,
            "cost": None,
            "success": bool(fallback_patch.strip()),
            "message": "agent crashed; returning the on-disk repository diff",
        }
