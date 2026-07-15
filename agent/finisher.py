"""Short finisher loop for empty runs. Standard library only."""

from __future__ import annotations

import os
import re
import time

from agent.agent_loop import _cap_messages, _is_submission, _parse_single_command
from agent.environment import execute_command, truncate_text
from agent.model import ChatModel
from agent.prompts import COMPLETION_SENTINEL, format_help_message, render_observation
from agent.repo_diff import collect_repo_patch
from agent.verify import patch_passes_runtime, syntax_errors

_FILE_IN_ISSUE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|json|yaml|yml|md|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
    re.I,
)

FINISHER_MAX_STEPS = 15
PHASE_MSG_CAP = 24000
PHASE_OBS_CAP = 4000
REQUEST_SLACK = 4.0
ISSUE_CAP = 6000
CONTEXT_CAP = 3500
_MODEL_QUERY_RETRIES = 2

FINISHER_SYSTEM = (
    "You are a software engineer finishing an urgent fix in the repository at the "
    "current working directory. Little time remains. Every turn: one short sentence "
    "of reasoning, then exactly ONE bash code block with exactly ONE command. "
    "Your FIRST command must create or modify the source file that owns the requested "
    "behavior — no read-only exploration. Prefer one complete heredoc write or a "
    "focused sed edit per turn, ONE file per turn. When the requested behavior is "
    "implemented and the diff is non-empty, submit with: echo " + COMPLETION_SENTINEL
)

FINISHER_USER_TAIL = (
    "\nThe working tree currently has NO changes and little time remains. Implement "
    "the core named requirement now with direct, tightly scoped edits in the file(s) "
    "the task names. Add nothing the task does not ask for."
)


def run_finisher(base_config, repo: str, issue_text: str, deadline: float) -> None:
    """Run a bounded finisher on the live repo tree. Never raises."""
    if not repo or not os.path.isdir(repo):
        return
    if time.monotonic() >= deadline - REQUEST_SLACK:
        return
    ctx = _named_file_context(repo, issue_text)
    user = (
        "Please solve this issue:\n\n<task>\n"
        + (issue_text or "")[:ISSUE_CAP]
        + "\n</task>\n"
        + ctx
        + FINISHER_USER_TAIL
    )
    messages = [
        {"role": "system", "content": FINISHER_SYSTEM},
        {"role": "user", "content": user},
    ]
    model = ChatModel(
        model_name=base_config.model_name,
        base_url=base_config.base_url,
        auth_token=base_config.auth_token,
        max_completion_tokens=base_config.max_tokens,
    )
    fmt_retries = 0
    for step in range(1, FINISHER_MAX_STEPS + 1):
        if time.monotonic() >= deadline - REQUEST_SLACK:
            return
        msgs = _cap_messages(messages, max_chars=PHASE_MSG_CAP)
        reply = _query_with_retries(model, msgs)
        if reply is None:
            return
        messages.append({"role": "assistant", "content": reply})
        command = _parse_single_command(reply)
        if command is None:
            fmt_retries += 1
            if fmt_retries > 3:
                return
            messages.append({"role": "user", "content": format_help_message()})
            continue
        fmt_retries = 0
        try:
            result = execute_command(command, cwd=repo, timeout=base_config.command_timeout)
        except Exception:
            return
        out = result.get("output") or ""
        rc = result.get("returncode")
        if _is_submission(out, rc):
            return
        patch = collect_repo_patch(repo)
        if patch.strip() and not syntax_errors(repo, patch) and patch_passes_runtime(
            repo, patch, issue_text
        ):
            return
        remaining = max(0.0, deadline - time.monotonic())
        messages.append(
            {
                "role": "user",
                "content": render_observation(
                    returncode=int(rc or 0),
                    output_text=truncate_text(out, PHASE_OBS_CAP),
                    remaining_steps=FINISHER_MAX_STEPS - step,
                    remaining_wall=remaining,
                ),
            }
        )


def _query_with_retries(model: ChatModel, messages: list) -> str | None:
    for attempt in range(_MODEL_QUERY_RETRIES):
        try:
            return model.query(messages)
        except Exception:
            if attempt + 1 >= _MODEL_QUERY_RETRIES:
                return None
            time.sleep(1.5)
    return None


def _named_file_context(repo: str, issue_text: str) -> str:
    blocks: list[str] = []
    used = 0
    seen: set[str] = set()
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text or ""):
        rel = (match.group(1) or "").strip().lstrip("./")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        path = os.path.join(repo, rel)
        if not os.path.isfile(path):
            continue
        try:
            content = open(path, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        room = CONTEXT_CAP - used
        if room <= 200:
            break
        clip = content[:room]
        blocks.append(f"-----\nFILE: {rel}\n```\n{clip}\n```\n-----")
        used += len(clip)
        if len(seen) >= 3:
            break
    return ("\n<context>\n" + "\n".join(blocks) + "\n</context>\n") if blocks else ""
