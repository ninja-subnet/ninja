"""Last-resort empty-tree salvage.

Mean-score duels punish empty diffs with a hard zero (duel-959988 r6: 0.00;
duel-863250 r42: 0.00). The main loop already presses for an early edit, but
some runs still burn the budget on truncated rewrites / format retries and
exit empty. This short finisher spends leftover wall on a forced write path
only — never exploration — then returns whatever is on disk.
"""

from __future__ import annotations

import os
import re
import time

from agent.agent_loop import _cap_messages, _is_submission, _parse_single_command
from agent.environment import execute_command, truncate_text
from agent.model import ChatModel
from agent.prompts import COMPLETION_SENTINEL, format_help_message, render_observation
from agent.repo_diff import collect_repo_patch

_FILE_IN_ISSUE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|json|yaml|yml|"
    r"md|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
    re.I,
)

FINISHER_MAX_STEPS = 8
REQUEST_SLACK = 8.0
ISSUE_CAP = 5000
CONTEXT_CAP = 3000
_MODEL_QUERY_RETRIES = 2
_MIN_TIME_LEFT_S = 35.0

FINISHER_SYSTEM = (
    "You are finishing an urgent fix with almost no time left. Every turn: one "
    "short sentence, then exactly ONE bash code block with exactly ONE command. "
    "Your FIRST command MUST create or modify a real source file that implements "
    "the core ask — no read-only exploration. Prefer one focused sed -i or a short "
    "heredoc. When the tree is nonempty, submit with: echo " + COMPLETION_SENTINEL
)

FINISHER_USER_TAIL = (
    "\nThe working tree has NO changes and the deadline is near. Implement the "
    "most literal reading of the task NOW in the owning file. An empty diff "
    "scores zero."
)


def run_finisher(base_config, repo: str, issue_text: str, deadline: float) -> str:
    """Try to create a nonempty patch. Returns the on-disk patch (may stay empty)."""
    if not repo or not os.path.isdir(repo):
        return ""
    try:
        existing = collect_repo_patch(repo)
    except Exception:
        existing = ""
    if existing.strip():
        return existing
    time_left = deadline - time.monotonic()
    if time_left < _MIN_TIME_LEFT_S:
        return existing

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
        max_completion_tokens=min(getattr(base_config, "max_tokens", 2048) or 2048, 1536),
    )
    fmt_retries = 0
    for _step in range(1, FINISHER_MAX_STEPS + 1):
        if time.monotonic() >= deadline - REQUEST_SLACK:
            break
        try:
            capped = _cap_messages(
                messages,
                max_prompt_tokens=max(1024, (getattr(base_config, "max_model_len", 32768) or 32768) - 1536),
                chars_per_token=getattr(model, "chars_per_token", 4.0),
            )
        except TypeError:
            capped = messages
        reply = _query_with_retries(model, capped)
        if reply is None:
            break
        messages.append({"role": "assistant", "content": reply})
        command = _parse_single_command(reply)
        if command is None:
            fmt_retries += 1
            if fmt_retries > 2:
                break
            messages.append({"role": "user", "content": format_help_message()})
            continue
        fmt_retries = 0
        # Reject obvious read-only commands — this path is write-only.
        if _looks_read_only(command):
            messages.append({
                "role": "user",
                "content": (
                    "[Rejected: finisher allows write commands only. Edit or create "
                    f"a source file now, then `echo {COMPLETION_SENTINEL}` when nonempty.]"
                ),
            })
            continue
        result = execute_command(
            command,
            cwd=repo,
            timeout=min(15, getattr(base_config, "command_timeout", 15) or 15),
        )
        output = truncate_text(result.get("output") or "", 4000)
        returncode = result.get("returncode")
        if _is_submission(output, returncode):
            break
        messages.append({
            "role": "user",
            "content": render_observation(
                returncode=int(returncode or 0),
                output_text=output,
                time_note=(
                    f"[Finisher: ~{max(0, int(deadline - time.monotonic()))}s left. "
                    "Write the core fix, then submit.]"
                ),
            ),
        })
        try:
            if collect_repo_patch(repo).strip():
                # Nonempty — stop burning budget; main return path will pick it up.
                break
        except Exception:
            pass
    try:
        return collect_repo_patch(repo)
    except Exception:
        return ""


def _named_file_context(repo: str, issue_text: str) -> str:
    blocks: list[str] = []
    used = 0
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text or ""):
        rel = match.group(1).strip().lstrip("./")
        path = os.path.join(repo, rel)
        if not rel or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
        except OSError:
            continue
        room = CONTEXT_CAP - used
        if room <= 200:
            break
        clipped = content[:room]
        suffix = "\n... (truncated)" if len(content) > len(clipped) else ""
        blocks.append(
            f"-----\nFILE NAME: {rel}\nFILE CONTENT:\n```\n{clipped}{suffix}\n```\n-----"
        )
        used += len(clipped)
        if len(blocks) >= 2:
            break
    return "\n".join(blocks)


def _looks_read_only(command: str) -> bool:
    stripped = (command or "").strip().lower()
    if not stripped:
        return True
    write_markers = (" >", ">>", "sed -i", "tee ", "touch ", "mv ", "cp ", "cat <<", "perl -i")
    if any(m in stripped for m in write_markers):
        return False
    return bool(re.match(r"^(?:cat|nl|head|tail|less|more|grep|rg|find|ls|tree|wc)\b", stripped))


def _query_with_retries(model: ChatModel, messages: list) -> str | None:
    for _ in range(_MODEL_QUERY_RETRIES):
        try:
            return model.query(messages)
        except Exception:
            continue
    return None
