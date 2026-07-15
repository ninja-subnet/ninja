"""The agent step loop: query the model, run one bash action, feed the
observation back, finish when the agent echoes the completion sentinel.
Uses a text-based action format.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .environment import execute_command, truncate_text
from .model import ChatModel, ModelQueryError
from .prompts import (
    COMPLETION_SENTINEL,
    SYSTEM_PROMPT,
    build_task_prompt,
    format_help_message,
    render_observation,
)
from .repo_diff import collect_repo_patch
from .verify import (
    command_is_runtime_verify,
    patch_passes_runtime,
    submit_readiness_for_submit,
    submit_readiness_light,
)

_ACTION_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n?```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.DOTALL)
_DOLLAR_LINE_RE = re.compile(r"(?m)^\s*\$[ \t]+(\S.*?)\s*$")
_READ_ONLY_COMMAND_RE = re.compile(
    r"^\s*(?:cat|nl|head|tail|less|more|grep|rg|find|ls|tree|wc)\b",
    re.I,
)
_MAX_FORMAT_RETRIES = 5
_NO_PATCH_NUDGE_STEP = 3
_STRONG_WRITE_NUDGE_STEP = 6
_READ_REJECT_AFTER_STEP = 8
_WRITE_NUDGE_WALL_SECONDS = 70.0
_SUBMIT_NUDGE_WALL_SECONDS = 30.0
_SUBMIT_AFTER_PATCH_STEPS = 4
_SUBMIT_AFTER_PATCH_STEPS_VERIFIED = 2
_POST_PATCH_READ_REJECT_WALL = 15.0
_MODEL_ERROR_RETRIES = 1
_RECENT_MESSAGE_COUNT = 8
_COMPACT_MESSAGE_CHARS = 1200
_COMPACT_MESSAGE_CHARS_LATE = 800
_MIN_COMPACT_MESSAGE_CHARS = 600
_LATE_COMPACT_FROM_STEP = 6


@dataclass
class AgentRunConfig:
    repo_dir: str
    model_name: str
    base_url: str
    auth_token: str
    max_steps: int = 50
    command_timeout: int = 15
    max_tokens: int = 8192
    max_observation_chars: int = 16000
    max_log_chars: int = 260000
    max_message_chars: int = 90000
    wall_clock_limit: float = 0.0
    issue_text: str = ""


@dataclass
class AgentOutcome:
    success: bool
    patch: str
    logs: str
    steps: int
    cost: float | None
    message: str
    exit_status: str = "Submitted"
    transcript: list = field(default_factory=list)


def run_agent_loop(*, config: AgentRunConfig, task: str) -> AgentOutcome:
    model = ChatModel(
        model_name=config.model_name,
        base_url=config.base_url,
        auth_token=config.auth_token,
        max_completion_tokens=config.max_tokens,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task if "<task>" in task else build_task_prompt(task_text=task)},
    ]
    started = time.monotonic()
    log_lines: list = []
    exit_status = "LimitsExceeded"
    message = f"step limit of {config.max_steps} reached"
    format_retries = 0
    no_patch_nudge_sent = False
    strong_write_nudge_sent = False
    write_pressure_sent = False
    submit_pressure_sent = False
    model_error_retries = 0
    steps_since_patch = 0
    submit_ready_nudge_sent = False
    runtime_verified = False

    for step in range(1, max(1, config.max_steps) + 1):
        elapsed = time.monotonic() - started
        remaining_wall = config.wall_clock_limit - elapsed if config.wall_clock_limit > 0 else 9999.0
        if 0 < config.wall_clock_limit <= elapsed:
            exit_status = "TimeExceeded"
            message = f"wall clock limit of {config.wall_clock_limit:.0f}s reached"
            break
        messages = _cap_messages(messages, max_chars=config.max_message_chars, step=step)
        try:
            reply = model.query(messages)
        except ModelQueryError as exc:
            if model_error_retries < _MODEL_ERROR_RETRIES and remaining_wall > 8:
                model_error_retries += 1
                log_lines.append(f"[step {step}] model error, retry {model_error_retries}: {exc}")
                time.sleep(min(3.0, max(0.5, remaining_wall - 2.0)))
                continue
            exit_status = "ModelError"
            message = str(exc)
            log_lines.append(f"[step {step}] model error: {exc}")
            break
        messages.append({"role": "assistant", "content": reply})
        log_lines.append(f"[step {step}] assistant:\n{reply}")

        command = _parse_single_command(reply)
        if command is None:
            format_retries += 1
            if format_retries > _MAX_FORMAT_RETRIES:
                exit_status = "FormatError"
                message = "model kept replying without exactly one bash code block"
                break
            messages.append({"role": "user", "content": format_help_message()})
            log_lines.append(f"[step {step}] format retry {format_retries}")
            continue
        format_retries = 0

        if _should_reject_read_only_command(
            command=command,
            repo_dir=config.repo_dir,
            step=step,
            no_patch_nudge_sent=no_patch_nudge_sent,
            strong_write_nudge_sent=strong_write_nudge_sent,
        ):
            output_text = _read_budget_message(command)
            returncode = 2
            log_lines.append(f"[step {step}] read-only command rejected after write nudges")
        elif _should_reject_post_patch_read_only(
            command=command,
            repo_dir=config.repo_dir,
            remaining_wall=remaining_wall,
            steps_since_patch=steps_since_patch,
            runtime_verified=runtime_verified,
        ):
            output_text = _post_patch_read_message(command)
            returncode = 2
            log_lines.append(f"[step {step}] post-patch read-only command rejected")
        else:
            result = execute_command(command, cwd=config.repo_dir, timeout=config.command_timeout)
            output_text = result.get("output") or ""
            returncode = result.get("returncode")
            if not returncode and command_is_runtime_verify(command):
                patch = collect_repo_patch(config.repo_dir)
                if patch.strip() and patch_passes_runtime(
                    config.repo_dir, patch, config.issue_text
                ):
                    runtime_verified = True
        log_lines.append(f"[step {step}] $ {command}\n{truncate_text(output_text, 2000)}")
        if _is_submission(output_text, returncode):
            patch = collect_repo_patch(config.repo_dir)
            if not patch.strip():
                messages.append({"role": "user", "content": _empty_submit_guard_message()})
                log_lines.append(f"[step {step}] empty submit rejected")
                if not no_patch_nudge_sent:
                    no_patch_nudge_sent = True
                    messages.append({"role": "user", "content": _no_patch_nudge_message(step)})
                continue
            blocked = submit_readiness_for_submit(
                config.repo_dir,
                patch,
                config.issue_text,
                runtime_verified=runtime_verified,
            )
            if blocked:
                messages.append({"role": "user", "content": blocked})
                log_lines.append(f"[step {step}] submit quality guard rejected")
                continue
            exit_status = "Submitted"
            message = f"submitted after {step} step(s)"
            break
        observation = render_observation(
            returncode=int(returncode or 0),
            output_text=truncate_text(output_text, config.max_observation_chars),
            remaining_steps=config.max_steps - step,
            patch_text=collect_repo_patch(config.repo_dir),
            issue_text=config.issue_text,
            remaining_wall=remaining_wall,
        )
        messages.append({"role": "user", "content": observation})
        patch_now = collect_repo_patch(config.repo_dir)
        if patch_now.strip():
            steps_since_patch += 1
        else:
            steps_since_patch = 0
            runtime_verified = False
        submit_after = (
            _SUBMIT_AFTER_PATCH_STEPS_VERIFIED if runtime_verified else _SUBMIT_AFTER_PATCH_STEPS
        )
        if (
            patch_now.strip()
            and not submit_ready_nudge_sent
            and steps_since_patch >= submit_after
            and not submit_readiness_light(config.repo_dir, patch_now)
            and patch_passes_runtime(config.repo_dir, patch_now, config.issue_text)
        ):
            submit_ready_nudge_sent = True
            messages.append({"role": "user", "content": _submit_ready_nudge_message(remaining_wall)})
            log_lines.append(f"[step {step}] submit-ready nudge sent")
        if (
            not collect_repo_patch(config.repo_dir).strip()
            and not write_pressure_sent
            and 0 < remaining_wall <= _WRITE_NUDGE_WALL_SECONDS
        ):
            write_pressure_sent = True
            messages.append({"role": "user", "content": _time_pressure_write_message(remaining_wall)})
            log_lines.append(f"[step {step}] time-pressure write nudge sent")
        if (
            collect_repo_patch(config.repo_dir).strip()
            and not submit_pressure_sent
            and 0 < remaining_wall <= _SUBMIT_NUDGE_WALL_SECONDS
        ):
            submit_pressure_sent = True
            messages.append({"role": "user", "content": _time_pressure_submit_message(remaining_wall)})
            log_lines.append(f"[step {step}] time-pressure submit nudge sent")
        if _should_send_no_patch_nudge(
            step=step,
            repo_dir=config.repo_dir,
            already_sent=no_patch_nudge_sent,
        ):
            no_patch_nudge_sent = True
            messages.append({"role": "user", "content": _no_patch_nudge_message(step)})
            log_lines.append(f"[step {step}] no-patch progress nudge sent")
        if _should_send_strong_write_nudge(
            step=step,
            repo_dir=config.repo_dir,
            already_sent=strong_write_nudge_sent,
        ):
            strong_write_nudge_sent = True
            messages.append({"role": "user", "content": _strong_write_nudge_message(step)})
            log_lines.append(f"[step {step}] strong write nudge sent")

    patch = collect_repo_patch(config.repo_dir)
    logs = truncate_text("\n".join(log_lines), config.max_log_chars)
    return AgentOutcome(
        success=bool(patch.strip()),
        patch=patch,
        logs=logs,
        steps=model.calls,
        cost=None,
        message=message,
        exit_status=exit_status,
        transcript=messages,
    )


def _is_submission(output_text: str, returncode) -> bool:
    lines = output_text.lstrip().splitlines()
    return bool(lines) and lines[0].strip() == COMPLETION_SENTINEL and not returncode


def _parse_single_command(reply: str) -> str | None:
    strict = [item.strip() for item in _ACTION_BLOCK_RE.findall(reply or "") if item.strip()]
    if len(strict) == 1:
        return strict[0]
    if strict:
        return None

    fenced = [item.strip() for item in _ANY_FENCE_RE.findall(reply or "") if item.strip()]
    if len(fenced) == 1:
        return fenced[0]
    if fenced:
        return None

    prompted = [item.strip() for item in _DOLLAR_LINE_RE.findall(reply or "") if item.strip()]
    return prompted[0] if len(prompted) == 1 else None


def _empty_submit_guard_message() -> str:
    return (
        "[Submit rejected: the repository has no changes on disk.]\n\n"
        "Create or modify one real source file for this task, then submit again "
        f"with `echo {COMPLETION_SENTINEL}`."
    )


def _no_patch_nudge_message(step: int) -> str:
    return (
        f"[Progress check: step {step} and the working tree is still empty.]\n\n"
        "You have enough context to act. Use the next command to create or modify "
        "the source file that owns the main requested behavior. Implement reachable "
        "code for the core requirement before doing more broad exploration."
    )


def _strong_write_nudge_message(step: int) -> str:
    return (
        f"[Urgent: step {step} and the repository still has no changes.]\n\n"
        "Stop all exploration. Your next command MUST write the owning source file "
        "(heredoc or sed -i). Further read-only commands will be rejected."
    )


def _time_pressure_write_message(remaining_wall: float) -> str:
    return (
        f"[Time pressure: about {remaining_wall:.0f}s remain and the working tree is still empty.]\n\n"
        "Write the minimal correct fix in the file the task names now. One focused edit, then verify syntax."
    )


def _time_pressure_submit_message(remaining_wall: float) -> str:
    return (
        f"[Time pressure: about {remaining_wall:.0f}s remain and you already have a diff.]\n\n"
        "Run a quick syntax check on edited files. If it passes, submit immediately with "
        f"`echo {COMPLETION_SENTINEL}` — do not explore further."
    )


def _submit_ready_nudge_message(remaining_wall: float) -> str:
    return (
        "[Submit-ready: syntax and runtime smoke checks passed.]\n\n"
        "Verify every acceptance checklist item against your changes. If complete, submit now "
        f"with `echo {COMPLETION_SENTINEL}`. Avoid further read-only exploration."
    )


def _post_patch_read_message(command: str) -> str:
    return (
        "[Post-patch read rejected: you already have a diff on disk.]\n\n"
        f"Rejected read-only command: {command[:240]}\n\n"
        "Run a runtime smoke check (`python3 -c '...'` or repo tests), fix any failures, "
        f"then submit with `echo {COMPLETION_SENTINEL}`."
    )


def _should_reject_post_patch_read_only(
    *,
    command: str,
    repo_dir: str,
    remaining_wall: float,
    steps_since_patch: int,
    runtime_verified: bool,
) -> bool:
    if not collect_repo_patch(repo_dir).strip():
        return False
    if command_is_runtime_verify(command):
        return False
    if steps_since_patch < 1 or remaining_wall <= _POST_PATCH_READ_REJECT_WALL:
        return False
    if submit_readiness_light(repo_dir, collect_repo_patch(repo_dir)):
        return False
    if not runtime_verified:
        return False
    return _is_obvious_read_only_command(command)


def _should_send_no_patch_nudge(
    *,
    step: int,
    repo_dir: str,
    already_sent: bool,
) -> bool:
    if already_sent or collect_repo_patch(repo_dir).strip():
        return False
    return step >= _NO_PATCH_NUDGE_STEP


def _should_send_strong_write_nudge(
    *,
    step: int,
    repo_dir: str,
    already_sent: bool,
) -> bool:
    if already_sent or collect_repo_patch(repo_dir).strip():
        return False
    return step >= _STRONG_WRITE_NUDGE_STEP


def _should_reject_read_only_command(
    *,
    command: str,
    repo_dir: str,
    step: int,
    no_patch_nudge_sent: bool,
    strong_write_nudge_sent: bool,
) -> bool:
    if collect_repo_patch(repo_dir).strip():
        return False
    if not strong_write_nudge_sent or step < _READ_REJECT_AFTER_STEP:
        return False
    return _is_obvious_read_only_command(command)


def _is_obvious_read_only_command(command: str) -> bool:
    stripped = (command or "").strip()
    if not stripped or _command_has_write_operator(stripped):
        return False
    return bool(_READ_ONLY_COMMAND_RE.match(stripped))


def _command_has_write_operator(command: str) -> bool:
    lowered = command.lower()
    return any(
        marker in lowered
        for marker in (
            " >",
            ">>",
            "sed -i",
            "tee ",
            "touch ",
            "mv ",
            "cp ",
            "cat <<",
        )
    )


def _read_budget_message(command: str) -> str:
    return (
        "[Read budget exhausted: command not run because the repository still has "
        "no changes on disk.]\n\n"
        f"Rejected read-only command: {command[:240]}\n\n"
        "Use the next command to edit or create the source file that owns the main "
        "requested behavior. If you need a multi-line edit, use a heredoc or a "
        "short script that writes the target file."
    )


def _cap_messages(messages: list, *, max_chars: int, step: int = 0) -> list:
    compact_limit = (
        _COMPACT_MESSAGE_CHARS_LATE if step >= _LATE_COMPACT_FROM_STEP else _COMPACT_MESSAGE_CHARS
    )
    if max_chars <= 0 or _messages_chars(messages) <= max_chars or len(messages) <= 2:
        return list(messages)

    pinned = list(messages[:2])
    rest = list(messages[2:])
    recent_count = min(len(rest), _RECENT_MESSAGE_COUNT)
    older = rest[:-recent_count] if recent_count else rest
    recent = rest[-recent_count:] if recent_count else []
    capped = pinned + [_compact_message(item, compact_limit) for item in older] + recent
    if _messages_chars(capped) <= max_chars:
        return capped

    compacted_tail = [_compact_message(item, _MIN_COMPACT_MESSAGE_CHARS) for item in capped[2:]]
    capped = pinned + compacted_tail
    while len(capped) > 6 and _messages_chars(capped) > max_chars:
        capped = pinned + capped[3:]
    return capped


def _messages_chars(messages: list) -> int:
    return sum(len(str(item.get("role", ""))) + len(str(item.get("content", ""))) for item in messages)


def _compact_message(message: dict, limit: int) -> dict:
    content = str(message.get("content") or "")
    if len(content) <= limit:
        return dict(message)
    return {**message, "content": _compact_text(content, limit)}


def _compact_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n[... compacted older turn ...]\n"
    room = max(1, limit - len(marker))
    head = max(1, room // 2)
    tail = max(1, room - head)
    return text[:head] + marker + text[-tail:]
