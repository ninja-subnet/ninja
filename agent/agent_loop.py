"""The agent step loop: query the model, run one bash action, feed the
observation back, finish when the agent echoes the completion sentinel.
Uses a text-based action format.
"""

from __future__ import annotations

import os

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
_ACE_MAX_ADDED_LINES = 15
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



def _guard_syntax_count(repo, patch_text):
    import py_compile
    bad, seen = 0, set()
    for line in (patch_text or "").splitlines():
        if line.startswith("+++ b/"):
            rel = line[6:].strip()
            if rel in seen or not rel.endswith(".py"):
                continue
            seen.add(rel)
            try:
                py_compile.compile(os.path.join(repo, rel), doraise=True)
            except py_compile.PyCompileError:
                bad += 1
            except (OSError, ValueError):
                pass
    return bad


def _guard_snapshot(repo, command, patch_text):
    files = set()
    for line in (patch_text or "").splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip())
    for tok in re.findall(r"[\w./-]+\.\w+", command or ""):
        if os.path.isfile(os.path.join(repo, tok)):
            files.add(tok)
    snap = {}
    for rel in files:
        try:
            with open(os.path.join(repo, rel), "rb") as fh:
                snap[rel] = fh.read()
        except OSError:
            snap[rel] = None
    return snap


def _guard_restore(repo, snap):
    import tempfile
    for rel, data in snap.items():
        p = os.path.join(repo, rel)
        try:
            if data is None:
                if os.path.isfile(p):
                    os.remove(p)
            else:
                d = os.path.dirname(p) or "."
                fd, tmp = tempfile.mkstemp(dir=d)
                with os.fdopen(fd, "wb") as h:
                    h.write(data)
                os.replace(tmp, p)
        except OSError:
            pass


def _ace_ready(issue_text, patch_text):
    """uid3's ace condition proxy: the FIRST code symbol named in the issue appears in the
    added lines. When a runtime-verified patch already covers it, we are at the point uid3
    would submit - and adding MORE past here is what breaks 0.9-scored solves down to 0.25."""
    m = re.search(r"`([A-Za-z_][\w.]*)`", issue_text or "")
    if not m:
        return True
    sym = m.group(1).split(".")[-1]
    if len(sym) < 3:
        return True
    return _primary_symbol_in_scope(sym, patch_text or "")


def _primary_symbol_in_scope(sym, patch_text):
    """Scan added lines for the symbol; for a MINIMAL patch (<= _ACE_MAX_ADDED_LINES) also
    scan the hunk context / @@ funcname header. The most common true-ace is a tiny in-place
    edit to the BODY of the named function - the added '+' lines never repeat the function's
    own name, so an added-only scan misses it and the ace-protection never fires -> over-work
    -> 0.9->0.25. Widening to context ONLY for small patches catches the in-place ace without
    matching sprawling multi-file over-work."""
    word = re.compile(r"\b" + re.escape(sym) + r"\b")
    added, hunk = [], []
    for line in (patch_text or "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith(" ") or line.startswith("@@"):
            hunk.append(line)
    if word.search("\n".join(added)):
        return True
    return len(added) <= _ACE_MAX_ADDED_LINES and bool(word.search("\n".join(hunk)))


def _ace_submit_message(remaining_wall):
    return (
        "[Verified working fix in place.] You have a runtime-verified patch implementing the "
        "primary requirement. On tasks where a small correct fix already works, ADDING more code "
        "usually LOWERS the score by disturbing behaviour that was already correct. Submit NOW with "
        f"`echo {COMPLETION_SENTINEL}` UNLESS a specific assertion in the preloaded test is clearly "
        "still failing - if so, implement ONLY that one thing, then submit. Do not broaden the patch."
    )


def _revert_added_file(repo, rel):
    import subprocess
    try:
        r = subprocess.run(["git", "-C", repo, "checkout", "HEAD", "--", rel],
                           capture_output=True, timeout=15, check=False)
        if r.returncode != 0:
            try:
                os.remove(os.path.join(repo, rel))
            except OSError:
                pass
    except (OSError, subprocess.SubprocessError):
        pass


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
    submit_nudge_count = 0
    runtime_verified = False
    _safe_snap = None
    _safe_files = set()
    _safe_patch = ""

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
            _g_pre_patch = collect_repo_patch(config.repo_dir)
            _g_pre_syn = _guard_syntax_count(config.repo_dir, _g_pre_patch) if _g_pre_patch.strip() else 0
            _g_snap = _guard_snapshot(config.repo_dir, command, _g_pre_patch)
            result = execute_command(command, cwd=config.repo_dir, timeout=config.command_timeout)
            output_text = result.get("output") or ""
            returncode = result.get("returncode")
            _g_post_patch = collect_repo_patch(config.repo_dir)
            if _g_post_patch.strip():
                _g_post_cnt = _guard_syntax_count(config.repo_dir, _g_post_patch)
                if _g_post_cnt > _g_pre_syn:
                    _guard_restore(config.repo_dir, _g_snap)
                    output_text = ("[Edit auto-reverted: it left a NEW syntax error ("
                                   + "%d file(s) now fail to compile" % _g_post_cnt
                                   + "). The tree was restored to before this command. Re-apply the "
                                   "change so every file stays syntactically valid.]")
                    returncode = 1
            _post = collect_repo_patch(config.repo_dir)
            if not returncode and command_is_runtime_verify(command):
                # EXPLICIT runtime-verify (unchanged cand_v13 behavior): the agent
                # CHOSE to run a verify command, so a passing smoke both captures a
                # salvage snapshot AND arms the submit-side signals via runtime_verified
                # (ace-nudge, submit_after=2, adaptive coverage bar).
                if _post.strip() and patch_passes_runtime(
                    config.repo_dir, _post, config.issue_text
                ):
                    runtime_verified = True
                    _safe_snap = _guard_snapshot(config.repo_dir, "", _post)
                    _safe_files = {ln[6:].strip() for ln in _post.splitlines()
                                   if ln.startswith("+++ b/")}
                    _safe_patch = _post
            elif (
                _post.strip()
                and _post != _safe_patch
                and not submit_readiness_light(config.repo_dir, _post)
                and patch_passes_runtime(config.repo_dir, _post, config.issue_text)
            ):
                # PROACTIVE salvage snapshot (protection WITHOUT pressure): the tree
                # reached a NEW non-empty lint-clean state that passes the runtime smoke,
                # but the agent did NOT run a verify command. Capture a restore point for
                # the wall-safe salvage so tasks the agent breaks before EVER self-verifying
                # still fall back to a good state at the wall (cuts the uncaught 0.65->0.00
                # catastrophic breaks). Crucially this does NOT set runtime_verified: the
                # ace-nudge, submit_after timing, and adaptive submit bar stay exactly as in
                # cand_v13, so it adds ZERO submit pressure and cannot force premature submit
                # on hard tasks (avoids the cand_v14 recovery-collapse trap). The _post !=
                # _safe_patch guard bounds the smoke to distinct new tree states only.
                _safe_snap = _guard_snapshot(config.repo_dir, "", _post)
                _safe_files = {ln[6:].strip() for ln in _post.splitlines()
                               if ln.startswith("+++ b/")}
                _safe_patch = _post
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
        _ace = bool(patch_now.strip()) and runtime_verified and _ace_ready(config.issue_text, patch_now)
        if (
            patch_now.strip()
            and (submit_nudge_count == 0 or (_ace and submit_nudge_count < 3))
            and steps_since_patch >= (1 if _ace else submit_after)
            and not submit_readiness_light(config.repo_dir, patch_now)
            and patch_passes_runtime(config.repo_dir, patch_now, config.issue_text)
        ):
            submit_nudge_count += 1
            _msg = _ace_submit_message(remaining_wall) if _ace else _submit_ready_nudge_message(remaining_wall)
            messages.append({"role": "user", "content": _msg})
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

    # WALL-SAFE SALVAGE: if the final tree is broken/empty but a runtime-verified
    # good state was captured earlier, restore it. Over-work (or a wall kill mid-edit)
    # that leaves a broken tree is the catastrophic 0.00-vs-uid3-ace loss; never ship
    # worse than a verified-good patch. Do-no-harm: fires ONLY when the final is
    # actually broken, so a genuinely-working refinement is never reverted.
    if _safe_snap is not None:
        _cur = collect_repo_patch(config.repo_dir)
        _broken = (not _cur.strip()) or _guard_syntax_count(config.repo_dir, _cur) > 0 \
            or not patch_passes_runtime(config.repo_dir, _cur, config.issue_text)
        if _broken:
            _cur_files = {ln[6:].strip() for ln in _cur.splitlines() if ln.startswith("+++ b/")}
            for _rel in _cur_files - _safe_files:
                _revert_added_file(config.repo_dir, _rel)
            _guard_restore(config.repo_dir, _safe_snap)
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
