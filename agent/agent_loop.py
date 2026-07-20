"""The agent step loop: query the model, run one bash action, feed the
observation back, finish when the agent echoes the completion sentinel.
Uses a text-based action format.
"""

import re
import time
from dataclasses import dataclass, field

from .audit import (
    audit_has_budget,
    audit_reply_flags_gaps,
    build_audit_message,
    build_correctness_audit_message,
    correctness_enforcement_message,
    correctness_reply_flags_issues,
    enforcement_message,
)
from .environment import execute_command, truncate_text
from .model import ChatModel, ContextLengthError, ModelQueryError, messages_chars
from .prompts import (
    COMPLETION_SENTINEL,
    SYSTEM_PROMPT,
    build_task_prompt,
    format_help_message,
    render_observation,
    time_pressure_note,
)
from .change_echo import FUSE_NOTE, PatchSnapshotter, build_change_echo
from .repo_diff import collect_repo_patch

_ACTION_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n?```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.DOTALL)
_DOLLAR_LINE_RE = re.compile(r"(?m)^\s*\$[ \t]+(\S.*?)\s*$")
# The solver streams its chain-of-thought inline and closes it with this tag
# (the opening <think> is often absent -- the chat template injects it). Code
# fenced inside the thinking is quotation, not an action; only the text after
# the tag can carry the command.
_THINK_CLOSE_TAG = "</think>"
_OPEN_BASH_FENCE_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n")
_HEREDOC_DELIM_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
_READ_ONLY_COMMAND_RE = re.compile(
    r"^\s*(?:cat|nl|head|tail|less|more|grep|rg|find|ls|tree|wc)\b",
    re.I,
)
_MAX_FORMAT_RETRIES = 3
_NO_PATCH_NUDGE_STEP = 4
# The no-patch nudge stays at step 4 (advisory early-edit pressure), but the
# HARD read rejection additionally waits out 30% of the wall clock: at step 4 a
# run still has ~90% of its budget, and rejecting the read that would ground
# the first edit converts unknown APIs into invented ones (duel-20260716 RCA:
# both invented-API losses were born on the reply after a rejection fired
# under 15% into the budget).
_READ_GATE_MIN_ELAPSED_FRACTION = 0.3
# Wall-clock pressure: one note each at fixed budget fractions plus a final
# window, denominated in real seconds (prompts._TIME_NOTES). Fractions, not
# projected commands-left: the mean-step projection under-warns exactly the
# runs whose late steps run several times the early mean, and those runs die
# mid-edit having never been warned.
_TIME_NOTE_HALF_FRACTION = 0.5
_TIME_NOTE_LATE_FRACTION = 0.8
_FINAL_WINDOW_SECONDS = 60.0
_RECENT_MESSAGE_COUNT = 8
_COMPACT_MESSAGE_CHARS = 1200
_MIN_COMPACT_MESSAGE_CHARS = 600
# One overflow-and-retry is the normal case (the error body hands us the true token
# count, so the retry is sized correctly); a second is slack for a pathological turn.
_MAX_CONTEXT_RETRIES = 2

# The served window is a HARD ceiling on prompt + requested max_tokens -- vLLM's
# --max-model-len, which the solver endpoint reports as 32768. Exceeding it is a 400,
# not a truncation. Every context bound below is derived from these two numbers.
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_TOKENS = 4096


@dataclass
class AgentRunConfig:
    repo_dir: str
    model_name: str
    base_url: str
    auth_token: str
    # The step cap is a backstop, not the pacer: wrap-up pressure comes from
    # ``_commands_left``, which tracks whichever limit actually binds. The
    # canonical default lives in agent.py (AGENT_MAX_STEPS); sized so the wall
    # clock always runs out first on a healthy run.
    max_steps: int = 500
    command_timeout: int = 15
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_observation_chars: int = 16000
    max_log_chars: int = 260000
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    wall_clock_limit: float = 0.0


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


def run_agent_loop(*, config: AgentRunConfig, task: str, task_updater=None) -> AgentOutcome:
    """``task_updater``: optional zero-arg callable polled once per step. When it
    returns a non-empty string, that string REPLACES the task message (index 1,
    which ``_cap_messages`` pins) -- the model is stateless, so from its next call
    onward the task simply always carried it. ``agent/prefetch.py`` uses this to
    swap in its background file re-rank without ever blocking the loop; None
    (the default) leaves the loop exactly as it was."""
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
    # What the prompt may occupy once the reply's reservation is set aside: the server
    # rejects prompt + max_tokens > max_model_len, so this is the real compaction line.
    prompt_token_budget = max(0, config.max_model_len - config.max_tokens)
    started = time.monotonic()
    log_lines: list = []
    exit_status = "LimitsExceeded"
    message = f"step limit of {config.max_steps} reached"
    format_retries = 0
    no_patch_nudge_sent = False
    audit_sent = False
    awaiting_audit_reply = False
    audit_enforced = False
    correctness_sent = False
    correctness_enforced = False
    last_command = None
    last_result = None
    repeat_count = 0
    snapshotter = PatchSnapshotter()
    fuse_noted = False
    time_notes_sent: set = set()

    for step in range(1, max(1, config.max_steps) + 1):
        if 0 < config.wall_clock_limit <= time.monotonic() - started:
            exit_status = "TimeExceeded"
            message = f"wall clock limit of {config.wall_clock_limit:.0f}s reached"
            break
        if task_updater is not None:
            try:
                updated_task = task_updater()
            except Exception as exc:  # noqa: BLE001 - a hint is never worth a crash
                updated_task = ""
                log_lines.append(f"[step {step}] task update failed: {exc}")
            if updated_task:
                messages[1] = {"role": "user", "content": updated_task}
                log_lines.append(f"[step {step}] task context updated (re-ranked files)")
        try:
            reply, messages = _query_within_window(
                model=model, messages=messages, max_prompt_tokens=prompt_token_budget,
                log_lines=log_lines, step=step,
            )
        except ModelQueryError as exc:
            exit_status = "ModelError"
            message = str(exc)
            log_lines.append(f"[step {step}] model error: {exc}")
            break
        messages.append({"role": "assistant", "content": reply})
        log_lines.append(f"[step {step}] assistant:\n{reply}")

        command = _parse_single_command(reply)
        if command is None:
            if awaiting_audit_reply:
                # Fail open: the audit must never cost a scoreable patch. A reply
                # with no command (mapping-only, token-cap spiral, off-format)
                # accepts the submit the audit interrupted.
                exit_status = "Submitted"
                message = f"submitted after {step} step(s); audit reply carried no command"
                log_lines.append(f"[step {step}] audit fail-open: accepting the prior submit")
                break
            format_retries += 1
            if format_retries > _MAX_FORMAT_RETRIES:
                exit_status = "FormatError"
                message = "model kept replying without exactly one bash code block"
                break
            messages.append({"role": "user", "content": format_help_message()})
            log_lines.append(f"[step {step}] format retry {format_retries}")
            continue
        format_retries = 0
        was_audit_reply = awaiting_audit_reply
        awaiting_audit_reply = False

        change_echo = ""
        if _should_reject_read_only_command(
            command=command,
            repo_dir=config.repo_dir,
            no_patch_nudge_sent=no_patch_nudge_sent,
            config=config,
            started=started,
        ):
            output_text = _read_budget_message(command)
            returncode = 2
            log_lines.append(f"[step {step}] read-only command rejected after no-patch nudge")
        else:
            pre_patch = (
                None if _is_obvious_read_only_command(command)
                else snapshotter.snapshot(config.repo_dir)
            )
            result = execute_command(command, cwd=config.repo_dir, timeout=config.command_timeout)
            output_text = result.get("output") or ""
            returncode = result.get("returncode")
            if pre_patch is not None:
                post_patch = snapshotter.snapshot(config.repo_dir)
                if post_patch is not None:
                    change_echo = build_change_echo(
                        pre_patch,
                        post_patch,
                        write_shaped=_command_has_write_operator(command),
                        repo_dir=config.repo_dir,
                    )
            # Whichever snapshot tripped the fuse (pre or post), tell the model
            # exactly once -- silence would read as "nothing changed" under the
            # echo contract the system prompt establishes.
            if snapshotter.fused and not fuse_noted:
                fuse_noted = True
                change_echo = FUSE_NOTE
                log_lines.append(f"[step {step}] change echo disabled: slow repo snapshot")
        log_lines.append(f"[step {step}] $ {command}\n{truncate_text(output_text, 2000)}")
        if _is_submission(output_text, returncode):
            patch_text = collect_repo_patch(config.repo_dir)
            if not patch_text.strip():
                messages.append({"role": "user", "content": _empty_submit_guard_message()})
                log_lines.append(f"[step {step}] empty submit rejected")
                if not no_patch_nudge_sent:
                    no_patch_nudge_sent = True
                    messages.append({"role": "user", "content": _no_patch_nudge_message(step)})
                continue
            reply_body = reply.split(_THINK_CLOSE_TAG, 1)[-1]
            _audit_budget = audit_has_budget(
                commands_left=_commands_left(config=config, step=step, started=started),
                time_left_s=(
                    config.wall_clock_limit - (time.monotonic() - started)
                    if config.wall_clock_limit > 0
                    else None
                ),
            )
            # Stage 1 -- completeness audit: a reply whose own checklist lists a
            # NOT COVERED requirement contradicts its submit. Reject once; re-arming
            # awaiting_audit_reply keeps the fail-open (a next reply with no command
            # still accepts this submit unchanged).
            if (
                was_audit_reply
                and not correctness_sent
                and not audit_enforced
                and audit_reply_flags_gaps(reply_body)
            ):
                audit_enforced = True
                awaiting_audit_reply = True
                messages.append({"role": "user", "content": enforcement_message()})
                log_lines.append(f"[step {step}] audit lists gaps; submit rejected once")
                continue
            if not audit_sent and _audit_budget:
                audit_sent = True
                awaiting_audit_reply = True
                messages.append({
                    "role": "user",
                    "content": build_audit_message(
                        task_message=str(messages[1].get("content") or ""),
                        patch_text=patch_text,
                    ),
                })
                log_lines.append(f"[step {step}] pre-submit completeness audit injected")
                continue
            # Stage 2 -- correctness audit (only after completeness has run): the
            # same in-context turn, now checking the patch compiles/wires. A reply
            # whose own audit reports a compile/symbol/wiring problem yet submits
            # anyway is rejected once, under the same fail-open contract.
            if (
                was_audit_reply
                and correctness_sent
                and not correctness_enforced
                and correctness_reply_flags_issues(reply_body)
            ):
                correctness_enforced = True
                awaiting_audit_reply = True
                messages.append({"role": "user", "content": correctness_enforcement_message()})
                log_lines.append(
                    f"[step {step}] correctness audit lists issues; submit rejected once"
                )
                continue
            if audit_sent and not correctness_sent and _audit_budget:
                correctness_sent = True
                awaiting_audit_reply = True
                messages.append({
                    "role": "user",
                    "content": build_correctness_audit_message(patch_text=patch_text),
                })
                log_lines.append(f"[step {step}] pre-submit correctness audit injected")
                continue
            exit_status = "Submitted"
            message = f"submitted after {step} step(s)"
            break
        if command == last_command and (returncode, output_text) == last_result:
            repeat_count += 1
        else:
            repeat_count = 0
        last_command, last_result = command, (returncode, output_text)
        observation = render_observation(
            returncode=int(returncode or 0),
            output_text=truncate_text(output_text, config.max_observation_chars),
            time_note=_time_pressure_note(config=config, started=started, sent=time_notes_sent),
        )
        if change_echo:
            observation += "\n" + change_echo
            log_lines.append(f"[step {step}] change echo: {len(change_echo)} chars")
        if repeat_count:
            observation += "\n" + _repeat_command_warning(repeat_count + 1)
            log_lines.append(f"[step {step}] repeat-command warning ({repeat_count + 1}x)")
        messages.append({"role": "user", "content": observation})
        if _should_send_no_patch_nudge(
            step=step,
            repo_dir=config.repo_dir,
            already_sent=no_patch_nudge_sent,
        ):
            no_patch_nudge_sent = True
            messages.append({"role": "user", "content": _no_patch_nudge_message(step)})
            log_lines.append(f"[step {step}] no-patch progress nudge sent")

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
    reply = reply or ""
    if _THINK_CLOSE_TAG in reply:
        # Thinking regularly quotes code it is planning or second-guessing;
        # those fences collide with the real action block (and a block the
        # thinking just declared wrong must never run), so the thinking is
        # cut away rather than parsed around.
        reply = reply.split(_THINK_CLOSE_TAG, 1)[1]

    strict = [item.strip() for item in _ACTION_BLOCK_RE.findall(reply) if item.strip()]
    if len(strict) == 1:
        return strict[0]
    if strict:
        return None

    fenced = [item.strip() for item in _ANY_FENCE_RE.findall(reply) if item.strip()]
    if len(fenced) == 1:
        return fenced[0]
    if fenced:
        return None

    recovered = _unterminated_heredoc_command(reply)
    if recovered is not None:
        return recovered

    prompted = [item.strip() for item in _DOLLAR_LINE_RE.findall(reply) if item.strip()]
    return prompted[0] if len(prompted) == 1 else None


def _unterminated_heredoc_command(reply: str) -> str | None:
    """Recover a heredoc command whose closing ``` never arrived.

    Long file-writing replies routinely stop right after the heredoc
    terminator -- to the model that line already reads as the end of the
    block -- leaving one opening fence and no closing one. The command is
    recovered ONLY when every heredoc it opens is terminated: a reply
    truncated mid-heredoc (token cap, runaway repetition) must keep failing
    to parse, because executing it would silently write a partial file that
    still lands as a successful edit. Non-heredoc unterminated fences stay
    unparsed for the same reason -- nothing proves the command ended where
    the reply did."""
    if reply.count("```") != 1:
        return None
    fence = _OPEN_BASH_FENCE_RE.search(reply)
    if fence is None:
        return None
    body = reply[fence.end():].strip()
    if not body:
        return None
    delimiters = [d for _quote, d in _HEREDOC_DELIM_RE.findall(body)]
    if not delimiters:
        return None
    lines = {line.strip() for line in body.splitlines()}
    if all(d in lines for d in delimiters):
        return body
    return None


def _empty_submit_guard_message() -> str:
    return (
        "[Submit rejected: the repository has no changes on disk.]\n\n"
        "Create or modify one real source file for this task, then submit again "
        f"with `echo {COMPLETION_SENTINEL}`."
    )


def _time_pressure_note(*, config: AgentRunConfig, started: float, sent: set) -> str:
    """The wall-clock note due this turn, or "". Each phase fires once; when one
    slow step crosses several thresholds at once only the most urgent note is
    shown (the earlier ones are marked sent, not queued -- a stale "half" note
    after the final window would read as MORE time, not less)."""
    if config.wall_clock_limit <= 0:
        return ""
    elapsed = time.monotonic() - started
    time_left = config.wall_clock_limit - elapsed
    due = []
    if elapsed >= config.wall_clock_limit * _TIME_NOTE_HALF_FRACTION:
        due.append("half")
    if elapsed >= config.wall_clock_limit * _TIME_NOTE_LATE_FRACTION:
        due.append("late")
    if time_left <= _FINAL_WINDOW_SECONDS:
        due.append("final")
    fresh = [phase for phase in due if phase not in sent]
    sent.update(due)
    if not fresh:
        return ""
    return time_pressure_note(phase=fresh[-1], time_left_s=time_left)


def _commands_left(*, config: AgentRunConfig, step: int, started: float) -> int:
    """Commands the run can still fit -- the step cap or the wall clock,
    whichever binds first -- used ONLY to size the pre-submit audit's budget
    (``audit_has_budget``). Time is projected onto commands at the run's mean
    observed pace; good enough for a budget estimate, but the model-facing
    wrap-up pressure is wall-clock denominated instead (``_time_pressure_note``)
    because the mean under-warns runs whose late steps dwarf the early ones."""
    steps_left = config.max_steps - step
    if config.wall_clock_limit <= 0:
        return steps_left
    elapsed = time.monotonic() - started
    time_left = config.wall_clock_limit - elapsed
    if time_left <= 0:
        return 0
    mean_step_s = elapsed / max(1, step)
    if mean_step_s <= 0:
        return steps_left
    return max(0, min(steps_left, int(time_left / mean_step_s)))


def _repeat_command_warning(times: int) -> str:
    """Break command-repetition fixed points. The solver sometimes re-issues the
    exact same command turn after turn (observed: 11 consecutive identical greps
    whose empty result never changed) because the identical observation reproduces
    the identical reply. The escalating count keeps each warning textually unique,
    so the warning itself cannot become part of a new fixed point. Only fires when
    the RESULT is also identical -- re-running a build or flaky test that produces
    different output is legitimate and stays unwarned."""
    return (
        f"[You have now run this exact command {times} times in a row and the "
        "result was identical every time. Running it again will not change "
        "anything. Take a DIFFERENT action: read a specific file, change the "
        "search pattern, or start making the edit.]"
    )


def _no_patch_nudge_message(step: int) -> str:
    return (
        f"[Progress check: step {step} and the working tree is still empty.]\n\n"
        "You have enough context to act. Use the next command to create or modify "
        "the source file that owns the main requested behavior. Implement reachable "
        "code for the core requirement before doing more broad exploration."
    )


def _should_send_no_patch_nudge(
    *,
    step: int,
    repo_dir: str,
    already_sent: bool,
) -> bool:
    if already_sent or collect_repo_patch(repo_dir).strip():
        return False
    return step >= _NO_PATCH_NUDGE_STEP


def _should_reject_read_only_command(
    *,
    command: str,
    repo_dir: str,
    no_patch_nudge_sent: bool,
    config: AgentRunConfig,
    started: float,
) -> bool:
    if not no_patch_nudge_sent or collect_repo_patch(repo_dir).strip():
        return False
    elapsed = time.monotonic() - started
    if elapsed < config.wall_clock_limit * _READ_GATE_MIN_ELAPSED_FRACTION:
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


def _query_within_window(*, model: ChatModel, messages: list, max_prompt_tokens: int,
                         log_lines: list, step: int) -> tuple:
    """Compact to fit the window, then query. An overflow is not fatal: the server tells
    us the true token count, ``model`` re-measures its chars-per-token from it, and the
    same turn is retried against a correctly-sized budget. Returns (reply, messages)."""
    for attempt in range(_MAX_CONTEXT_RETRIES + 1):
        messages = _cap_messages(
            messages, max_prompt_tokens=max_prompt_tokens, chars_per_token=model.chars_per_token
        )
        try:
            return model.query(messages), messages
        except ContextLengthError as exc:
            if attempt == _MAX_CONTEXT_RETRIES:
                raise
            log_lines.append(
                f"[step {step}] prompt overflowed the window; recompacting at "
                f"{model.chars_per_token:.2f} chars/token and retrying ({exc})"
            )
    raise AssertionError("unreachable")


def _cap_messages(messages: list, *, max_prompt_tokens: int, chars_per_token: float) -> list:
    """Hold the prompt under ``max_prompt_tokens``. We can only measure characters, so
    the token budget is converted with the ratio ``model`` measured on this run's own
    content -- a fixed char cap cannot do this, because chars-per-token swings from 2.6
    on dense code to 5.1 on prose, and the wrong end of that range overflows the window."""
    max_chars = int(max_prompt_tokens * chars_per_token)
    if max_chars <= 0 or messages_chars(messages) <= max_chars or len(messages) <= 2:
        return list(messages)

    pinned = list(messages[:2])
    rest = list(messages[2:])
    recent_count = min(len(rest), _RECENT_MESSAGE_COUNT)
    older = rest[:-recent_count] if recent_count else rest
    recent = rest[-recent_count:] if recent_count else []
    capped = pinned + [_compact_message(item, _COMPACT_MESSAGE_CHARS) for item in older] + recent
    if messages_chars(capped) <= max_chars:
        return capped

    compacted_tail = [_compact_message(item, _MIN_COMPACT_MESSAGE_CHARS) for item in capped[2:]]
    capped = pinned + compacted_tail
    while len(capped) > 6 and messages_chars(capped) > max_chars:
        capped = pinned + capped[3:]
    return capped


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
