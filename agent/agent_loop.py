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
from .criteria import extract_criteria, format_checklist
from .guards import (
    patch_acceptable,
    patch_reject_reason,
    sprawl_patch_reason,
    thin_relative_to_task_reason,
)

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
# Duel-101923: empty/trivial patches cost mean points. Nudge by step AND wall.
_NO_PATCH_NUDGE_STEP = 2
_NO_PATCH_ESCALATION_STEP = 4
_NO_PATCH_NUDGE_ELAPSED_FRACTION = 0.12
_NO_PATCH_ESCALATION_ELAPSED_FRACTION = 0.22
_READ_GATE_MIN_ELAPSED_FRACTION = 0.15
_EMPTY_TREE_HARD_GATE_FRACTION = 0.28
_TIME_NOTE_HALF_FRACTION = 0.35
_TIME_NOTE_LATE_FRACTION = 0.55
_FINAL_WINDOW_SECONDS = 100.0
# Autosubmit locks a nonempty patch before the kill. Keep the window short so
# late gap-closing still happens; require patch_acceptable so gut/munge diffs
# are not frozen in.
_AUTO_SUBMIT_SECONDS = 35.0
# Below this remaining wall, accept thin-but-nonempty rather than risk empty.
_AUTO_SUBMIT_FORCE_SECONDS = 18.0
_FINAL_WINDOW_READ_LIMIT = 2
# Reject only near-empty cosmetics while time remains; do not force large floors
# that stall real small-but-valid fixes.
_MIN_SUBSTANTIVE_CHANGED_LINES = 3
_TRIVIAL_PATCH_MIN_TIME_LEFT_S = 50.0
# One expand reject for clearly under-scoped multi-req stubs (863250 r14/r32).
_THIN_EXPAND_MIN_TIME_LEFT_S = 70.0
_RECENT_MESSAGE_COUNT = 8
_COMPACT_MESSAGE_CHARS = 1200
_MIN_COMPACT_MESSAGE_CHARS = 600
_MAX_CONTEXT_RETRIES = 2

DEFAULT_MAX_MODEL_LEN = 32768
# Duel-863250: max_tokens=1536 correlated with thinner incomplete patches
# (mean lines 246→181, Δ +0.037→-0.068). Restore the 645807-era budget.
DEFAULT_MAX_TOKENS = 2048
_EMPTY_TREE_MAX_TOKENS = 1536
# Only reject clearly pathological actionable bodies (full-file paste spirals).
_MAX_ACTIONABLE_REPLY_CHARS = 14000
_CHECKLIST_MIN_STEP = 3
_CHECKLIST_MIN_ITEMS = 2
_CHECKLIST_MIN_TIME_LEFT_S = 80.0


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
    # Raw issue text for checklist / patch-hygiene guards (not the full prompt).
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
    # Prompt budget is recomputed each step from the live max_completion_tokens
    # (empty-tree runs use a tighter completion cap, which frees prompt room).
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
    empty_escalation_sent = False
    submit_now_sent = False
    final_window_reads = 0
    checklist_sent = False
    hygiene_reject_sent = False
    thin_expand_sent = False
    sprawl_nudge_sent = False

    for step in range(1, max(1, config.max_steps) + 1):
        elapsed = time.monotonic() - started
        time_left = (
            config.wall_clock_limit - elapsed if config.wall_clock_limit > 0 else None
        )
        if 0 < config.wall_clock_limit <= elapsed:
            exit_status = "TimeExceeded"
            message = f"wall clock limit of {config.wall_clock_limit:.0f}s reached"
            break
        # Lock in a scoreable nonempty patch before the kill path. Prefer not to
        # freeze gut/munge or obviously thin multi-req stubs while a little wall
        # remains; force-accept near the deadline to avoid empty zeros.
        on_disk = collect_repo_patch(config.repo_dir)
        if (
            time_left is not None
            and time_left <= _AUTO_SUBMIT_SECONDS
            and _patch_is_substantive(on_disk)
            and patch_acceptable(on_disk)
            and (
                time_left <= _AUTO_SUBMIT_FORCE_SECONDS
                or thin_relative_to_task_reason(config.issue_text, on_disk) is None
            )
        ):
            exit_status = "Submitted"
            message = (
                f"auto-submitted with {time_left:.0f}s wall left after {step - 1} "
                "step(s); nonempty on-disk patch"
            )
            log_lines.append(f"[step {step}] {message}")
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
        # Tighten the completion budget while the tree is empty so explore/rewrite
        # spirals cannot monopolize the wall; restore the configured cap once a
        # real patch exists and larger regional edits are useful.
        if collect_repo_patch(config.repo_dir).strip():
            model.max_completion_tokens = config.max_tokens
        else:
            model.max_completion_tokens = min(config.max_tokens, _EMPTY_TREE_MAX_TOKENS)
        step_prompt_budget = max(0, config.max_model_len - model.max_completion_tokens)
        try:
            reply, messages = _query_within_window(
                model=model, messages=messages, max_prompt_tokens=step_prompt_budget,
                log_lines=log_lines, step=step,
            )
        except ModelQueryError as exc:
            exit_status = "ModelError"
            message = str(exc)
            log_lines.append(f"[step {step}] model error: {exc}")
            break
        messages.append({"role": "assistant", "content": reply})
        log_lines.append(f"[step {step}] assistant:\n{reply}")

        if model.last_reply_hit_token_cap() or _actionable_reply_too_long(reply):
            # Truncated / oversized full-file rewrites burned wall clock in
            # duel-101923 and still hit 4k-6k completion bursts in 645807.
            # Refuse to parse/run; demand a short regional edit instead.
            messages.append({"role": "user", "content": _truncated_reply_message()})
            log_lines.append(
                f"[step {step}] oversized/truncated reply "
                f"(completion_tokens={model.last_completion_tokens}, "
                f"finish_reason={model.last_finish_reason or 'size'}, "
                f"actionable_chars={_actionable_reply_chars(reply)}); "
                "demanding a smaller edit"
            )
            continue

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
        hard_empty_gate = _empty_tree_hard_gate_active(
            repo_dir=config.repo_dir,
            config=config,
            started=started,
        )
        if hard_empty_gate and not _command_has_write_operator(command):
            output_text = _empty_tree_hard_gate_message(command)
            returncode = 2
            log_lines.append(f"[step {step}] empty-tree hard gate rejected non-write command")
        elif _should_reject_read_only_command(
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
            if _should_reject_trivial_patch(
                patch_text=patch_text,
                config=config,
                started=started,
            ):
                messages.append({"role": "user", "content": _trivial_patch_guard_message(patch_text)})
                log_lines.append(
                    f"[step {step}] trivial patch rejected "
                    f"({_count_changed_lines(patch_text)} changed line(s))"
                )
                continue
            hygiene_reason = patch_reject_reason(
                issue_text=config.issue_text,
                patch_text=patch_text,
                repo_dir=config.repo_dir,
            )
            if hygiene_reason and not hygiene_reject_sent:
                time_left_now = (
                    config.wall_clock_limit - (time.monotonic() - started)
                    if config.wall_clock_limit > 0
                    else None
                )
                if time_left_now is None or time_left_now >= _TRIVIAL_PATCH_MIN_TIME_LEFT_S:
                    hygiene_reject_sent = True
                    messages.append({
                        "role": "user",
                        "content": _hygiene_reject_message(hygiene_reason),
                    })
                    log_lines.append(f"[step {step}] patch hygiene rejected: {hygiene_reason}")
                    continue
            thin_reason = thin_relative_to_task_reason(config.issue_text, patch_text)
            if thin_reason and not thin_expand_sent:
                time_left_now = (
                    config.wall_clock_limit - (time.monotonic() - started)
                    if config.wall_clock_limit > 0
                    else None
                )
                if time_left_now is None or time_left_now >= _THIN_EXPAND_MIN_TIME_LEFT_S:
                    thin_expand_sent = True
                    messages.append({
                        "role": "user",
                        "content": _thin_expand_message(thin_reason),
                    })
                    log_lines.append(f"[step {step}] thin-patch expand: {thin_reason}")
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

        # Final-window exploration tax: with a patch already on disk, further
        # reads rarely raise the judge score and often burn the submit window.
        on_disk_now = collect_repo_patch(config.repo_dir)
        has_patch = _patch_is_substantive(on_disk_now)
        in_final = (
            config.wall_clock_limit > 0
            and (config.wall_clock_limit - (time.monotonic() - started))
            <= _FINAL_WINDOW_SECONDS
        )
        if has_patch and in_final and _is_obvious_read_only_command(command):
            final_window_reads += 1
        elif has_patch and in_final and _command_has_write_operator(command):
            final_window_reads = 0
        if has_patch and in_final and final_window_reads >= _FINAL_WINDOW_READ_LIMIT:
            exit_status = "Submitted"
            message = (
                f"submitted after {step} step(s); final-window read limit with "
                "nonempty patch"
            )
            log_lines.append(f"[step {step}] {message}")
            break

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
        if has_patch and in_final and not submit_now_sent:
            submit_now_sent = True
            observation += "\n" + _submit_now_message()
            log_lines.append(f"[step {step}] submit-now pressure injected")
        if (
            has_patch
            and not sprawl_nudge_sent
            and not in_final
            and config.issue_text
        ):
            sprawl_reason = sprawl_patch_reason(config.issue_text, on_disk_now)
            if sprawl_reason:
                sprawl_nudge_sent = True
                observation += "\n" + _sprawl_nudge_message(sprawl_reason)
                log_lines.append(f"[step {step}] sprawl soft-nudge: {sprawl_reason}")
        if (
            not checklist_sent
            and has_patch
            and step >= _CHECKLIST_MIN_STEP
            and config.issue_text
        ):
            time_left_now = (
                config.wall_clock_limit - (time.monotonic() - started)
                if config.wall_clock_limit > 0
                else None
            )
            if time_left_now is None or time_left_now >= _CHECKLIST_MIN_TIME_LEFT_S:
                criteria = extract_criteria(config.issue_text)
                checklist = format_checklist(criteria)
                if checklist and len(criteria) >= _CHECKLIST_MIN_ITEMS:
                    checklist_sent = True
                    observation += (
                        "\n[Progress checklist: a nonempty patch is on disk. "
                        "Before you submit, close every still-open item below "
                        "with a real edit — do not mark covered from memory.]\n"
                        + checklist
                    )
                    log_lines.append(f"[step {step}] mid-run acceptance checklist injected")
        messages.append({"role": "user", "content": observation})
        if _should_send_no_patch_nudge(
            step=step,
            repo_dir=config.repo_dir,
            already_sent=no_patch_nudge_sent,
            config=config,
            started=started,
        ):
            no_patch_nudge_sent = True
            messages.append({"role": "user", "content": _no_patch_nudge_message(step)})
            log_lines.append(f"[step {step}] no-patch progress nudge sent")
        elif _should_send_empty_escalation(
            step=step,
            repo_dir=config.repo_dir,
            already_sent=empty_escalation_sent,
            config=config,
            started=started,
        ):
            empty_escalation_sent = True
            messages.append({"role": "user", "content": _empty_tree_escalation_message(step)})
            log_lines.append(f"[step {step}] empty-tree escalation nudge sent")

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
        f"with `echo {COMPLETION_SENTINEL}`. An empty diff scores zero."
    )


def _truncated_reply_message() -> str:
    return (
        "[Reply rejected: your previous response was truncated or far too large "
        "to execute safely, and that command was NOT run.]\n\n"
        "Do NOT rewrite a whole file. Make ONE small regional edit with "
        "`sed -i` or a short heredoc over at most ~40 lines, then continue. "
        "Long full-file rewrites burn the wall clock and score near zero when "
        "they time out unfinished."
    )


def _actionable_reply_chars(reply: str) -> int:
    text = reply or ""
    if _THINK_CLOSE_TAG in text:
        text = text.split(_THINK_CLOSE_TAG, 1)[1]
    return len(text)


def _actionable_reply_too_long(reply: str) -> bool:
    return _actionable_reply_chars(reply) > _MAX_ACTIONABLE_REPLY_CHARS


def _trivial_patch_guard_message(patch_text: str) -> str:
    n = _count_changed_lines(patch_text)
    return (
        f"[Submit rejected: the on-disk patch only changes ~{n} line(s), which "
        "is too thin for this task.]\n\n"
        "A near-empty or cosmetic diff scores near zero. Implement the core "
        "requested behavior with a real edit in the owning source file "
        "(add/replace the missing logic, not just a comment or rename), then "
        f"submit again with `echo {COMPLETION_SENTINEL}`."
    )


def _hygiene_reject_message(reason: str) -> str:
    return (
        f"[Submit rejected: {reason}.]\n\n"
        "Fix that problem with a real edit to the correct source file(s), then "
        f"submit again with `echo {COMPLETION_SENTINEL}`. Do not add scratch "
        "helper files or gut unrelated code."
    )


def _thin_expand_message(reason: str) -> str:
    return (
        f"[Submit rejected: {reason}.]\n\n"
        "Grow the patch with real behavior for the uncovered requirements — "
        "prefer the owning source file(s) the task names. Do not add cosmetic "
        "churn. When the requirements are covered, submit with "
        f"`echo {COMPLETION_SENTINEL}`."
    )


def _sprawl_nudge_message(reason: str) -> str:
    return (
        f"[Scope warning: {reason}.]\n\n"
        "This is advisory, not a hard reject. Prefer reverting unrelated files "
        "and keeping only the edits that implement the task. A tight complete "
        "patch scores higher than a sprawling partial rewrite."
    )


def _count_changed_lines(patch_text: str) -> int:
    """Count added+removed content lines in a unified diff (excludes headers)."""
    count = 0
    for line in (patch_text or "").splitlines():
        if not line or line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def _patch_is_substantive(patch_text: str) -> bool:
    return _count_changed_lines(patch_text) >= _MIN_SUBSTANTIVE_CHANGED_LINES


def _should_reject_trivial_patch(
    *,
    patch_text: str,
    config: AgentRunConfig,
    started: float,
) -> bool:
    if _patch_is_substantive(patch_text):
        return False
    if config.wall_clock_limit <= 0:
        return True
    time_left = config.wall_clock_limit - (time.monotonic() - started)
    return time_left >= _TRIVIAL_PATCH_MIN_TIME_LEFT_S


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
        "code for the core requirement before doing more broad exploration. "
        "Finishing with an empty diff scores zero."
    )


def _empty_tree_escalation_message(step: int) -> str:
    return (
        f"[URGENT: step {step} and the repository still has NO file changes.]\n\n"
        "Stop searching. Your next command MUST write or edit a real source file "
        "that implements the most literal reading of the task (sed -i, a short "
        "heredoc, or similar). More reads will be rejected. An empty diff at the "
        "deadline is a score of zero."
    )


def _submit_now_message() -> str:
    return (
        "[Submit window: a nonempty patch is already on disk and little wall "
        "clock remains.]\n\n"
        "Do NOT explore further. If any critical requirement is still missing, "
        "make ONE tight edit; otherwise reply with only:\n\n"
        f"```bash\necho {COMPLETION_SENTINEL}\n```"
    )


def _should_send_no_patch_nudge(
    *,
    step: int,
    repo_dir: str,
    already_sent: bool,
    config: AgentRunConfig | None = None,
    started: float | None = None,
) -> bool:
    if already_sent or collect_repo_patch(repo_dir).strip():
        return False
    if step >= _NO_PATCH_NUDGE_STEP:
        return True
    if config is None or started is None or config.wall_clock_limit <= 0:
        return False
    elapsed = time.monotonic() - started
    return elapsed >= config.wall_clock_limit * _NO_PATCH_NUDGE_ELAPSED_FRACTION


def _should_send_empty_escalation(
    *,
    step: int,
    repo_dir: str,
    already_sent: bool,
    config: AgentRunConfig | None = None,
    started: float | None = None,
) -> bool:
    if already_sent or collect_repo_patch(repo_dir).strip():
        return False
    if step >= _NO_PATCH_ESCALATION_STEP:
        return True
    if config is None or started is None or config.wall_clock_limit <= 0:
        return False
    elapsed = time.monotonic() - started
    return elapsed >= config.wall_clock_limit * _NO_PATCH_ESCALATION_ELAPSED_FRACTION


def _empty_tree_hard_gate_active(
    *,
    repo_dir: str,
    config: AgentRunConfig,
    started: float,
) -> bool:
    if collect_repo_patch(repo_dir).strip():
        return False
    if config.wall_clock_limit <= 0:
        return False
    elapsed = time.monotonic() - started
    return elapsed >= config.wall_clock_limit * _EMPTY_TREE_HARD_GATE_FRACTION


def _empty_tree_hard_gate_message(command: str) -> str:
    return (
        "[Empty-tree hard gate: command not run because the repository still has "
        "no changes on disk and a large fraction of the wall clock is gone.]\n\n"
        f"Rejected command: {command[:240]}\n\n"
        "Your next command MUST modify or create a source file (sed -i, heredoc "
        "write, tee, etc.). Reading or searching alone is no longer allowed."
    )


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
            "perl -i",
            "ruby -i",
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
