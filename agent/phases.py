"""Trajectory-state-gated phases run after / during the king's primary draw.

Every function here operates on the LIVE primary repo tree and is written to
never raise into the caller: on any error it returns, leaving the on-disk diff
as the outcome. Snapshot/restore gives byte-exact rollback so a phase can only
keep a strictly-better state and otherwise leaves the primary draw untouched.
Standard library only.
"""

from __future__ import annotations

import os
import time

from agent.agent_loop import _cap_messages, _is_submission, _parse_single_command
from agent.environment import execute_command, truncate_text
from agent.model import ChatModel
from agent.prompts import (
    COMPLETION_SENTINEL,
    format_help_message,
    render_observation,
)
from agent.repo_diff import collect_repo_patch

COMPLETION_MAX_STEPS = 6
FINISHER_MAX_STEPS = 8
PHASE_MSG_CAP = 24000
PHASE_OBS_CAP = 4000
REQUEST_SLACK = 8.0
FINISHER_LINE_CAP = 400
ISSUE_CAP = 6000
CONTEXT_CAP = 3500
SNAPSHOT_MAX_FILES = 40
SNAPSHOT_MAX_BYTES = 4_000_000
COMPLETION_ABS_CAP = 600


# --------------------------------------------------------------- shared loop
def _run_phase(base_config, repo, messages, deadline, max_steps, per_command=None):
    """One bounded ReAct loop on the live primary tree. Never raises."""
    model = ChatModel(
        model_name=base_config.model_name,
        base_url=base_config.base_url,
        auth_token=base_config.auth_token,
        max_completion_tokens=base_config.max_tokens,
    )
    fmt_retries = 0
    executed = 0
    for step in range(1, max_steps + 1):
        if time.monotonic() >= deadline - REQUEST_SLACK:
            return
        msgs = _cap_messages(messages, max_chars=PHASE_MSG_CAP)
        try:
            reply = model.query(msgs)
        except Exception:
            return
        messages.append({"role": "assistant", "content": reply})
        command = _parse_single_command(reply)
        if command is None:
            fmt_retries += 1
            if fmt_retries > 1:
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
        executed += 1
        stop = False
        if per_command is not None:
            try:
                stop = bool(per_command(executed))
            except Exception:
                stop = False
        if stop:
            return
        messages.append({"role": "user", "content": render_observation(
            returncode=int(rc or 0),
            output_text=truncate_text(out, PHASE_OBS_CAP),
            remaining_steps=max_steps - step)})


# ----------------------------------------------------------- snapshot/restore
def _touched_paths(patch_text):
    paths = set()
    for ln in (patch_text or "").splitlines():
        if ln.startswith("+++ b/"):
            p = ln[6:].strip()
            if p and p != "/dev/null":
                paths.add(p)
        elif ln.startswith("--- a/"):
            p = ln[6:].strip()
            if p and p != "/dev/null":
                paths.add(p)
    return paths


def _snapshot(repo, patch_text):
    files = sorted(_touched_paths(patch_text))
    if len(files) > SNAPSHOT_MAX_FILES:
        return None
    snap, total = {}, 0
    for rel in files:
        p = os.path.join(repo, rel)
        try:
            data = open(p, "rb").read() if os.path.isfile(p) else None
        except OSError:
            return None
        total += len(data or b"")
        if total > SNAPSHOT_MAX_BYTES:
            return None
        snap[rel] = data
    return snap


def _restore(repo, snap, later_patch):
    if snap is None:
        return
    for rel in _touched_paths(later_patch or ""):
        if rel not in snap:
            p = os.path.join(repo, rel)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
    for rel, data in snap.items():
        p = os.path.join(repo, rel)
        try:
            if data is None:
                if os.path.isfile(p):
                    os.remove(p)
            else:
                d = os.path.dirname(p)
                if d:
                    os.makedirs(d, exist_ok=True)
                open(p, "wb").write(data)
        except OSError:
            pass


# ------------------------------------------------------- additions containment
def _substantive_added(patch_text):
    n = 0
    for ln in (patch_text or "").splitlines():
        if not ln.startswith("+") or ln.startswith("+++"):
            continue
        s = ln[1:].strip()
        if s and len(s) >= 3 and not s.startswith("#"):
            n += 1
    return n


def _added_by_file(patch_text):
    by, cur = {}, None
    for ln in (patch_text or "").splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:].strip()
            by.setdefault(cur, set())
        elif cur is not None and ln.startswith("+") and not ln.startswith("+++"):
            s = ln[1:].strip()
            if s and len(s) >= 3 and not s.startswith("#"):
                by[cur].add(s)
    return by


def _contains(prev_patch, new_patch):
    prev, new = _added_by_file(prev_patch), _added_by_file(new_patch)
    for f, lines in prev.items():
        if not lines <= new.get(f, set()):
            return False
    return True


# ------------------------------------------------------------- completion phase
COMPLETION_DIRECTIVE = (
    "[The changes on disk implement only part of the task.]\n"
    "The task names these items not yet covered by your diff: {missing}.\n"
    "Add the remaining implementation on top of your current changes with real, "
    "reachable code. Do NOT revert, rewrite, or restyle the edits already on disk. "
    "Edit one file per command, the owning source file first. When every named item "
    "is implemented, submit with `echo {sentinel}`."
)


def run_completion(base_config, outcome_a, repo, issue_text,
                   named_files, named_syms, deadline, info_a, patch_a,
                   measure, key_fn, missing_fn):
    best_snap = _snapshot(repo, patch_a)
    if best_snap is None:
        return
    draft_sub = _substantive_added(patch_a)
    line_cap = min(COMPLETION_ABS_CAP, max(200, draft_sub + 300))

    msgs = list(getattr(outcome_a, "transcript", None) or [])
    if len(msgs) < 2 or msgs[0].get("role") != "system":
        return
    miss_f, miss_s = missing_fn(repo, patch_a, named_files, named_syms)
    missing = "; ".join((list(miss_f) + list(miss_s))[:6]) or \
        "the parts of the task not yet implemented"
    msgs = list(msgs)
    msgs.append({"role": "user",
                 "content": COMPLETION_DIRECTIVE.format(missing=missing,
                                                        sentinel=COMPLETION_SENTINEL)})

    best = {"patch": patch_a, "info": info_a, "snap": best_snap}

    def per_command(_i):
        cur = collect_repo_patch(repo)
        try:
            ci = measure(repo, cur, named_files, named_syms)
        except Exception:
            _restore(repo, best["snap"], cur)
            return False
        ok = (ci.py_parses
              and key_fn(ci) > key_fn(best["info"])
              and _substantive_added(cur) <= line_cap
              and _contains(best["patch"], cur))
        if ok:
            snap = _snapshot(repo, cur)
            if snap is not None:
                best["patch"], best["info"], best["snap"] = cur, ci, snap
            else:
                _restore(repo, best["snap"], cur)
        else:
            _restore(repo, best["snap"], cur)
        return False

    _run_phase(base_config, repo, msgs, deadline, COMPLETION_MAX_STEPS,
               per_command=per_command)
    if collect_repo_patch(repo) != best["patch"]:
        _restore(repo, best["snap"], collect_repo_patch(repo))


# --------------------------------------------------------------- empty finisher
FINISHER_SYSTEM = (
    "You are a software engineer finishing an urgent fix in the repository at the "
    "current working directory. Little time remains. Every turn: one short sentence "
    "of reasoning, then exactly ONE bash code block with exactly ONE command (fresh "
    "subshell at the repo root). Your FIRST command must create or modify the source "
    "file that owns the requested behavior - no read-only exploration first. Prefer "
    "one complete heredoc write or a focused sed edit per turn, ONE file per turn. "
    "When the requested behavior is implemented and the diff is non-empty, submit "
    "with: echo " + COMPLETION_SENTINEL
)
FINISHER_USER_TAIL = (
    "\nThe working tree currently has NO changes and little time remains. Implement "
    "the core named requirement now with direct, tightly scoped edits in the file(s) "
    "the task names. Add nothing the task does not ask for."
)


def run_finisher(base_config, repo, issue_text, named_files, named_syms, deadline):
    ctx = _named_file_context(repo, named_files)
    user = ("Please solve this issue:\n\n<task>\n" + (issue_text or "")[:ISSUE_CAP]
            + "\n</task>\n" + ctx + FINISHER_USER_TAIL)
    msgs = [{"role": "system", "content": FINISHER_SYSTEM},
            {"role": "user", "content": user}]

    def per_command(_i):
        return _substantive_added(collect_repo_patch(repo)) > FINISHER_LINE_CAP

    _run_phase(base_config, repo, msgs, deadline, FINISHER_MAX_STEPS,
               per_command=per_command)


def _named_file_context(repo, named_files):
    blocks, used = [], 0
    for rel in sorted(named_files)[:3]:
        try:
            content = open(os.path.join(repo, rel), "r",
                           encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        room = CONTEXT_CAP - used
        if room <= 200:
            break
        clip = content[:room]
        blocks.append("-----\nFILE: %s\n```\n%s\n```\n-----" % (rel, clip))
        used += len(clip)
    return ("\n<context>\n" + "\n".join(blocks) + "\n</context>\n") if blocks else ""

