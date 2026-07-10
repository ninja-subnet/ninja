"""Conditional best-of-two orchestrator for the fresh-king symmetric regime.

Attempt #1 runs the base ReAct loop with the UNMODIFIED base budget, so it is
byte-identical to the king's single draw and stays physically on the primary
tree as a floor. Only when attempt #1 is objectively weak AND budget remains do
we take a second independent draw in an isolated copy of the repo, then keep the
better of the two by a deterministic, size-excluded, groundedness-aligned key.
Every failure path falls open to attempt #1's on-disk state, so the worst case
is exactly one base draw. Standard library only; no sampling, no extra judge.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import os
import re
import subprocess
import time

from agent.agent_loop import AgentOutcome, run_agent_loop
from agent.repo_diff import collect_repo_patch
from agent import phases

WATCHDOG_EXTRA = 20.0
PHASE_END_MARGIN = 15.0
STUB_CHECK_MARGIN = 150.0
STABLE_STEPS = 4
STUB_MAX = 6
COMPLETION_MIN_WALL = 55.0
FINISHER_MIN_WALL = 60.0

try:  # reuse the base's issue-file regex; fall back to a local copy
    from agent import _FILE_IN_ISSUE_RE as _FILE_RE
except Exception:
    _FILE_RE = re.compile(
        r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|"
        r"json|yaml|yml|md|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
        re.I,
    )

_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})`")

ATTEMPT2_MIN_REMAINING = 160.0
ATTEMPT2_MARGIN = 100.0
MATERIALIZE_MIN_MARGIN = 15.0
_MIN_ATTEMPT2_WALL = 60.0
_GIT_TIMEOUT = 30


@dataclasses.dataclass
class _PatchInfo:
    nonempty: bool
    py_parses: bool
    touches_named_target: bool
    named_reqs: int
    is_trivial: bool


def run_best_of_two(base_config, task, issue_text) -> AgentOutcome:
    """Byte-identical king primary draw + trajectory-state-gated phases.

    The primary loop runs unchanged except for a stub-stuck monitor that only
    ever breaks a diff that is a small, byte-stable stub on a multi-named issue
    late in the wall. After the loop, a directed completion phase (on a deficient
    diff) or a fresh finisher (on an empty tree) may improve the on-disk state,
    always behind a strict keep-better + additions-only floor. Every failure
    path falls open to the on-disk diff, so the worst case is the king's draw.
    """
    repo = getattr(base_config, "repo_dir", "") or ""
    t0 = time.monotonic()
    try:
        budget = float(getattr(base_config, "wall_clock_limit", 0.0) or 0.0)
    except (TypeError, ValueError):
        budget = 0.0
    if budget <= 0.0:
        budget = 280.0
    hard_end = t0 + budget + WATCHDOG_EXTRA - PHASE_END_MARGIN

    named_files, named_syms = _named_tokens(issue_text)
    named_files = {f for f in named_files
                   if repo and os.path.isfile(os.path.join(repo, f))}
    monitor = _make_stub_monitor(budget, named_files, named_syms)

    cfg1 = dataclasses.replace(base_config, stub_monitor=monitor)
    try:
        outcome = run_agent_loop(config=cfg1, task=task)
    except Exception:
        outcome = _floor_outcome(repo)

    try:
        outcome = _dispatch_phase(base_config, outcome, repo, issue_text, hard_end,
                                  named_files, named_syms)
    except Exception:
        outcome = _outcome_on_disk(outcome, repo)

    return _outcome_on_disk(outcome, repo)


def _make_stub_monitor(budget, named_files, named_syms):
    multi = (len(named_files) + len(named_syms)) >= 2
    check_at = max(90.0, budget - STUB_CHECK_MARGIN)
    state = {"sig": None, "count": 0}

    def monitor(step, elapsed, patch_text):
        if not multi:
            return False
        t = patch_text or ""
        if not t.strip():
            state["sig"] = None
            state["count"] = 0
            return False
        sig = hashlib.sha1(t.encode("utf-8", "replace")).hexdigest()
        if sig == state["sig"]:
            state["count"] += 1
        else:
            state["sig"] = sig
            state["count"] = 0
        if elapsed < check_at or state["count"] < STABLE_STEPS:
            return False
        return _substantive_added(t) < STUB_MAX

    return monitor


def _substantive_added(patch_text):
    n = 0
    for ln in (patch_text or "").splitlines():
        if not ln.startswith("+") or ln.startswith("+++"):
            continue
        s = ln[1:].strip()
        if s and len(s) >= 3 and not s.startswith("#"):
            n += 1
    return n


def _missing_named(repo, patch_text, named_files, named_syms):
    touched = _touched_paths(patch_text)
    tset = {p.lstrip("./") for p in touched} | {os.path.basename(p) for p in touched}
    added_blob = "\n".join(ln[1:] for ln in patch_text.splitlines()
                           if ln.startswith("+") and not ln.startswith("+++"))
    miss_f = [f for f in sorted(named_files)
              if f not in tset and os.path.basename(f) not in tset]
    miss_s = [s for s in sorted(named_syms)
              if not re.search(r"\b" + re.escape(s) + r"\b", added_blob)]
    return miss_f, miss_s


def _dispatch_phase(base_config, outcome, repo, issue_text, hard_end,
                    named_files, named_syms):
    now = time.monotonic()
    to_end = hard_end - now
    patch = collect_repo_patch(repo)
    stub_fired = getattr(outcome, "exit_status", "") == "StubStuck"

    if not patch.strip():
        if to_end >= FINISHER_MIN_WALL:
            phases.run_finisher(base_config, repo, issue_text,
                                named_files, named_syms, hard_end)
        return _outcome_on_disk(outcome, repo)

    try:
        info_a = _measure(repo, patch, named_files, named_syms)
    except Exception:
        return outcome

    if stub_fired:
        if to_end >= COMPLETION_MIN_WALL:
            phases.run_completion(base_config, outcome, repo, issue_text,
                                  named_files, named_syms, hard_end, info_a, patch,
                                  _measure, _key, _missing_named)
        return _outcome_on_disk(outcome, repo)

    miss_f, miss_s = _missing_named(repo, patch, named_files, named_syms)
    multi = (len(named_files) + len(named_syms)) >= 2
    deficient = (
        (not info_a.py_parses)
        or (multi and _substantive_added(patch) < STUB_MAX)
        or (multi and (miss_f or miss_s))
    )
    if deficient and to_end >= COMPLETION_MIN_WALL:
        phases.run_completion(base_config, outcome, repo, issue_text,
                              named_files, named_syms, hard_end, info_a, patch,
                              _measure, _key, _missing_named)
    return _outcome_on_disk(outcome, repo)


# ---------------------------------------------------------------- git helpers
def _git_out(repo, args):
    try:
        r = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_GIT_TIMEOUT, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip()


def _git_run(repo, args):
    try:
        r = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_GIT_TIMEOUT, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def _reset_verify(repo, orig_sha):
    if not _git_run(repo, ["reset", "--hard", orig_sha]):
        return False
    _git_run(repo, ["clean", "-fd"])          # -fd, never -fdx
    if _git_out(repo, ["rev-parse", "HEAD"]) != orig_sha:
        return False
    if _git_out(repo, ["status", "--porcelain"]) != "":
        return False
    try:
        return collect_repo_patch(repo).strip() == ""
    except Exception:
        return False


def _materialize(repo, orig_sha, patch_text):
    """Reset primary to base, then git-apply the patch UNSTAGED. Returns True
    only if the on-disk diff is non-empty afterward."""
    if not patch_text.strip():
        return False
    if not _git_run(repo, ["reset", "--hard", orig_sha]):
        return False
    _git_run(repo, ["clean", "-fd"])
    if not _git_apply(repo, patch_text):
        return False
    try:
        return bool(collect_repo_patch(repo).strip())
    except Exception:
        return False


def _git_apply(repo, patch_text):
    data = patch_text if patch_text.endswith("\n") else patch_text + "\n"
    for extra in (["--whitespace=nowarn"], ["--3way", "--whitespace=nowarn"]):
        try:
            r = subprocess.run(
                ["git", "apply", *extra], cwd=repo, input=data,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=_GIT_TIMEOUT, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:  # NO --index -> changes stay unstaged
            return True
    return False


# ------------------------------------------------------- measurement + select
def _named_tokens(issue_text):
    text = issue_text or ""
    files = set()
    for m in _FILE_RE.finditer(text):
        rel = (m.group(1) or "").strip().lstrip("./")
        if rel:
            files.add(rel)
    syms = {m.group(1) for m in _SYMBOL_RE.finditer(text)}
    return files, syms


def _touched_paths(text):
    paths = set()
    for ln in text.splitlines():
        if ln.startswith("+++ b/"):
            p = ln[6:].strip()
            if p and p != "/dev/null":
                paths.add(p)
        elif ln.startswith("--- a/"):
            p = ln[6:].strip()
            if p and p != "/dev/null":
                paths.add(p)
    return paths


def _all_py_parse(repo, touched):
    for rel in touched:
        if not rel.endswith(".py"):
            continue
        try:
            with open(os.path.join(repo, rel), "r", encoding="utf-8",
                      errors="replace") as fh:
                src = fh.read()
        except FileNotFoundError:
            continue          # deleted file: nothing to parse
        except OSError:
            continue
        try:
            ast.parse(src)    # parse the WHOLE on-disk file, never the diff text
        except (SyntaxError, ValueError):
            return False
    return True


def _touches_named(touched, named_files, added_blob, named_syms):
    base_named = {os.path.basename(f) for f in named_files}
    for p in touched:
        q = p.lstrip("./")
        if p in named_files or q in named_files or os.path.basename(p) in base_named:
            return True
    for sym in named_syms:
        if re.search(r"\b" + re.escape(sym) + r"\b", added_blob):
            return True
    return False


def _named_reqs(touched, named_files, added_blob, named_syms):
    file_hit = 0
    base_named = {os.path.basename(f) for f in named_files}
    for p in touched:
        q = p.lstrip("./")
        if p in named_files or q in named_files or os.path.basename(p) in base_named:
            file_hit = 1
            break
    sym_hits = 0
    for sym in named_syms:
        if re.search(r"\b" + re.escape(sym) + r"\b", added_blob):
            sym_hits += 1
    return file_hit + sym_hits          # capped at len(named_syms)+1: junk can't inflate


def _measure(repo, text, named_files, named_syms):
    nonempty = bool(text.strip())
    touched = _touched_paths(text)
    py_parses = _all_py_parse(repo, touched)
    added = [ln[1:] for ln in text.splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    added_blob = "\n".join(added)
    substantive = 0
    for ln in added:
        s = ln.strip()
        if s and len(s) >= 3 and not s.startswith("#"):
            substantive += 1
    is_trivial = substantive < 2
    touched_named = _touches_named(touched, named_files, added_blob, named_syms)
    named_reqs = _named_reqs(touched, named_files, added_blob, named_syms)
    return _PatchInfo(nonempty, py_parses, touched_named, named_reqs, is_trivial)


def _is_weak(info, multi_req):
    return (
        not info.nonempty
        or not info.py_parses
        or not info.touches_named_target
        or (info.is_trivial and multi_req)
    )


def _key(info):
    return (
        int(info.nonempty),
        int(info.py_parses),
        int(info.touches_named_target),
        info.named_reqs,
        int(not info.is_trivial),
    )


# --------------------------------------------------------------- outcome glue
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
        message="reroll fell open to the on-disk repository diff",
        exit_status="Submitted",
    )
