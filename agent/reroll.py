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
import os
import re
import shutil
import subprocess
import tempfile
import time

from agent.agent_loop import AgentOutcome, run_agent_loop
from agent.repo_diff import collect_repo_patch

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
    """Full-wall attempt #1 + conditional independent attempt #2; keep the better.

    Never worse than one base draw: attempt #1 uses base_config unchanged and
    stays on the primary tree; attempt #2 can only replace it when strictly
    higher on the objective key. Any error falls open to the on-disk diff.
    """
    repo = getattr(base_config, "repo_dir", "") or ""
    t0 = time.monotonic()
    try:
        budget = float(getattr(base_config, "wall_clock_limit", 0.0) or 0.0)
    except (TypeError, ValueError):
        budget = 0.0
    if budget <= 0.0:
        budget = 280.0

    # Capture the PRISTINE base commit BEFORE attempt #1 so a copy can be reset
    # to it and (when #2 wins) the primary can be rebased onto it. Also confirm
    # we START from a clean git checkout we can safely reset -- attempt #1 will
    # dirty the tree, so this must be measured up front, not afterward. Anything
    # else -> keep attempt #1 and never re-roll.
    orig_sha = _git_out(repo, ["rev-parse", "HEAD"])
    clean_start = (
        orig_sha is not None and _git_out(repo, ["status", "--porcelain"]) == ""
    )

    # Attempt #1 == the king's exact draw (base_config untouched, full wall).
    try:
        outcome_a = run_agent_loop(config=base_config, task=task)
    except Exception:
        return _floor_outcome(repo)

    if not clean_start:
        return outcome_a  # not a clean checkout: cannot safely reset -> keep #1

    named_files, named_syms = _named_tokens(issue_text)
    patch_a = outcome_a.patch or ""
    try:
        info_a = _measure(repo, patch_a, named_files, named_syms)
    except Exception:
        return outcome_a

    multi_req = (len(named_files) + len(named_syms)) >= 2
    remaining = budget - (time.monotonic() - t0)
    if not _is_weak(info_a, multi_req) or remaining < ATTEMPT2_MIN_REMAINING:
        return outcome_a  # good #1, or no budget: keep the king-equivalent draw

    tmp_root = None
    try:
        tmp_root = tempfile.mkdtemp(prefix="reroll_")
        copy_repo = os.path.join(tmp_root, "repo")
        shutil.copytree(repo, copy_repo, symlinks=True)
        if not _reset_verify(copy_repo, orig_sha):
            return outcome_a
        remaining = budget - (time.monotonic() - t0)
        if remaining < ATTEMPT2_MIN_REMAINING:
            return outcome_a
        attempt2_wall = max(_MIN_ATTEMPT2_WALL, remaining - ATTEMPT2_MARGIN)
        cfg2 = dataclasses.replace(
            base_config, repo_dir=copy_repo, wall_clock_limit=attempt2_wall
        )
        try:
            outcome_b = run_agent_loop(config=cfg2, task=task)
        except Exception:
            return outcome_a
        patch_b = outcome_b.patch or ""
        try:
            info_b = _measure(copy_repo, patch_b, named_files, named_syms)
        except Exception:
            return outcome_a

        if _key(info_b) <= _key(info_a):
            return outcome_a  # not strictly better -> keep P1, already on primary

        if (budget - (time.monotonic() - t0)) < MATERIALIZE_MIN_MARGIN:
            return outcome_a  # too close to the kill to swap safely

        if _materialize(repo, orig_sha, patch_b):
            return _outcome_on_disk(outcome_b, repo)
        # P2 apply failed: restore the P1 floor.
        _materialize(repo, orig_sha, patch_a)
        return _outcome_on_disk(outcome_a, repo)
    except Exception:
        return _outcome_on_disk(outcome_a, repo)
    finally:
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)


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
