"""Pre-submit patch hygiene checks.

Ported from the 081 king lineage and tuned for mean-score duels: reject
destructive gutting, scratch/munge artifacts, refactor-as-delete, and patches
that ignore task-named files. Complements the completeness/correctness audits
in ``audit.py`` with cheap structural checks that do not need another model turn.
"""

from __future__ import annotations

import os
import re
from typing import Optional

_FILE_IN_ISSUE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|json|yaml|yml|"
    r"md|R|r|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
    re.I,
)
_MUNGE_PATH_RE = re.compile(
    r"^(?:fix|clean|cleanup|replace|update|patch|apply|munge|modify|gen|generate|"
    r"rewrite|migrate|refactor)_[\w.-]+$",
    re.I,
)
_MUNGE_FILE_RE = re.compile(
    r"^(?:fix|update|replace|refactor|patch|apply|clean|generate|rewrite|migrate|"
    r"modify)_[\w.-]+\.(?:py|sh|js|ts|rb|pl)$",
    re.I,
)
_REFACTOR_ISSUE_RE = re.compile(
    r"\b(refactor|rename|restructur|convert|migrate|reorganiz)\b",
    re.I,
)


def _changed_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :].strip()
            if path and path != "/dev/null" and path not in paths:
                paths.append(path)
    return paths


def _line_stats(patch_text: str) -> tuple[int, int]:
    added = removed = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def destructive_patch_reason(patch_text: str) -> Optional[str]:
    added, removed = _line_stats(patch_text)
    if removed >= 60 and added < max(5, removed // 4):
        return (
            f"the patch removes far more than it adds ({removed} deletions vs "
            f"{added} additions); restore required logic instead of gutting the codebase"
        )
    return None


def munge_artifact_reason(patch_text: str) -> Optional[str]:
    for path in _changed_paths(patch_text):
        base = path.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0] if "." in base else base
        if (
            _MUNGE_PATH_RE.match(stem)
            or _MUNGE_FILE_RE.match(base)
            or base.endswith((".new", ".bak", ".orig", ".tmp", ".rej"))
        ):
            return (
                f"the patch adds scratch or munge artifact `{path}`; "
                "edit source files directly and remove helper/backup files"
            )
    return None


def refactor_delete_reason(issue_text: str, patch_text: str) -> Optional[str]:
    if not _REFACTOR_ISSUE_RE.search(issue_text or ""):
        return None
    added, removed = _line_stats(patch_text)
    if removed >= 30 and added < max(8, removed // 3):
        return (
            f"refactor/rename task but patch mostly deletes code "
            f"({removed} deletions vs {added} additions); implement the change in place"
        )
    return None


def task_coverage_reason(
    issue_text: str,
    patch_text: str,
    repo_dir: Optional[str] = None,
) -> Optional[str]:
    mentioned: list[str] = []
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text or ""):
        path = match.group(1).strip().lstrip("./")
        if path not in mentioned:
            mentioned.append(path)
    if not mentioned:
        return None
    touched = _changed_paths(patch_text)
    if not touched:
        return None

    if repo_dir is not None:
        valid_mentioned = []
        for m in mentioned:
            exists_on_disk = os.path.exists(os.path.join(repo_dir, m))
            is_touched = any(
                t == m or t.endswith("/" + m) or m.endswith("/" + t) for t in touched
            )
            if exists_on_disk or is_touched:
                valid_mentioned.append(m)
        mentioned = valid_mentioned

    if not mentioned:
        return None

    hit = sum(
        1
        for m in mentioned
        if any(t == m or t.endswith("/" + m) or m.endswith("/" + t) for t in touched)
    )
    if hit == 0:
        sample = ", ".join(mentioned[:6])
        return (
            f"the task names specific files ({sample}) but the patch does not touch "
            "any of them; find and edit the correct targets"
        )
    return None


# Duel-645807: r16 shipped 1088 changed lines (king 130) for score 0.2; r43
# shipped 357 (king 71) for 0.05. Oversized diffs are usually wrong-scope churn
# the judge penalizes harder than a tight incomplete patch.
_SPRAWL_CHANGED_LINES = 450
_SPRAWL_FILE_COUNT = 10


def sprawl_patch_reason(issue_text: str, patch_text: str) -> Optional[str]:
    added, removed = _line_stats(patch_text)
    changed = added + removed
    files = _changed_paths(patch_text)
    mentioned = {
        m.group(1).strip().lstrip("./")
        for m in _FILE_IN_ISSUE_RE.finditer(issue_text or "")
    }
    # Tasks that explicitly name many files may legitimately touch them.
    named_budget = max(_SPRAWL_FILE_COUNT, len(mentioned) + 2)
    if changed >= _SPRAWL_CHANGED_LINES:
        return (
            f"the patch is sprawling ({changed} changed lines across "
            f"{len(files)} file(s)); shrink it to only the behavior the task "
            "names — revert unrelated churn, keep the owning-file edits"
        )
    if len(files) >= named_budget and changed >= 200:
        return (
            f"the patch touches {len(files)} files with {changed} changed lines, "
            "which is likely out of scope; edit only the files that own the "
            "requested behavior and drop unrelated changes"
        )
    return None


# Mean-score mode: multi-requirement tasks that ship a tiny stub lose by
# 0.5–0.85 (duel-863250 r14/r32/r38). Reject once while budget remains so the
# model can grow a real implementation — but keep the floor LOW so legitimate
# small fixes are not frozen into zeros (the over-strict 863250 regression).
_THIN_MULTI_CRITERIA = 4
_THIN_MULTI_MAX_LINES = 8
_THIN_NAMED_FILES = 2
_THIN_NAMED_MAX_LINES = 20


def thin_relative_to_task_reason(issue_text: str, patch_text: str) -> Optional[str]:
    """Return a reason when the patch is clearly too thin for the task shape."""
    if not (patch_text or "").strip():
        return None
    changed = count_changed_lines(patch_text)
    touched = _changed_paths(patch_text)
    mentioned = []
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text or ""):
        path = match.group(1).strip().lstrip("./")
        if path and path not in mentioned:
            mentioned.append(path)

    # Bullet/numbered criteria only — ignore the generic integration hints that
    # extract_criteria appends (those inflate counts on every task).
    criteria = 0
    for line in (issue_text or "").splitlines():
        s = line.strip()
        if re.match(r"^[-*•]\s+\S", s) or re.match(r"^\d+[.)]\s+\S", s):
            criteria += 1

    if criteria >= _THIN_MULTI_CRITERIA and changed < _THIN_MULTI_MAX_LINES:
        return (
            f"the task lists ~{criteria} concrete requirements but the patch "
            f"only changes {changed} line(s); implement the missing behaviors "
            "in the owning file(s) before submitting"
        )
    if (
        len(mentioned) >= _THIN_NAMED_FILES
        and changed < _THIN_NAMED_MAX_LINES
        and len(touched) < min(2, len(mentioned))
    ):
        sample = ", ".join(mentioned[:4])
        return (
            f"the task names multiple files ({sample}) but the patch is a thin "
            f"{changed}-line edit touching {len(touched)} file(s); cover the "
            "named targets with real behavior, not a stub"
        )
    return None


def patch_reject_reason(
    *,
    issue_text: str,
    patch_text: str,
    repo_dir: str = "",
) -> Optional[str]:
    """First structural reason to reject a submit, or None if the patch is fine.

    Sprawl is intentionally NOT a hard reject: duel-863250 showed that forcing
    the model to trim oversized diffs produced thin incomplete patches that
    scored far worse (r32: 37 lines / 0.05) than the prior sprawl cases.
    Thin-relative expand is applied separately by the loop (once, with budget)
    so autosubmit can still lock in a nonempty patch near the kill.
    """
    if not (patch_text or "").strip():
        return None
    for reason in (
        destructive_patch_reason(patch_text),
        munge_artifact_reason(patch_text),
        refactor_delete_reason(issue_text, patch_text),
        task_coverage_reason(issue_text, patch_text, repo_dir or None),
    ):
        if reason:
            return reason
    return None


def patch_acceptable(patch_text: str) -> bool:
    if not patch_text.strip():
        return False
    if destructive_patch_reason(patch_text) or munge_artifact_reason(patch_text):
        return False
    return True


def count_changed_lines(patch_text: str) -> int:
    added, removed = _line_stats(patch_text)
    return added + removed


def count_changed_files(patch_text: str) -> int:
    return len(_changed_paths(patch_text))
