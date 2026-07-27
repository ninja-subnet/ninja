"""Collect the repository patch the same way the validator harness does:
tracked changes via `git diff --binary` plus untracked files via no-index
diffs against /dev/null.
"""

import re
import subprocess

# Regenerated Python bytecode/tool caches only -- never part of a hand-written
# change. Dropping them keeps a stray recompiled cache (swept in by the untracked
# scan when the repo lacks a .gitignore for it) out of the returned patch, so the
# submitted diff reflects only real edits. Deliberately does NOT match node_modules
# or *.egg-info, which can be legitimate edit targets and must survive untouched.
_SAFE_ARTIFACT_RE = re.compile(
    r"(?:^|/)(?:__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)/|\.py[co]$"
)
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/")


def collect_repo_patch(repo_dir: str) -> str:
    diff = _run_git(["diff", "--binary", "--", "."], repo_dir)
    listing = _run_git(["ls-files", "--others", "--exclude-standard", "-z"], repo_dir)
    for relative_path in [item for item in listing.split("\0") if item]:
        if _SAFE_ARTIFACT_RE.search(relative_path):
            continue  # skip regenerated caches; never a real edit target
        file_diff = _run_git_diff_no_index(relative_path, repo_dir)
        diff += file_diff
    return _strip_safe_artifact_sections(diff)


def _strip_safe_artifact_sections(diff: str) -> str:
    """Drop any ``diff --git`` section whose path is a regenerated Python cache.
    Belt-and-suspenders to the untracked-scan skip above: also removes a cache that
    slipped in as a tracked change. Fail-open: unparsed input is returned as-is."""
    if "diff --git" not in diff:
        return diff
    kept, keep = [], True
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            m = _DIFF_HEADER_RE.match(line)
            keep = not (m and _SAFE_ARTIFACT_RE.search(m.group(1)))
        if keep:
            kept.append(line)
    return "".join(kept)


def _run_git(args: list, repo_dir: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or ""


def _run_git_diff_no_index(relative_path: str, repo_dir: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
            cwd=repo_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode in (0, 1):
        return completed.stdout or ""
    return ""
