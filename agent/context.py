"""Initial-context builder: full repository summary plus a selective preload
of files the issue names. Deterministic, read-only, standard library."""

from __future__ import annotations

import os
import re

from agent.prompts import build_task_prompt

SUMMARY_ITEM_LIMIT = 400
PRELOAD_MAX_CHARS = 8000
PRELOAD_MAX_FILES = 2

_SKIP_DIR_NAMES = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "node_modules", ".next", "dist", "build",
    "target", "vendor", "coverage", ".gradle",
})
_FILE_TOKEN_RE = re.compile(
    r"`?([\w./-]+\.(?:py|tsx|ts|jsx|json|js|go|rs|java|cs|rb|php|vue|html|"
    r"css|yaml|yml|md|cpp|hpp|h|c|toml|xml|sql|sh|txt))(?![\w-])`?",
    re.I,
)


def build_context_task(issue, repo_dir):
    if not repo_dir or not os.path.isdir(repo_dir):
        return ""
    paths = _repo_paths(repo_dir)
    shown = paths[:SUMMARY_ITEM_LIMIT]
    more = len(paths) - len(shown)
    note = "" if more <= 0 else f"\n... ({more} more items)"
    summary = ("\n".join(shown) + note) if shown else ""
    preload = _issue_file_context(issue, repo_dir, paths)
    return build_task_prompt(task_text=(issue or "").strip(),
                             repo_summary=summary,
                             preloaded_context=preload)


def _repo_paths(repo_dir):
    root_dir = os.path.abspath(repo_dir)
    paths = []
    for root, dir_names, file_names in os.walk(root_dir, topdown=True, followlinks=False):
        dir_names[:] = sorted(n for n in dir_names if n not in _SKIP_DIR_NAMES)
        rel_root = os.path.relpath(root, root_dir)
        prefix = "" if rel_root == "." else rel_root.replace("\\", "/")
        for n in dir_names:
            paths.append((f"{prefix}/{n}" if prefix else n) + "/")
        for n in sorted(file_names):
            paths.append(f"{prefix}/{n}" if prefix else n)
        if len(paths) >= SUMMARY_ITEM_LIMIT * 3:
            break
    return sorted(paths)


def _issue_file_context(issue, repo_dir, paths):
    blocks, used = [], 0
    for rel in _resolve_issue_files(issue, repo_dir, paths):
        content = _read_file(repo_dir, rel)
        if not content:
            continue
        room = PRELOAD_MAX_CHARS - used
        if room <= 200:
            break
        clip = content[:room]
        suffix = "\n... (truncated)" if len(content) > len(clip) else ""
        blocks.append(
            f"-----\nFILE NAME: {rel}\n"
            "NOTE: current content of a file named by the task; use it as context.\n"
            f"FILE CONTENT:\n```\n{clip}{suffix}\n```\n-----"
        )
        used += len(clip)
    return "\n".join(blocks)


def _resolve_issue_files(issue, repo_dir, paths):
    truncated = len(paths) >= SUMMARY_ITEM_LIMIT * 3
    by_base = {}
    for p in paths:
        if not p.endswith("/"):
            by_base.setdefault(os.path.basename(p), []).append(p)
    out = []
    for m in _FILE_TOKEN_RE.finditer(issue or ""):
        rel = (m.group(1) or "").strip().lstrip("./")
        if not rel:
            continue
        pick = None
        if os.path.isfile(os.path.join(repo_dir, rel)):
            pick = rel
        elif not truncated:
            hits = by_base.get(os.path.basename(rel), [])
            if len(hits) == 1 and os.path.isfile(os.path.join(repo_dir, hits[0])):
                pick = hits[0]
        if pick and pick not in out:
            out.append(pick)
        if len(out) >= PRELOAD_MAX_FILES:
            break
    return out


def _read_file(repo_dir, rel):
    try:
        with open(os.path.join(repo_dir, rel), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(PRELOAD_MAX_CHARS + 200)
    except OSError:
        return ""
