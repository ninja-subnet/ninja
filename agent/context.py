"""Initial-context builder: full repository summary plus a selective preload
of files the issue names. Deterministic, read-only, standard library."""

from __future__ import annotations

import os
import re

from agent.prompts import build_task_prompt

SUMMARY_ITEM_LIMIT = 400
PRELOAD_MAX_CHARS = 8000
PRELOAD_MAX_FILES = 3
PRELOAD_MAX_FILES_MULTI = 5

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
    named = _resolve_issue_files(issue, repo_dir, paths)
    preload = _issue_file_context(issue, repo_dir, paths, named)
    if len(named) != 1:
        preload = _append_rg_context(issue, repo_dir, preload, max_hits=2 if len(named) >= 2 else 1)
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


def _append_rg_context(issue, repo_dir, existing, max_hits=1):
    if not issue or not repo_dir:
        return existing
    used = len(existing)
    if used >= PRELOAD_MAX_CHARS - 500:
        return existing
    extra = []
    for rel in _rg_owner_hits(issue, repo_dir, max_hits=max_hits):
        if rel in existing:
            continue
        content = _read_file(repo_dir, rel)
        if not content:
            continue
        room = PRELOAD_MAX_CHARS - used
        if room <= 200:
            break
        clip = content[:room]
        suffix = "\n... (truncated)" if len(content) > len(clip) else ""
        extra.append(
            f"-----\nFILE NAME: {rel}\n"
            "NOTE: likely owner file from repository search; use it as context.\n"
            f"FILE CONTENT:\n```\n{clip}{suffix}\n```\n-----"
        )
        used += len(clip)
        if used >= PRELOAD_MAX_CHARS - 500:
            break
    if not extra:
        return existing
    return (existing + "\n" + "\n".join(extra)).strip()


def _rg_owner_hits(issue, repo_dir, max_hits=2):
    terms = _search_terms(issue)
    if not terms:
        return []
    try:
        import subprocess

        proc = subprocess.run(
            ["rg", "-l", terms[0], "--glob", "!{.git,node_modules,dist,build,target,vendor}/**"],
            cwd=repo_dir,
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode not in (0, 1):
        return []
    out = []
    for line in (proc.stdout or "").splitlines():
        rel = line.strip().lstrip("./")
        if rel and rel not in out:
            out.append(rel)
        if len(out) >= max_hits:
            break
    return out


def _search_terms(issue):
    terms = []
    for m in re.finditer(r"`([A-Za-z_][\w.]*)`", issue or ""):
        token = m.group(1)
        if len(token) >= 3 and token not in terms:
            terms.append(token)
    for m in re.finditer(r"\b(?:class|function|def|method|module)\s+([A-Za-z_]\w+)", issue or "", re.I):
        token = m.group(1)
        if token not in terms:
            terms.append(token)
    return terms[:4]


def _issue_file_context(issue, repo_dir, paths, named_files=None):
    blocks, used = [], 0
    file_limit = PRELOAD_MAX_FILES_MULTI if len(named_files or []) >= 2 else PRELOAD_MAX_FILES
    for rel in (named_files or _resolve_issue_files(issue, repo_dir, paths))[:file_limit]:
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
    file_limit = _file_limit_for_issue(issue)
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
        if len(out) >= file_limit:
            break
    return out


def _file_limit_for_issue(issue):
    tokens = set()
    for match in _FILE_TOKEN_RE.finditer(issue or ""):
        rel = (match.group(1) or "").strip().lstrip("./")
        if rel:
            tokens.add(rel)
    return PRELOAD_MAX_FILES_MULTI if len(tokens) >= 2 else PRELOAD_MAX_FILES


def _read_file(repo_dir, rel):
    try:
        with open(os.path.join(repo_dir, rel), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(PRELOAD_MAX_CHARS + 200)
    except OSError:
        return ""
