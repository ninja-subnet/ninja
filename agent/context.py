"""Initial-context builder: full repository summary plus a selective preload
of files the issue names. Deterministic, read-only, standard library."""

from __future__ import annotations

import os
import re

from agent.prompts import build_task_prompt

SUMMARY_ITEM_LIMIT = 400
PRELOAD_MAX_CHARS = 14000
PRELOAD_MAX_FILES = 4
PRELOAD_MAX_FILES_MULTI = 6

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
    # Directory-only entries are redundant with the file paths that already reveal
    # them; omit them from the model-visible summary to shrink the pinned prefix
    # resent (at full prompt-token cost) every request. `paths` itself is untouched,
    # so issue-file resolution and truncation stay identical.
    file_paths = [p for p in paths if not p.endswith("/")]
    shown = [p for p in paths[:SUMMARY_ITEM_LIMIT] if not p.endswith("/")]
    more = len(file_paths) - len(shown)
    note = "" if more <= 0 else f"\n... ({more} more files)"
    summary = ("\n".join(shown) + note) if shown else ""
    named = _resolve_issue_files(issue, repo_dir, paths)
    preload = _issue_file_context(issue, repo_dir, paths, named)
    terms = _search_terms(issue)
    owner = list(named)
    if len(named) != 1:
        ranked = _rg_owner_hits(issue, repo_dir, max_hits=3)
        owner += [r for r in ranked if r not in owner]
        preload = _append_rg_context(issue, repo_dir, preload, max_hits=3)
    preload = _append_test_context(issue, repo_dir, preload, owner, terms)
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


def _rg_count_hits(term, repo_dir):
    """{rel_path: match_count} for one term via `rg -c` (deterministic, read-only, stdlib)."""
    try:
        import subprocess

        proc = subprocess.run(
            ["rg", "-c", "--no-messages", "-e", term,
             "--glob", "!{.git,node_modules,dist,build,target,vendor,__pycache__}/**"],
            cwd=repo_dir, text=True, capture_output=True, timeout=8, check=False,
        )
    except (OSError, ValueError):
        return {}
    except Exception:
        return {}
    if proc.returncode not in (0, 1):
        return {}
    hits = {}
    for line in (proc.stdout or "").splitlines():
        rel, sep, cnt = line.rpartition(":")
        if not sep:
            continue
        rel = rel.strip().lstrip("./")
        if not rel:
            continue
        try:
            hits[rel] = int(cnt)
        except ValueError:
            hits[rel] = 1
    return hits


def _rg_owner_hits(issue, repo_dir, max_hits=2):
    """Multi-term RANKED localizer: score files by DISTINCT issue-terms matched then total
    matches (source over tests on ties). The owner file matches the most terms - far stronger
    than the king's first-term-only `rg -l`. (Full files are preloaded; no span starving.)"""
    terms = _search_terms(issue)
    if not terms:
        return []
    scores = {}
    for term in terms:
        for rel, cnt in _rg_count_hits(term, repo_dir).items():
            slot = scores.setdefault(rel, [0, 0])
            slot[0] += 1
            slot[1] += cnt
    # owner = SOURCE files to EDIT; tests are preloaded separately as the criteria oracle
    src = {rel: v for rel, v in scores.items() if "test" not in rel.lower()}
    pool = src or scores

    def _rank_key(item):
        rel, (distinct_terms, total) = item
        return (distinct_terms, total)

    return [rel for rel, _ in sorted(pool.items(), key=_rank_key, reverse=True)[:max_hits]]


def _find_test_files(repo_dir, owner_files, terms, max_files=2):
    """Rank TEST files by references to the owner module names + issue terms. Tests encode the
    acceptance criteria, so preloading them directly raises implemented-criteria coverage."""
    owner_mods = set()
    for rel in owner_files or []:
        stem = os.path.splitext(os.path.basename(rel))[0]
        if stem and stem not in ("__init__",) and len(stem) >= 3:
            owner_mods.add(stem)
    search = []
    for token in list(owner_mods) + list(terms or []):
        if token and token not in search:
            search.append(token)
    if not search:
        return []
    scores = {}
    for term in search[:8]:
        for rel, cnt in _rg_count_hits(term, repo_dir).items():
            if "test" not in rel.lower():
                continue
            slot = scores.setdefault(rel, [0, 0])
            slot[0] += 1
            slot[1] += cnt
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
    return [rel for rel, _ in ranked[:max_files]]


_TEST_ASSERT_CALL_RE = re.compile(
    r"\b(?:\w+\.)?assert[A-Za-z_]*\(|\b(?:\w+\.)?fail(?:Unless|If)[A-Za-z_]*\("
)
# Tests often define the target output in a named variable, then assert against it
# (`expected = {...}` ... `assert got == expected`). The assert line alone hides the
# concrete value; capturing the literal assignment surfaces the exact required output.
_TEST_EXPECT_RE = re.compile(
    r"^\s*[A-Za-z_]*(?:expected|wanted|want|golden|desired|EXPECTED)[A-Za-z_0-9]*\s*=\s*[\[{(\"'0-9tfrbN]"
)


def _extract_test_requirements(test_content):
    reqs = []
    for line in (test_content or "").splitlines():
        st = line.strip()
        if st.startswith(("def test_", "async def test_")):
            name = st.split("(", 1)[0].replace("async ", "").replace("def ", "").strip()
            reqs.append("behavior: " + name.replace("test_", "").replace("_", " "))
        elif st.startswith("assert ") and len(st) < 200:
            reqs.append(st)
        elif "pytest.raises(" in st and len(st) < 200:
            reqs.append(st)
        elif _TEST_ASSERT_CALL_RE.search(st) and len(st) < 200:
            reqs.append(st)
        elif st.startswith("@") and "parametrize" in st and len(st) < 200:
            reqs.append("cases: " + st)
        elif _TEST_EXPECT_RE.match(line) and len(st) < 200:
            reqs.append("expected value -> " + st)
    seen, out = set(), []
    for r in reqs:
        if r not in seen:
            seen.add(r)
            out.append(r)
        if len(out) >= 28:
            break
    return out


def _append_test_context(issue, repo_dir, existing, owner_files, terms):
    """Preload the top test file(s) exercising the owner code - the acceptance-criteria oracle."""
    used = len(existing)
    if used >= PRELOAD_MAX_CHARS - 500:
        return existing
    extra = []
    for rel in _find_test_files(repo_dir, owner_files, terms):
        if rel in existing:
            continue
        content = _read_file(repo_dir, rel)
        if not content:
            continue
        room = PRELOAD_MAX_CHARS - used
        if room <= 300:
            break
        clip = content[:room]
        suffix = "\n... (truncated)" if len(content) > len(clip) else ""
        reqs = _extract_test_requirements(content)
        req_block = ""
        if reqs:
            req_block = ("REQUIRED BEHAVIORS this test checks (implement EACH correctly):\n"
                         + "\n".join(f"  - {r}" for r in reqs) + "\n")
        extra.append(
            f"-----\nFILE NAME: {rel}\n"
            "NOTE: TEST file exercising the code above. It encodes the REQUIRED behavior / "
            "acceptance criteria - implement everything needed to make ALL of it pass.\n"
            f"{req_block}FILE CONTENT:\n```\n{clip}{suffix}\n```\n-----"
        )
        used += len(clip)
        if used >= PRELOAD_MAX_CHARS - 500:
            break
    if not extra:
        return existing
    return (existing + "\n" + "\n".join(extra)).strip()


def _search_terms(issue):
    """Ranked code-identifier terms: backtick symbols, declared names, exception/error types,
    dotted attributes (+ last component), snake/Camel identifiers - for multi-term ranking."""
    text = issue or ""
    terms, seen = [], set()
    _STOP = {"the", "and", "for", "this", "that", "with", "when", "then", "value", "error",
             "should", "return", "returns", "class", "method", "function", "module", "test"}

    def _add(token):
        token = (token or "").strip().strip(".")
        if len(token) >= 3 and token not in seen and not token.isdigit() and token.lower() not in _STOP:
            seen.add(token)
            terms.append(token)

    for m in re.finditer(r"`([A-Za-z_][\w.]*)`", text):
        _add(m.group(1))
    for m in re.finditer(r"\b(?:class|function|def|method|module)\s+([A-Za-z_]\w+)", text, re.I):
        _add(m.group(1))
    for m in re.finditer(r"\b([A-Z]\w*(?:Error|Exception|Warning))\b", text):
        _add(m.group(1))
    for m in re.finditer(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_]\w+)+)\b", text):
        _add(m.group(1))
        _add(m.group(1).split(".")[-1])
    for m in re.finditer(r"\b([a-z][a-z0-9]*_[a-z0-9_]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+)\b", text):
        _add(m.group(1))
    return terms[:8]


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
