"""Deterministic file ranker for the initial task prompt (no model calls).

The agent starts blind: with no ground truth of the repository it probes with
targeted greps/finds against a *guessed* model (wrong file extension, wrong root
dir, wrong filenames), and every wrong guess is a full, slow model round-trip.
This module runs as local Python before the loop starts, so it can afford to read
file *contents* and rank every file by issue relevance (issue-keyword BM25 over
path + content) -- ``rank_files`` is the whole public surface, and ``context.py``
turns its top slice into the prompt's shortlist.

Ranking quality on the primary pool, scored against reference patches by
``scripts/prefetch_ab_cli.py`` (per-task means): the top-2 hit a real edit target
on 85% of tasks (precision 0.60, recall 0.42), the top-8 on 100% (precision 0.40,
recall 0.71). Precision is the reason the shortlist shows ranked files as PATHS
only -- a wrong path costs a ``cat``, whereas a wrong file's *content* primes the
model with the wrong code.

Ported from ``deltax/agent/repo_analyser.py`` (the engine's scanner), trimmed to
the ranking core: the engine's line-block mining, confidence scoring and LLM
re-rank pool are not used here.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Files listed in the tree but never worth reading/ranking on content: binaries,
# assets, and vendored/build output. They can still be *edited* (a dep bump in a
# lockfile), so they stay in the file list -- we only skip scanning their bytes.
_BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp", ".pdf",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".mp3", ".mp4", ".mov", ".zip",
    ".gz", ".tar", ".jar", ".class", ".pyc", ".so", ".dylib", ".dll", ".wasm",
    ".map", ".min.js", ".min.css",
}
# Dependency manifests / lockfiles: real edit targets but low discovery value
# (trivially top-level, and huge). Kept in the tree; skipped for content scan.
_LOCK_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
    "poetry.lock", "Cargo.lock", "Gemfile.lock", "go.sum",
    "bun.lock", "bun.lockb",
}
# Documentation prose: occasionally an edit target, but it matches nearly every
# domain word (a README describes the whole system), so it crowds out the real
# code. Ranked, but heavily discounted so a source file always wins a tie.
_DOC_EXT = {".md", ".mdx", ".rst", ".txt", ".adoc"}
_DOC_PENALTY = 0.2
# Test files match the issue's vocabulary (they test the very behavior) but the
# fix belongs in the code under test; a new regression test is written fresh, not
# found. Discounted so the implementation file outranks its test.
_TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec|specs|e2e)(/|$)|(^|/)test_|[._](test|spec)\.", re.I)
_TEST_PENALTY = 0.35
# Example/demo/sample code shares the issue's vocabulary (it demonstrates the
# very feature) but the fix lives in the library it demonstrates. Same prior as
# tests: demoted unless the issue names the file verbatim. The segment may
# continue after the keyword -- Unity's tilde convention (``Samples~/``) and
# suffixed names (``Examples & Extras/``) evaded the exact-segment match and
# put a demo file at rank #0 (duel-20260716174441, ccd2a500/02e3f907).
_EXAMPLE_RE = re.compile(
    r"(^|/)(examples?|samples?|demos?|fixtures|playground|sandbox)([ ~][^/]*)?(/|$)|(^|/)demo_",
    re.I,
)
# Vendored third-party code. ``git ls-files`` includes it when a repo COMMITS
# its dependencies (the walk-fallback skip set never runs), and it carries the
# framework's whole vocabulary: a committed node_modules bundle ranked as the
# deterministic top-2 (4ede4e7a). Only the universal conventions are matched --
# node_modules and minified bundles -- deliberately no per-ecosystem dir names.
# Demoted like infra, exempt on a verbatim mention.
_VENDOR_RE = re.compile(r"(^|/)node_modules(/|$)|[.-]min\.(js|css)$", re.I)
_VENDOR_PENALTY = 0.1
# Build/CI/manifest/config scaffolding: real files, but a feature fix almost never
# lives here, and they match generic tokens (package name, version strings, "config")
# so they crowd the top -- e.g. package.json ranking #0 on a footer task. Demoted
# hard (like docs) so the real code outranks them; still surfaced if nothing else
# scores. Lockfiles are handled separately (skipped from content scan entirely).
_INFRA_BASENAMES = {
    "package.json", "composer.json", "go.mod", "cargo.toml", "requirements.txt",
    "pipfile", "gemfile", "build.gradle", "pom.xml", "setup.py", "setup.cfg",
    "pyproject.toml", "manifest.json", "makefile", "dockerfile",
    "docker-compose.yml", "docker-compose.yaml", "procfile",
    ".gitignore", ".dockerignore", ".editorconfig", ".gitattributes",
    ".npmrc", ".nvmrc", "vite.svg",
}
_INFRA_RE = re.compile(
    r"(\.config\.(js|ts|mjs|cjs)$)"          # vite.config.js, jest.config.ts, ...
    r"|(^|/)tsconfig[^/]*\.json$|(^|/)jsconfig\.json$"
    r"|(^|/)\.eslintrc|(^|/)eslint\.config\.|(^|/)\.prettierrc"
    r"|(^|/)\.babelrc|(^|/)babel\.config\."
    r"|(^|/)\.github/|(^|/)deploy/|\.service$",
    re.I,
)
_INFRA_PENALTY = 0.1
# Structural "wiring" files a feature change almost always must touch (register a
# route, add to the layout/nav, mount a page) but which are keyword-poor -- they
# carry the app's plumbing, not the feature's vocabulary, so pure TF-IDF buries
# them (routes/web.php at #15, layouts/app.blade.php at #8). Given a mild boost so
# they clear the intruders when they already carry *some* issue signal.
_STRUCT_RE = re.compile(
    r"(^|/)(routes?|router|urls)(/|\.|$)"
    r"|(^|/)layouts?/|(^|/)_?layout\.|(^|/)app\.blade\.php$"
    r"|(^|/)(App|main|index)\.(jsx|tsx|vue|svelte)$"
    r"|(^|/)(main|app|server|__init__|routes|urls)\.(py|rb|php|go|ts|js)$",
    re.I,
)
_STRUCT_BOOST = 1.3
# BM25 length normalization: a big file that mentions many keywords shallowly used
# to out-sum a short file that is *about* the issue. BM25 saturates repeat hits
# (k1) and normalizes by document length (b), so the focused file wins.
_BM25_K1 = 1.5
_BM25_B = 0.75
_MAX_CONTENT_BYTES = 131072  # 128 KB: enough to fingerprint a source file

# Generic tokens that carry no discriminating signal for *which* file to open.
_STOPWORDS = frozenset("""
the a an and or but if then else for while with without into onto from to of in on at by as is are be
this that these those it its their there here when where which what who whom how why not no yes can
should must need needs will would could may might do does done make makes made use used using add adds
added new create creates created update updates updated change changes changed fix fixes fixed remove
removes removed implement implements support supports allow allows show shows display displays return
returns based given also only just more most some any all each every other another such same than
task issue feature request currently value values field fields file files code page pages user users
data list lists item items type types name names function functions method methods class classes
component components property properties option options example e.g eg i.e ie etc via per set sets
""".split())

# High-signal issue terms: quoted/backticked code, CamelCase, snake_case,
# ALLCAPS acronyms, and package/path-like tokens.
_BACKTICK_RE = re.compile(r"`([^`\n]{2,60})`")
_CAMEL_RE = re.compile(r"\b[A-Za-z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")   # BuildingInfo, useGetAllBBLs
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")           # stamp_ambiguous
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}s?\b")                          # BBL, BBLs, API
_PATHY_RE = re.compile(r"@?[\w.-]+(?:[/.][\w.-]+)+")                   # @scope/pkg, src/api/hooks.ts
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")

_W_QUOTED = 4.0
_W_IDENT = 3.0
_W_WORD = 1.0

# tf contributions
_TF_BASENAME = 5.0   # the file is literally named after an issue term
_TF_DIRPATH = 2.0    # a directory on the path matches
_TF_CONTENT_CAP = 5  # cap content hits per keyword (a big file can't dominate)

# A filename the issue spells out verbatim ("fix `test_tool_discovery.py`") is
# the strongest possible signal -- the author pointed at the file. Such a file
# is exempt from the test/doc/infra demotions (the issue overrides the prior
# that tests/docs are rarely the fix site) and boosted so it ranks at the top.
_FILENAME_MENTION_RE = re.compile(
    r"\b[\w./-]*[\w-]+\.(?:py|js|jsx|ts|tsx|rs|go|rb|php|java|kt|c|cc|cpp|h|hpp|"
    r"cs|swift|vue|svelte|css|scss|less|html|blade\.php|sql|sh|md|rst|json|ya?ml|toml)\b",
    re.I,
)
_MENTION_BOOST = 3.0


# Never worth listing: caches, virtualenvs, vendored code and build output. Only
# the os.walk fallback below consults this -- ``git ls-files`` already excludes
# .git and everything the repo's own .gitignore names, which is a better answer
# than any hardcoded set. This is the fallback's approximation of that.
_SKIP_DIR_NAMES = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "node_modules", ".next", "dist", "build",
    "target", "vendor", "coverage", ".gradle",
})


def list_repo_files(repo_dir: str) -> List[str]:
    """Tracked files, relative to repo root. ``git ls-files`` (respects
    .gitignore, skips .git/vendored dirs) with an os.walk fallback."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "ls-files"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return [line for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    files: List[str] = []
    for root, dirs, names in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES]
        for n in names:
            files.append(os.path.relpath(os.path.join(root, n), repo_dir))
    return files


def _ext_of(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    if base.startswith(".") and base.count(".") == 1:
        return base           # dotfile like .gitignore
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else "(none)"


def extension_histogram(files: List[str], top: int = 8) -> str:
    """One line naming the repo's real filetypes, most common first -- kills the
    wrong-extension grep (``--include='*.js'`` on a ``.tsx`` repo)."""
    counts = Counter(_ext_of(f) for f in files)
    return ", ".join(f"{n} {ext}" for ext, n in counts.most_common(top))


def top_level_directory_digest(files: List[str], top: int = 12) -> str:
    """The real top-level layout: each top dir with its file count, plus root
    files -- kills the wrong-root assumption (``src/`` on a monorepo)."""
    dir_counts: Counter = Counter()
    root_files: List[str] = []
    for f in files:
        if "/" in f:
            dir_counts[f.split("/", 1)[0] + "/"] += 1
        else:
            root_files.append(f)
    lines = [f"  {d} ({n} files)" for d, n in dir_counts.most_common(top)]
    if root_files:
        lines.append("  (root files) " + ", ".join(sorted(root_files)[:12]))
    return "\n".join(lines)


def repo_digest(files: List[str]) -> str:
    """Filetypes + top-level layout: the whole repo in ~100 tokens.

    A raw path listing cannot describe a large repo -- it can only show a
    truncated *prefix* of one, which on any big repo is a wall of paths from
    whichever directory sorts first (``cmds/``, ``data/``) and never reaches the
    one that owns the task. The digest names every top-level directory and its
    weight instead, so the model learns where the code lives at a fraction of the
    tokens."""
    return "\n".join([
        f"File types: {extension_histogram(files)}",
        f"Top-level layout ({len(files)} tracked files):",
        top_level_directory_digest(files),
    ])


def extract_keywords(issue: str) -> Dict[str, float]:
    """Issue -> {lowercased term: weight}. Higher weight = more discriminating
    (quoted code > identifiers > plain domain words)."""
    weights: Dict[str, float] = {}

    def add(term: str, w: float) -> None:
        t = term.strip().lower()
        if len(t) < 3 or t in _STOPWORDS:
            return
        if weights.get(t, 0.0) < w:
            weights[t] = w

    for m in _BACKTICK_RE.findall(issue):
        add(m, _W_QUOTED)
        # a backtick token is often a path/pkg -- also index its leaf and parts
        for part in re.split(r"[/@.\s]", m):
            if part:
                add(part, _W_IDENT)
    for rx in (_CAMEL_RE, _SNAKE_RE, _ACRONYM_RE, _PATHY_RE):
        for m in rx.findall(issue):
            add(m, _W_IDENT)
            for part in re.split(r"[/@.\-]", m):
                if part and part.lower() != m.lower():
                    add(part, _W_IDENT)
    for m in _WORD_RE.findall(issue):
        add(m, _W_WORD)
    return weights


def mentioned_filenames(issue: str) -> set:
    """Lowercased basenames of files the issue names verbatim (with extension)."""
    names = set()
    for m in _FILENAME_MENTION_RE.finditer(issue):
        names.add(m.group(0).lower().rsplit("/", 1)[-1])
    return names


def _read_text(repo_dir: str, path: str) -> str:
    if _ext_of(path) in _BINARY_EXT or path.rsplit("/", 1)[-1] in _LOCK_NAMES:
        return ""
    try:
        with open(os.path.join(repo_dir, path), "rb") as fh:
            raw = fh.read(_MAX_CONTENT_BYTES)
    except OSError:
        return ""
    if b"\x00" in raw:  # binary
        return ""
    return raw.decode("utf-8", "ignore").lower()


def rank_files(
    issue: str,
    repo_dir: str,
    files: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Ranked (path, score) list -- see ``_score_repo``."""
    return _score_repo(issue, repo_dir, files)[0]


def _score_repo(
    issue: str,
    repo_dir: str,
    files: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, float]], Dict[str, float], Dict[str, float]]:
    """Rank every file by issue relevance; returns (ranked, keywords, idf) so
    downstream scoring (line blocks) shares the same term statistics.

    Signal, strongest first:

      * a filename/path that contains an issue term (dominant), IDF-weighted so
        ubiquitous terms don't help;
      * BM25 over content -- repeat hits saturate (``_BM25_K1``) and the score is
        normalized by file length (``_BM25_B``) so a short file that is *about*
        the issue beats a large file that merely mentions it in passing;
      * demotions for docs, tests, and build/CI/manifest scaffolding (which match
        generic tokens and otherwise crowd the top), and a mild boost for the
        keyword-poor structural wiring files (routers/layouts/entrypoints) a
        feature change almost always has to touch.

    Tuned for recall at a *fixed* small file budget (the pre-fetch surfaces only a
    handful): on the 30-task primary ground truth this lifts core-file recall@8
    from 0.63 to 0.71 without surfacing more files."""
    if files is None:
        files = list_repo_files(repo_dir)
    keywords = extract_keywords(issue)
    if not keywords or not files:
        return [], keywords, {}

    # Pass 1: per-file name/path bonus + raw content counts per keyword + length.
    bonus: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    length: Dict[str, int] = {}
    df: Counter = Counter()
    lengths: List[int] = []
    kw_items = list(keywords.items())
    for path in files:
        if path.rsplit("/", 1)[-1] in _LOCK_NAMES:
            continue  # a lockfile is never a "read this first" target
        low_path = path.lower()
        base = low_path.rsplit("/", 1)[-1]
        content = _read_text(repo_dir, path)
        b_row: Dict[str, float] = {}
        c_row: Dict[str, int] = {}
        for kw, _w in kw_items:
            b = 0.0
            if kw in base:
                b = _TF_BASENAME
            elif kw in low_path:
                b = _TF_DIRPATH
            c = content.count(kw) if content else 0
            if b or c:
                if b:
                    b_row[kw] = b
                if c:
                    c_row[kw] = c
                df[kw] += 1
        L = max(1, len(content.split())) if content else 1
        lengths.append(L)
        if b_row or c_row:
            bonus[path], counts[path], length[path] = b_row, c_row, L

    n = len(files)
    idf = {kw: math.log((n + 1) / (df[kw] + 1)) + 1.0 for kw in keywords}
    avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0

    # Pass 2: BM25 content term + name bonus, IDF-weighted; then demote/boost.
    mentioned = mentioned_filenames(issue)
    scored: List[Tuple[str, float]] = []
    for path in bonus:
        b_row, c_row, L = bonus[path], counts[path], length[path]
        norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * L / avgdl)
        s = 0.0
        for kw in set(b_row) | set(c_row):
            c = c_row.get(kw, 0)
            bm25 = (c * (_BM25_K1 + 1.0)) / (c + norm) if c else 0.0
            term = b_row.get(kw, 0.0) + bm25 * _TF_CONTENT_CAP
            s += keywords[kw] * term * idf[kw]
        if path.lower().rsplit("/", 1)[-1] in mentioned:
            # The issue names this file verbatim: no demotions apply, and it
            # outranks everything that merely shares vocabulary.
            s *= _MENTION_BOOST
        else:
            if _ext_of(path) in _DOC_EXT or path.split("/", 1)[0] == "docs":
                # A doc *named after* an issue identifier (SKILL_LIBRARY.md for a
                # skill-library issue) is a real target; generic prose is not.
                named_after = any(
                    keywords[kw] >= _W_IDENT for kw in b_row
                    if b_row[kw] == _TF_BASENAME
                )
                if not named_after:
                    s *= _DOC_PENALTY
            elif _TEST_RE.search(path) or _EXAMPLE_RE.search(path):
                s *= _TEST_PENALTY
            if _is_infra(path):
                s *= _INFRA_PENALTY
            if _VENDOR_RE.search(path):
                s *= _VENDOR_PENALTY
        if s > 0 and _STRUCT_RE.search(path):
            s *= _STRUCT_BOOST
        scored.append((path, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored, keywords, idf


def _is_infra(path: str) -> bool:
    """Build/CI/manifest/config scaffolding -- demoted, never a feature's core fix."""
    return path.rsplit("/", 1)[-1].lower() in _INFRA_BASENAMES or bool(_INFRA_RE.search(path))


# --------------------------------------------------------------------------- #
# Confidence-scored prefetch: files + line blocks
# --------------------------------------------------------------------------- #
# Line blocks are only mined inside the top-ranked files: block extraction is a
# *localization* refinement of the file ranking, not an independent search, so a
# file the ranker buried contributes no block anyway.
