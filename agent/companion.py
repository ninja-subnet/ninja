"""Deterministic, stdlib-only COMPANION-SITE injector for multi-site fixes.

WHY THIS EXISTS
---------------
The judge (glm-5.2, coverage-of-reachable-code) rewards a patch that covers the
WHOLE requirement. A recurring king failure is the INCOMPLETE MULTI-SITE fix:
the gold change needs the implementation edit PLUS a coordinated companion edit
-- most reliably, when the task ADDS a new PUBLIC symbol, the package's
``__init__.py`` export list (``__all__`` / a re-export block) must ALSO gain the
new name, or the symbol is not actually reachable through the public API. The
issue text describes the symptom at the IMPL site and essentially never names the
export file by path, so the king's ``issue_named_context`` (literal-named-path
only) never surfaces it, the model implements the symbol, its inline
``python -c`` reproduce false-greens against the impl directly, and it submits
single-site -> partial coverage.

WHAT IT DOES
------------
On EXACTLY the rounds where (a) the issue expresses an ADD-a-new-public-symbol
intent naming a distinctive symbol, and (b) the repository has a package
``__init__.py`` that demonstrably curates a PUBLIC EXPORT SURFACE (an ``__all__``
list or a block of ``from .x import y`` re-exports) for that symbol's kind, and
(c) the new symbol is NOT ALREADY exported there, this module returns a compact
``<context>`` block naming that export file and showing its export region, with
an honest note that a newly added public symbol usually must also be registered
there. The model then tends (deterministically, temp-0) to edit BOTH sites.

WHY IT DOES NOT MISLEAD (learned from cand_leanpre_v1 -0.111)
------------------------------------------------------------
Three precision gates keep it silent unless the companion is near-certain, so a
non-firing round is BYTE-IDENTICAL to the base preload (guaranteed tie, zero
trajectory noise):
  1. ADD intent required. It fires only when the issue uses an add/expose/
     register verb next to a distinctive NEW symbol -- not on a generic "fix"/
     "update" issue, where "also edit a second place" would invent churn.
  2. The companion must be a CURATED export surface. A bare ``__init__.py`` with
     no ``__all__`` and no re-export block is NOT a companion (editing it would
     be churn); only a file that already exports SIBLING symbols of the same
     kind qualifies -- proof the project's convention is to list public names
     there.
  3. Already-exported -> silent. If the new symbol name is already present in the
     export surface, no companion edit is needed -> return "" (tie). This also
     means a re-run/idempotent case never double-fires.

It does NOT change the global prompt or the model's per-turn behavior. It only
fills/extends the ``<context>`` channel the base already uses, with a block the
model is free to ignore after reading it; the base reproduce-first verification
still runs unchanged.

Cost & safety: runs ONCE before the loop, off the agent wall clock. Hard-bounded
by file-count, per-file-byte, cumulative-byte and wall-clock ceilings; any
exception yields "". Deterministic (sorted walk; the deadline can only reduce how
much is scanned, never reorder a scanned set). stdlib-only (os, re, time). No
sampling override / RNG / network / credential / new dependency.
"""

from __future__ import annotations

import os
import re
import time

# --- scan bounds (ceilings, not targets; a normal repo finishes in << 1s) ---
_MAX_FILE_BYTES = 300_000
_MAX_INIT_FILES = 2000
_MAX_TOTAL_READ_BYTES = 32_000_000
_SCAN_TIME_BUDGET_S = 5.0

# --- output shape ---
_MAX_TARGET_FILES = 2          # emit at most this many export/companion files
_TOTAL_CHAR_BUDGET = 5000      # total companion preload budget (well under 8000)
_WINDOW_LINES = 70             # lines shown around the export surface

_SKIP_DIR = frozenset({
    ".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build",
    "__pycache__", ".tox", ".venv", "venv", "env", "site-packages",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "third_party",
    ".idea", ".eggs", ".next", "target", "coverage", ".gradle",
})

# Verbs that mark an ADD-a-new-thing intent (as opposed to fix/rename/remove).
_ADD_VERB_RE = re.compile(
    r"\b(add|adds|adding|added|introduce[sd]?|introducing|implement[s]?|"
    r"implementing|implemented|expose[sd]?|exposing|register[sd]?|registering|"
    r"support(?:s|ing|ed)?|provide[sd]?|new)\b",
    re.I,
)

# Distinctive code-symbol shapes we accept as "the new public symbol":
# snake_case, CamelCase, or an explicit `backtick` identifier.
_BACKTICK_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,80})`")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b")

_STOP_IDS = frozenset({
    "self", "cls", "this", "true", "false", "none", "null", "return", "value",
    "result", "data", "args", "kwargs", "params", "config", "test", "tests",
    "error", "errors", "exception", "response", "request", "handler",
    "function", "method", "class", "object", "string", "number", "index",
    "name", "names", "type", "types", "field", "model", "models", "view",
    "views", "utils", "util", "helper", "helpers", "main", "setup", "run",
    "update", "create", "delete", "remove", "make", "build", "parse", "format",
    "print", "input", "output", "should", "would", "could", "which", "there",
    "these", "those", "their", "about", "instead", "because", "however",
    "add_all", "get_all", "readme", "license",
})

# An __all__ list assignment, capturing the bracketed body.
_ALL_ASSIGN_RE = re.compile(r"__all__\s*(?::[^=\n]+)?=\s*[\[(](.*?)[\])]", re.S)
# A public re-export line: from .something import Name[, Name...]
_REEXPORT_RE = re.compile(r"^\s*from\s+\.[\w.]*\s+import\s+(.+)$", re.M)


def _norm(tok: str) -> str:
    return tok.strip().strip(".,:;()[]{}<>\"'`").strip()


def _is_distinctive(tok: str, *, strict: bool) -> bool:
    """A token is a usable NEW-symbol anchor when it is a plausible code symbol
    and not a generic word. ``strict`` (bare-prose tokens): require a multi-part
    snake_case/CamelCase shape -- a single lowercase word in prose is too
    ambiguous to anchor on. Non-strict (explicit ``backtick`` tokens): an author
    marked it as code, so also accept a single Capitalized class-like word
    (``Triangle``), but still reject a single lowercase word (``merge``)."""
    tok = _norm(tok)
    if len(tok) < 4 or tok.lower() in _STOP_IDS:
        return False
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
        return False
    multipart = bool(_SNAKE_RE.fullmatch(tok) or _CAMEL_RE.fullmatch(tok)) or "_" in tok
    if multipart:
        return True
    if strict:
        return False
    # non-strict (backtick): allow a single Capitalized class-like identifier.
    return tok[:1].isupper()


def _extract_new_symbols(issue: str) -> list[str]:
    """Distinctive symbol names the issue plausibly ADDS, best-signal first.

    Requires an add-intent verb somewhere in the issue; returns [] otherwise so
    the injector stays silent on fix/rename/remove issues.
    """
    text = issue or ""
    if not _ADD_VERB_RE.search(text):
        return []
    ordered: list[str] = []
    seen: set[str] = set()

    def add(tok: str, *, strict: bool) -> None:
        tok = _norm(tok)
        if tok and tok not in seen and _is_distinctive(tok, strict=strict):
            seen.add(tok)
            ordered.append(tok)

    # 1) explicit backtick identifiers (highest signal), keep dotted tails too.
    for m in _BACKTICK_RE.findall(text):
        for piece in re.split(r"[^\w]+", m):
            add(piece, strict=False)
    # 2) bare snake_case / CamelCase identifiers in prose (stricter shape gate).
    for rx in (_SNAKE_RE, _CAMEL_RE):
        for m in rx.findall(text):
            add(m, strict=True)
    return ordered[:12]


def _iter_init_files(repo_path: str, deadline: float):
    seen = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIR and not d.startswith("."))
        if "__init__.py" in files:
            full = os.path.join(root, "__init__.py")
            try:
                if os.path.getsize(full) <= _MAX_FILE_BYTES:
                    yield os.path.relpath(full, repo_path), full
                    seen += 1
            except OSError:
                pass
            if seen >= _MAX_INIT_FILES or time.monotonic() >= deadline:
                return


def _read(full: str) -> str:
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, ValueError):
        return ""


def _export_names(text: str) -> set[str]:
    """The set of public names an __init__.py curates: __all__ entries plus
    re-exported identifiers. Empty when the file is not a curated export
    surface (so it does not qualify as a companion)."""
    names: set[str] = set()
    for body in _ALL_ASSIGN_RE.findall(text):
        for piece in re.findall(r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]""", body):
            names.add(piece)
    for imp in _REEXPORT_RE.findall(text):
        for piece in re.split(r"[,\s]+", imp.replace("(", " ").replace(")", " ")):
            piece = piece.strip()
            if piece and piece != "import" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", piece):
                names.add(piece)
    return names


def _kind(name: str) -> str:
    # crude but deterministic: leading uppercase -> class-like, else func-like.
    return "class" if name[:1].isupper() else "func"


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        return 1
    return text.count("\n", 0, idx) + 1


def build_companion_context(repo_path: str, issue: str) -> str:
    """Return a ready-to-embed <context> body naming the export/companion site a
    newly ADDED public symbol must also be registered in, or "" when nothing
    qualifies confidently. Pure (reads files only). Fail-open: any error -> "".
    """
    try:
        if not repo_path or not os.path.isdir(repo_path) or not issue or not issue.strip():
            return ""
        new_syms = _extract_new_symbols(issue)
        if not new_syms:
            return ""
        new_lower = {s.lower() for s in new_syms}

        deadline = time.monotonic() + _SCAN_TIME_BUDGET_S
        read_bytes = 0
        # rel -> (text, export_names)
        curated: list[tuple[str, str, set[str]]] = []
        for rel, full in _iter_init_files(repo_path, deadline):
            text = _read(full)
            if not text:
                continue
            read_bytes += len(text)
            exports = _export_names(text)
            if len(exports) >= 2:  # curated surface: lists >=2 sibling names
                curated.append((rel, text, exports))
            if read_bytes >= _MAX_TOTAL_READ_BYTES or time.monotonic() >= deadline:
                break
        if not curated:
            return ""

        # Choose companion export files whose curated surface matches the KIND of
        # the new symbol (a class-exporting __init__ for a new class, etc.) but
        # does NOT yet list the new symbol. Prefer the surface that exports the
        # most siblings of the same kind (strongest convention signal), then the
        # shallowest path (top-level package public API).
        target_kinds = {_kind(s) for s in new_syms}
        scored: list[tuple[int, int, str, str, set[str]]] = []
        for rel, text, exports in curated:
            if new_lower & {e.lower() for e in exports}:
                continue  # already exported -> no companion needed -> stay silent
            same_kind = sum(1 for e in exports if _kind(e) in target_kinds)
            if same_kind < 1:
                continue
            depth = rel.count("/")
            scored.append((-same_kind, depth, rel, text, exports))
        if not scored:
            return ""
        scored.sort(key=lambda c: (c[0], c[1], c[2]))

        blocks: list[str] = []
        used = 0
        for _sk, _depth, rel, text, exports in scored[:_MAX_TARGET_FILES]:
            # anchor the window on the export surface (__all__ or first re-export)
            anchor = "__all__" if "__all__" in text else "import"
            line = _line_of(text, anchor)
            body = _window(text, line, max(1500, _TOTAL_CHAR_BUDGET // 2))
            sample = ", ".join(sorted(exports)[:6])
            note = (
                "NOTE: this package file curates the PUBLIC export surface "
                f"(exports: {sample}...). The task adds a new public symbol "
                f"({', '.join(new_syms[:3])}); a newly added public symbol "
                "usually must ALSO be registered here (add it to __all__ and/or "
                "re-export it) or it will not be reachable through the public "
                "API -- confirm against the task before editing, and do not "
                "change unrelated entries."
            )
            block = f"-----\nFILE NAME: {rel}\n{note}\nFILE CONTENT:\n```\n{body}\n```\n-----"
            if used + len(block) > _TOTAL_CHAR_BUDGET and blocks:
                break
            blocks.append(block)
            used += len(block)
        return "\n".join(blocks)
    except Exception:
        return ""


def _window(text: str, center_line: int, budget: int) -> str:
    lines = text.splitlines()
    n = len(lines)
    numbered_all = "\n".join(f"{i:>5}\t{ln}" for i, ln in enumerate(lines, 1))
    if len(numbered_all) <= budget:
        return numbered_all
    lo = max(0, center_line - 1 - _WINDOW_LINES // 2)
    hi = min(n, lo + _WINDOW_LINES)
    lo = max(0, hi - _WINDOW_LINES)
    win = "\n".join(f"{i:>5}\t{lines[i - 1]}" for i in range(lo + 1, hi + 1))
    head = "" if lo == 0 else f"... ({lo} earlier lines omitted)\n"
    tail = "" if hi >= n else f"\n... ({n - hi} later lines omitted)"
    return (head + win + tail)[:budget]
