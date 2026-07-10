"""Deterministic, stdlib-only AGGRESSIVE CRASH / DEFINITION / CALL-SITE locator.

WHY THIS EXISTS
---------------
The king's ``issue_named_context`` preloads a file ONLY when the issue text
literally names an existing repo-relative path (``foo/bar.py``). When the issue
names no such path -- it describes a symptom, mentions a function/class/error, or
pastes a traceback whose paths are ABSOLUTE / dotted / basename-only -- the
``<context>`` block is EMPTY, the model must discover the owning file by hand,
and on a large hard-pool repo the greedy first edit sometimes latches onto a
DECOY (a test, an __init__ re-export, a base class, a call site) instead of the
true owner. The judge grades reachable-coverage against the reference's touched
files, so a wrong-file diff reads as ~0 while the true-file fix is "substantially
more coverage" -- a decisive per-round dip (~0.1-0.3 vs ~0.75), the single
biggest and most frequent king loss on the hard pool.

This module fills that EMPTY slot with a HIGH-PRECISION lookup. It is called ONLY
when the king's named context is empty (see agent.py), so on every round where
the issue names an in-repo path it is a byte-for-byte no-op vs the king.

ANCHOR CLASSES, precision-first
-------------------------------
  * CRASH-SITE (highest confidence): a traceback frame ``File "...", line N`` or
    an explicit ``path:line`` the issue pastes. The line number IS the fix
    region -- no symbol disambiguation needed. We resolve the (often absolute /
    dotted / basename-only) frame path to an in-repo file by suffix match and
    emit a window around line N. This is the king's biggest UNCAUGHT slice: its
    file regex fails on absolute traceback paths (``/sandbox/.../m.py``) and
    dotted modules, so its preload is empty on exactly these rounds, and even
    when it is not, its 8000-char TOP-clip omits a deep crash line. Up to the
    TWO deepest resolved frames are surfaced (the crash site AND one caller) so a
    fix that lives one frame up is still in view.
  * DEFINITION-SITE + CALL-SITES: a file that literally DEFINES a distinctive
    symbol the issue names (``def foo`` / ``class Foo`` / module-level ``foo =``)
    or contains a quoted error-string VERBATIM. Defining a symbol is a
    near-certain ownership signal; merely mentioning it is not (that was
    cand_leanpre_v1's -0.111 bug). Crucially, alongside the DEFINITION we also
    surface the strongest CALL-SITE of the same symbol, mirroring what the
    model's own ``rg <sym>`` would show (definition + top usage). This converts
    the dominant mislead channel -- the fix living in a CALLER rather than at the
    definition -- from a wrong-file loss into a hit, because both the def and a
    representative use are already in view for the model to choose between.

ANTI-MISLEAD / ANTI-REDUNDANCY GATES (why aggression stays bounded)
-------------------------------------------------------------------
  1. DEFINITIONS not mentions. A file becomes a def candidate only if it defines
     the symbol or contains the verbatim error string / crash line.
  2. DISTINCTIVENESS. An anchor whose DEFINITION resolves to more than
     ``_MAX_DEF_FILES`` files is AMBIGUOUS and is DROPPED -- we degrade to the
     king's empty preload rather than guess which of many is the owner.
  3. ADDS-VALUE-OVER-GREP. A symbol anchor fires only when it appears in
     ``>= _MIN_MENTION_FILES`` distinct files (i.e. there ARE decoys/callers the
     greedy king could latch onto). A symbol that appears in exactly ONE file is
     something the king localizes in a single ``grep`` -- firing there is pure
     redundancy (a tie at best, a mislead at worst, never a win), so we stay
     silent and byte-identical to the king.
  4. ISSUE-RELEVANCE RANKING. Among symbol/error anchors we prefer the one the
     issue mentions FIRST (the bug's subject is named early / in the title), not
     the one with the fewest definitions -- an incidental symbol defined once can
     no longer out-rank the real, earlier-named bug symbol.
  5. NO PROMPT COERCION + HONEST FRAMING. The base scaffold prompt is unchanged;
     each block is framed as "one relevant location; the fix may be here or in
     code that calls it -- confirm against the task before editing," so the
     model's own reproduce-first verification still runs and it is not
     hard-anchored onto a single site.

Net: on a no-named-file round it can only ADD file(s) the issue provably talks
about (defines a named symbol, contains a quoted error, or is the crash frame),
only when that resolves unambiguously AND there is real decoy risk, and it pairs
a definition with a representative use so a caller-side fix is not missed;
otherwise it adds nothing.

Cost: runs ONCE before the loop, OUTSIDE the agent wall clock, hard-bounded by a
file-count cap, a cumulative-bytes cap and a 6s wall-clock deadline; any
exception yields "". Deterministic (sorted walk, no random; the deadline can only
REDUCE how much is scanned, never reorder a scanned set). stdlib-only (os, re,
time). No sampling override / RNG / network / credential.
"""

from __future__ import annotations

import os
import re
import time

# --- scan bounds (ceilings, not targets; a normal repo finishes in << 1s) ---
_MAX_FILE_BYTES = 300_000          # skip files larger than this (data/generated)
_MAX_FILES_SCANNED = 4000          # stop after this many candidate source files
_MAX_TOTAL_READ_BYTES = 48_000_000 # stop after cumulatively reading this much
_SCAN_TIME_BUDGET_S = 6.0          # hard wall-clock ceiling for the whole scan

# --- confidence / output shape ---
_MAX_DEF_FILES = 2                 # a symbol DEFINED in > this many files is
                                   # AMBIGUOUS -> dropped (the anti-mislead gate)
_MIN_MENTION_FILES = 2             # a symbol must appear in >= this many files to
                                   # fire (there must be decoys/callers; a single-
                                   # file symbol is grep-trivial -> stay silent)
_MAX_TARGET_FILES = 2              # emit at most this many located files
_TOTAL_CHAR_BUDGET = 8000          # total preload budget (matches king's cap)
_WINDOW_LINES = 130                # lines emitted around an anchor in a big file
                                   # (wider than the king's implicit head clip so
                                   # a deep owning block is fully visible)

_SKIP_DIR = frozenset({
    ".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build",
    "__pycache__", ".tox", ".venv", "venv", "env", "site-packages",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "third_party",
    ".idea", ".eggs", ".next", "target", "coverage", ".gradle",
})
_SRC_EXT = frozenset({
    ".py", ".pyi", ".pyx", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".rb", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php", ".scala",
    ".kt", ".swift", ".m", ".mm", ".sh", ".lua", ".vue",
})

# Explicit code spans the issue author marked: `backticks` and 'quotes'/"quotes".
_BACKTICK_RE = re.compile(r"`([^`\n]{2,120})`")
_QUOTED_RE = re.compile(r"""(['"])([^'"\n]{2,120})\1""")
# A distinctive multi-part identifier appearing in BARE prose (no marks needed):
# snake_case, CamelCase, or dotted a.b.c -- these are almost always real symbols.
_MULTIPART_ID_RE = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.){1,}[A-Za-z_][A-Za-z0-9_]*\b"   # dotted
    r"|\b[a-z0-9]+(?:_[a-z0-9]+)+\b"                                # snake_case
    r"|\b[A-Za-z]+[a-z0-9]+[A-Z][A-Za-z0-9]*\b"                     # CamelCase
)

# --- CRASH-SITE anchors ---------------------------------------------------
# Python-style traceback frame:  File "<path>", line <N>, in <name>
_PY_TB_FRAME_RE = re.compile(r'File "([^"\n]+)",\s*line\s+(\d+)', re.I)
# Generic <path><sep><line>:  src/foo.py:120 | foo.py, line 12 | foo.py line 12
_PATHLINE_RE = re.compile(
    r'\b([\w./-]+\.(?:py|pyi|pyx|js|jsx|ts|tsx|go|rs|java|rb|c|h|cc|cpp|hpp|cs|php|scala|kt|swift|lua|vue))'
    r'["\']?[:,]?\s*(?:line\s+)?(\d+)\b',
    re.I,
)

# Common identifiers that would resolve everywhere -> never use as an anchor.
_STOP_IDS = frozenset({
    "self", "cls", "this", "true", "false", "none", "null", "return", "value",
    "result", "data", "args", "kwargs", "params", "config", "test", "tests",
    "error", "errors", "exception", "response", "request", "handler", "func",
    "function", "method", "class", "object", "string", "number", "list",
    "dict", "array", "index", "name", "type", "types", "field", "model",
    "models", "view", "views", "utils", "util", "helper", "helpers", "main",
    "init", "setup", "run", "get", "set", "add", "remove", "update", "create",
    "delete", "make", "build", "parse", "format", "print", "input", "output",
})

# Test/spec path shapes: a fix almost never lives here, so a "caller" in a test
# file is not surfaced as a companion use site (it would only anchor a decoy).
_TEST_PATH_RE = re.compile(
    r"(^|/)(?:tests?|testing|__tests__|spec|specs)(/|$)"
    r"|(^|/)test_[^/]*$"
    r"|(^|/)[^/]*_test\.[A-Za-z0-9]+$"
    r"|\.spec\.[A-Za-z0-9]+$"
    r"|(^|/)conftest\.py$",
    re.I,
)

_DEF_KEYWORDS = r"(?:def|class|func|function|interface|struct|type|const|let|var)"


def _looks_like_test_path(rel: str) -> bool:
    return bool(_TEST_PATH_RE.search(rel.replace("\\", "/")))


def _norm_id(tok: str) -> str:
    return tok.strip().strip(".,:;()[]{}<>\"'`").strip()


def _extract_anchors(issue: str):
    """Return (id_anchors, str_anchors, crash_frames).

    id_anchors:  distinctive symbol names to search for a DEFINITION.
    str_anchors: quoted multi-word literals to match VERBATIM as error strings.
    crash_frames: list of (raw_path, line) from tracebacks / explicit path:line.
    All precision-first: short/stopword/non-distinct tokens are excluded so they
    can never anchor a wrong file.
    """
    ids: list[str] = []
    seen_ids = set()
    strs: list[str] = []
    seen_strs = set()

    def add_id(tok: str):
        tok = _norm_id(tok)
        if not tok or len(tok) < 4:
            return
        if tok.lower() in _STOP_IDS:
            return
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            return
        if tok not in seen_ids:
            seen_ids.add(tok)
            ids.append(tok)

    # 1) explicit `backtick` spans -> every identifier-ish token inside them.
    for m in _BACKTICK_RE.findall(issue or ""):
        for piece in re.split(r"[^\w.]+", m):
            if not piece:
                continue
            for p in [x for x in piece.split(".") if x]:
                add_id(p)
    # 2) 'quoted'/"quoted" spans -> error-string literals (multi-word) AND ids.
    for _q, body in _QUOTED_RE.findall(issue or ""):
        body_s = body.strip()
        if (" " in body_s or ":" in body_s) and len(body_s) >= 10:
            key = body_s.lower()
            if key not in seen_strs:
                seen_strs.add(key)
                strs.append(body_s)
        else:
            add_id(body_s)
    # 3) bare distinctive multi-part identifiers in prose (snake/Camel/dotted).
    for m in _MULTIPART_ID_RE.findall(issue or ""):
        for p in [x for x in m.split(".") if x]:
            add_id(p)

    # 4) CRASH-SITE frames: traceback frames first (deepest is most relevant),
    #    then any explicit path:line. Dedup on (normalized tail, line).
    frames: list[tuple[str, int]] = []
    seen_frames = set()

    def add_frame(raw: str, ln: str):
        raw = (raw or "").strip().strip("'\"")
        if not raw:
            return
        try:
            line = int(ln)
        except (TypeError, ValueError):
            return
        if line <= 0 or line > 10_000_000:
            return
        key = (os.path.basename(raw), line)
        if key in seen_frames:
            return
        seen_frames.add(key)
        frames.append((raw, line))

    for raw, ln in _PY_TB_FRAME_RE.findall(issue or ""):
        add_frame(raw, ln)
    for raw, ln in _PATHLINE_RE.findall(issue or ""):
        add_frame(raw, ln)

    return ids[:16], strs[:6], frames[:24]


def _iter_source_files(repo_path: str, deadline: float):
    seen = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIR and not d.startswith("."))
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in _SRC_EXT:
                continue
            full = os.path.join(root, name)
            try:
                if os.path.getsize(full) > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield os.path.relpath(full, repo_path), full
            seen += 1
            if seen >= _MAX_FILES_SCANNED or time.monotonic() >= deadline:
                return


def _read(full: str) -> str:
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, ValueError):
        return ""


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _resolve_frames(frames, rels):
    """Map traceback/explicit frames to in-repo files by suffix match.

    A frame path like ``/build/src/pkg/mod.py`` (absolute) or ``pkg/mod.py`` is
    resolved to the repo file whose relpath ends with the longest matching tail.
    Only a UNIQUE resolution is accepted (0 or >1 -> dropped), so an ambiguous
    basename (many ``__init__.py``) never anchors. Returns an ordered list of
    (rel, line) with DEEPEST-frame preference (issue's last frame first, since a
    Python traceback's crash site is its deepest frame).
    """
    rel_set = set(rels)
    # basename -> [rel, ...] for fast unique-suffix resolution
    by_base: dict[str, list[str]] = {}
    for rel in rels:
        by_base.setdefault(os.path.basename(rel), []).append(rel)

    resolved: list[tuple[str, int]] = []
    seen_rel = set()
    # deepest frame first: reverse issue order for tracebacks (last = crash site)
    for raw, line in reversed(frames):
        norm = raw.replace("\\", "/").lstrip("./")
        cand = None
        # 1) exact repo-relative path
        if norm in rel_set:
            cand = norm
        else:
            base = os.path.basename(norm)
            pool = by_base.get(base, [])
            if len(pool) == 1:
                cand = pool[0]
            elif len(pool) > 1:
                # disambiguate by longest matching path tail
                tail_parts = [p for p in norm.split("/") if p]
                best = None
                best_len = 0
                ties = 0
                for rel in pool:
                    rel_parts = rel.split("/")
                    k = 0
                    while (
                        k < len(tail_parts)
                        and k < len(rel_parts)
                        and tail_parts[-1 - k] == rel_parts[-1 - k]
                    ):
                        k += 1
                    if k > best_len:
                        best_len, best, ties = k, rel, 1
                    elif k == best_len:
                        ties += 1
                if best is not None and ties == 1 and best_len >= 2:
                    cand = best
        if cand and cand not in seen_rel:
            seen_rel.add(cand)
            resolved.append((cand, line))
    return resolved


def _issue_pos(issue_lower: str, anchor: str) -> int:
    """First character offset at which *anchor* appears in the issue (its
    relevance signal -- the bug's subject is usually named early). Missing -> a
    large sentinel so it sorts last."""
    p = issue_lower.find(anchor.lower())
    return p if p >= 0 else 1_000_000_000


def build_located_context(repo_path: str, issue: str) -> str:
    """Return a ready-to-embed <context> body naming the crash/definition/use
    site(s) of something the issue mentions, or "" when nothing resolves
    confidently. Pure (reads files only). Fail-open: any error -> "".
    """
    try:
        if not repo_path or not os.path.isdir(repo_path) or not issue or not issue.strip():
            return ""
        id_anchors, str_anchors, crash_frames = _extract_anchors(issue)
        if not id_anchors and not str_anchors and not crash_frames:
            return ""

        # One combined definition regex for all id anchors (single scan/file).
        def_re = None
        mention_re = None
        if id_anchors:
            alt = "|".join(re.escape(a) for a in id_anchors)
            def_re = re.compile(
                # def/class/func/... NAME  (indentation allowed: methods indent);
                # an optional decorator line is tolerated by ^[ \t]* anchoring.
                r"^[ \t]*(?:async[ \t]+)?" + _DEF_KEYWORDS + r"[ \t]+(" + alt + r")\b"
                # module-level NAME = / NAME: (column 0 only, so indented LOCAL
                # variables never count -- keeps the ownership signal precise)
                r"|^(" + alt + r")[ \t]*[:=][^=]",
                re.MULTILINE,
            )
            # Any word-boundary mention of an anchor (definition OR use). Used to
            # (a) gate on decoy-risk and (b) find a representative CALL-SITE.
            mention_re = re.compile(r"\b(" + alt + r")\b")

        deadline = time.monotonic() + _SCAN_TIME_BUDGET_S
        id_def_files: dict[str, dict[str, int]] = {a: {} for a in id_anchors}
        id_mention_files: dict[str, dict[str, int]] = {a: {} for a in id_anchors}
        str_files: dict[str, dict[str, int]] = {a: {} for a in str_anchors}
        all_rels: list[str] = []
        read_bytes = 0

        for rel, full in _iter_source_files(repo_path, deadline):
            all_rels.append(rel)
            # crash-frame resolution needs the file LIST only; content scan is for
            # def/mention/error-string anchors. Skip the read when nothing to scan.
            if def_re is None and not str_anchors:
                if read_bytes >= _MAX_TOTAL_READ_BYTES or time.monotonic() >= deadline:
                    break
                continue
            text = _read(full)
            if not text:
                continue
            read_bytes += len(text)
            if def_re is not None:
                for m in def_re.finditer(text):
                    hit = m.group(1) or m.group(2)
                    if hit and rel not in id_def_files.get(hit, {}):
                        id_def_files.setdefault(hit, {})[rel] = _line_of(text, m.start())
            if mention_re is not None:
                for m in mention_re.finditer(text):
                    hit = m.group(1)
                    bucket = id_mention_files.get(hit)
                    if bucket is not None and rel not in bucket:
                        bucket[rel] = _line_of(text, m.start())
            for s in str_anchors:
                if s in text and rel not in str_files[s]:
                    idx = text.find(s)
                    str_files[s][rel] = _line_of(text, idx)
            if read_bytes >= _MAX_TOTAL_READ_BYTES or time.monotonic() >= deadline:
                break

        # Resolve crash frames against the scanned file list (highest confidence).
        crash_targets = _resolve_frames(crash_frames, all_rels)

        issue_lower = (issue or "").lower()

        # Rank confident anchors. Confidence tiers (lower = better):
        #  -1  crash-site frame resolved to a single in-repo file  (the fix site)
        #   0  error-string literal resolving to a single file      (near-certain)
        #   1  symbol defined in exactly one file                   (very high)
        #   2  error-string literal in 2 files
        #   3  symbol defined in 2 files
        # A symbol DEFINED in 0 files (external) or > _MAX_DEF_FILES (ambiguous),
        # or appearing in < _MIN_MENTION_FILES files (grep-trivial, no decoy risk)
        # contributes NOTHING -> we degrade to the empty preload.
        # candidate = (sort_key, anchor, kind, {rel: line}, use_map_or_None)
        candidates = []
        # crash first, deepest order preserved via an incrementing index.
        for i, (rel, line) in enumerate(crash_targets[:_MAX_TARGET_FILES]):
            candidates.append(((0, i, -1, rel), rel, "crash site", {rel: line}, None))
        for a in str_anchors:
            fmap = str_files[a]
            n = len(fmap)
            if 1 <= n <= _MAX_DEF_FILES:
                tier = 0 if n == 1 else 2
                candidates.append(((1, _issue_pos(issue_lower, a), tier, a), a, "error string", fmap, None))
        for a in id_anchors:
            def_map = id_def_files.get(a, {})
            n_def = len(def_map)
            n_mention = len(id_mention_files.get(a, {}))
            if 1 <= n_def <= _MAX_DEF_FILES and n_mention >= _MIN_MENTION_FILES:
                tier = 1 if n_def == 1 else 3
                # callers = files that mention but do not define the symbol.
                use_map = {
                    r: ln for r, ln in id_mention_files.get(a, {}).items()
                    if r not in def_map
                }
                candidates.append(((1, _issue_pos(issue_lower, a), tier, a), a, "symbol", def_map, use_map))
        if not candidates:
            return ""
        candidates.sort(key=lambda c: c[0])

        # Collect distinct target files, best first. For a symbol that resolves to
        # exactly ONE definition file, spend a remaining slot on its strongest
        # non-test CALL-SITE (def + representative use), so a caller-side fix is
        # not missed. role in {crash site, definition, use site, error string}.
        targets: list[tuple[str, str, str, int]] = []  # (rel, anchor, role, line)
        chosen_rels: set[str] = set()

        def _try_add(rel: str, anchor: str, role: str, line: int) -> bool:
            if rel in chosen_rels or len(targets) >= _MAX_TARGET_FILES:
                return False
            chosen_rels.add(rel)
            targets.append((rel, anchor, role, line))
            return True

        for _key, anchor, kind, fmap, use_map in candidates:
            if len(targets) >= _MAX_TARGET_FILES:
                break
            if kind == "symbol":
                added_def = False
                for rel in sorted(fmap, key=lambda r: (len(r), r)):
                    if _try_add(rel, anchor, "definition", fmap[rel]):
                        added_def = True
                    if len(targets) >= _MAX_TARGET_FILES:
                        break
                # co-surface one representative non-test call site (only when this
                # symbol had a single def file, so the pairing stays def+use).
                if added_def and len(fmap) == 1 and use_map and len(targets) < _MAX_TARGET_FILES:
                    callers = sorted(
                        (r for r in use_map if not _looks_like_test_path(r)),
                        key=lambda r: (len(r), r),
                    )
                    if callers:
                        _try_add(callers[0], anchor, "use site", use_map[callers[0]])
            else:
                for rel in sorted(fmap, key=lambda r: (len(r), r)):
                    _try_add(rel, anchor, kind, fmap[rel])
                    if len(targets) >= _MAX_TARGET_FILES:
                        break
        if not targets:
            return ""

        # Which anchors have BOTH a definition and a paired use site in view
        # (so their notes cross-reference honestly instead of over-committing).
        anchors_in_view = {t[1] for t in targets}
        paired_anchors = {
            a for a in anchors_in_view
            if any(t[1] == a and t[2] == "definition" for t in targets)
            and any(t[1] == a and t[2] == "use site" for t in targets)
        }

        per_file = max(2000, _TOTAL_CHAR_BUDGET // max(1, len(targets)))
        blocks: list[str] = []
        used = 0
        for rel, anchor, role, line in targets:
            full = os.path.join(repo_path, rel)
            text = _read(full)
            if not text:
                continue
            body = _window(text, line, per_file)
            note = _note_for(role, anchor, line, paired=anchor in paired_anchors)
            block = f"-----\nFILE NAME: {rel}\n{note}\nFILE CONTENT:\n```\n{body}\n```\n-----"
            if used + len(block) > _TOTAL_CHAR_BUDGET and blocks:
                break
            blocks.append(block)
            used += len(block)
        return "\n".join(blocks)
    except Exception:
        return ""


def _note_for(role: str, anchor: str, line: int, *, paired: bool) -> str:
    """Honest, non-coercive note for a located block. Framing deliberately keeps
    the model from hard-anchoring on a single site: the fix may be here or in
    related code, and its own reproduce-first check still runs."""
    if role == "crash site":
        return (
            f"NOTE: a traceback / error location in the task points into this file "
            f"near line {line} (resolved by searching the repo). The fix is often at "
            f"or just above this line, but may be in a caller -- read the task and "
            f"confirm before editing."
        )
    if role == "use site":
        return (
            f"NOTE: `{anchor}` (referred to by the task) is USED in this file near "
            f"line {line}. The fix may be here (a call site) or at its definition "
            f"(shown in the companion block) -- confirm against the task before editing."
        )
    if role == "definition":
        if paired:
            return (
                f"NOTE: `{anchor}` (referred to by the task) is DEFINED in this file "
                f"near line {line}. A companion block shows where it is USED; the fix "
                f"may be at the definition or at a use site -- confirm against the "
                f"task before editing."
            )
        return (
            f"NOTE: `{anchor}` (referred to by the task) is DEFINED in this file near "
            f"line {line} (found by searching the repo). This is one relevant "
            f"location; the fix may be here or in code that uses it -- confirm "
            f"against the task before editing."
        )
    # error string
    return (
        f"NOTE: the task quotes text that appears in this file near line {line} "
        f"(found by searching the repo). This is a likely relevant location; "
        f"confirm it against the task before editing."
    )


def _window(text: str, center_line: int, budget: int) -> str:
    """Line-numbered slice of *text* centered on *center_line*, <= budget chars.
    Whole file when it fits; otherwise a window around the anchor so the relevant
    region is visible even in a large file (this is what the king's top-clip
    misses on a deep block).
    """
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
