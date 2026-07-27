"""One-hop dependency-graph expansion of the keyword-ranked seed files.

The keyword ranker only scores files that share the issue's vocabulary, so an
edit target that shares none of it -- a registry/route file that must reference a
new symbol, a same-package sibling of a found file, an importer of a found file,
or the other end of a data flow the found file participates in -- is invisible to
both localization stages. This module recovers those structural neighbours from
the top ranked seeds via cheap local edges and returns them as PATHS to append to
the candidate pool / opening shortlist (never as content -- a wrong path costs one
read, wrong content misleads). Fail-silent, and bounded by a hard aggregate time
budget so it can never spend the opening budget it exists to save.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Dict, List

from agent.repo_analyser import _ext_of, _TEST_RE, _VENDOR_RE, _BINARY_EXT, _DOC_EXT

GRAPH_SEED_K = 6            # expand the keyword ranker's top-6
MAX_REV_HITS = 25          # a stem referenced in >25 files is non-discriminating
GRAPH_TIME_BUDGET_S = 4.0  # hard aggregate cap on ALL git-grep on the sync path
SIBLING_CAP = 3            # sibling files admitted per seed directory
# Weights: forward-import + siblings (the implementation the judge scores) are
# valued at least as high as reverse-import (which skews toward small wiring
# edits), so recovered coverage lands on files that carry checklist weight.
_W_FWDIMPORT = 2.5
_W_SIBLING = 2.0
_W_REVIMPORT = 2.0
_NEIGHBOUR_MAX_FILE_BYTES = 200_000

# Stems too generic to reverse-import usefully (they match half the tree).
_GENERIC_STEMS = frozenset({
    "index", "main", "app", "init", "__init__", "utils", "util", "models",
    "model", "views", "view", "serializers", "serializer", "config", "types",
    "type", "const", "constants", "helpers", "helper", "base", "core", "api",
    "routes", "route", "urls", "url", "settings", "test", "tests", "common",
    "schema", "schemas", "handler", "handlers", "service", "services", "controller",
})
_IMPORT_LINE_RE = r"(?:^|\b)(?:import|require|include|from|use)\b"
_JS_EXT = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte"}
# Forward-import module-reference extractors, per language family.
_FWD_RE = {
    "py": re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.M),
    "js": re.compile(r"""(?:from|require\(|import\()\s*['"]([^'"]+)['"]""", re.M),
    "php": re.compile(r"^\s*use\s+([\\\w]+)", re.M),
    "go": re.compile(r'^\s*"([\w./-]+)"', re.M),
    "rb": re.compile(r"""require(?:_relative)?\s+['"]([^'"]+)['"]""", re.M),
    "rs": re.compile(r"^\s*use\s+(?:crate::)?([\w:]+)", re.M),
}


def _stem(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    return base[:base.rfind(".")] if "." in base else base


def _keepable(path: str) -> bool:
    try:
        return not (
            _TEST_RE.search(path) or _VENDOR_RE.search(path)
            or _ext_of(path) in _BINARY_EXT or _ext_of(path) in _DOC_EXT
        )
    except Exception:  # noqa: BLE001
        return False


def _reverse_import(stem: str, repo_dir: str, deadline: float) -> List[str]:
    """Files whose import/use/require lines reference ``stem`` -- the callers and
    registries that name the seed's module. IDF-gated + deadline-bounded."""
    if len(stem) < 4 or stem.lower() in _GENERIC_STEMS:
        return []
    if time.monotonic() >= deadline:
        return []
    pat = _IMPORT_LINE_RE + r".*\b" + re.escape(stem) + r"\b"
    remaining = max(0.5, deadline - time.monotonic())
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "grep", "-lIE", pat],
            capture_output=True, text=True, timeout=remaining,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    hits = [ln for ln in out.stdout.splitlines() if ln.strip()]
    return [] if len(hits) > MAX_REV_HITS else hits


def _forward_import(path: str, repo_dir: str, by_stem: Dict[str, List[str]]) -> List[str]:
    """Repo files the seed imports (the data-flow other end / helper it uses)."""
    ext = _ext_of(path)
    fam = "js" if ext in _JS_EXT else "py" if ext == ".py" else ext.lstrip(".")
    rx = _FWD_RE.get(fam)
    if rx is None:
        return []
    full = os.path.join(repo_dir, path)
    try:
        if os.path.getsize(full) > _NEIGHBOUR_MAX_FILE_BYTES:
            return []
        with open(full, "r", encoding="utf-8", errors="ignore") as fh:
            body = fh.read(_NEIGHBOUR_MAX_FILE_BYTES)
    except OSError:
        return []
    out: List[str] = []
    for ref in rx.findall(body):
        leaf = re.split(r"[\\./:]+", ref.strip())[-1]
        for cand in by_stem.get(leaf, []):
            if cand != path:
                out.append(cand)
    return out


def _siblings(path: str, by_dir: Dict[str, List[str]]) -> List[str]:
    d = path.rsplit("/", 1)[0] if "/" in path else ""
    return [p for p in by_dir.get(d, []) if p != path][:SIBLING_CAP]


def expand_by_graph(seeds: List[str], repo_dir: str, all_files: List[str],
                    cap: int = 10) -> List[str]:
    """Ranked one-hop structural neighbours of ``seeds`` (paths only), excluding
    the seeds. Best-effort: any error or exhausted time budget just yields fewer
    neighbours; it never raises."""
    try:
        seeds = [s for s in (seeds or [])[:GRAPH_SEED_K] if s]
        if not seeds or not all_files:
            return []
        seedset = set(seeds)
        by_stem: Dict[str, List[str]] = {}
        by_dir: Dict[str, List[str]] = {}
        for f in all_files:
            by_stem.setdefault(_stem(f), []).append(f)
            by_dir.setdefault(f.rsplit("/", 1)[0] if "/" in f else "", []).append(f)

        score: Dict[str, float] = {}

        def bump(nb: str, weight: float, seed_rank: int):
            if nb in seedset or not _keepable(nb):
                return
            # closer seeds (higher rank) contribute more; each distinct edge adds
            score[nb] = score.get(nb, 0.0) + weight * (1.0 + 1.0 / (seed_rank + 1))

        deadline = time.monotonic() + GRAPH_TIME_BUDGET_S
        for rank, seed in enumerate(seeds):
            for nb in _forward_import(seed, repo_dir, by_stem):
                bump(nb, _W_FWDIMPORT, rank)
            for nb in _siblings(seed, by_dir):
                bump(nb, _W_SIBLING, rank)
            for nb in _reverse_import(_stem(seed), repo_dir, deadline):
                bump(nb, _W_REVIMPORT, rank)

        ranked = sorted(score, key=lambda p: score[p], reverse=True)
        return ranked[:cap]
    except Exception:  # noqa: BLE001
        return []
