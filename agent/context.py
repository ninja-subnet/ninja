"""Initial-context builder: a repository summary plus a ranked SHORTLIST of the
files most likely to need editing.

The summary answers "what repo am I in": every path for a small one, and a
``repo_digest`` (filetypes + top-level layout) once it passes ``DIGEST_MIN_FILES``,
because past that size a listing is only ever a truncated prefix of the tree --
long enough to bury the task it sits next to, too short to reach the directory
that owns it.

The shortlist carries two channels, sized by how certain each signal is:

  * a file the TASK ITSELF NAMES is almost always a real edit target (precision
    1.00 on the primary pool, but it only names one on 4 tasks in 20), so its
    CONTENT is worth the tokens -- the model can start reasoning about the real
    code without spending a round-trip to read it;
  * the keyword-ranked candidates are right about 40-60% of the time, so they
    are shown as PATHS ONLY. A wrong path costs the model one ``cat``; a wrong
    file's *content* primes it with the wrong code, which is the failure mode
    that made an earlier low-precision file map lose tasks outright.
Both channels are scored offline by ``scripts/prefetch_ab_cli.py`` (precision /
recall of the shortlist against the reference patch's real edit targets), and
show up per duel as ``profile-duel``'s PfP / PfR columns.
"""

from __future__ import annotations

import os
import re

from agent.prompts import build_task_prompt
from agent.repo_analyser import list_repo_files, rank_files, repo_digest

# Above this many files a raw path listing is worse than useless: it is too long
# to read, it buries the <task> it is supposed to support (a 400-path listing of
# frameworks/native ran 13.8k chars and pushed the model into answering "I don't
# see a task description" on its first turn), and being alphabetical it is only a
# *prefix* of the tree -- on that repo it never reached `libs/`, where every file
# the task named lived. Past this size the repo is described by `repo_digest`
# instead: same information, ~100 tokens, and it names every top-level directory.
DIGEST_MIN_FILES = 50
# How many files the shortlist may name. Files the task names are always kept;
# the rest of the slots go to the ranker's top candidates. Small on purpose: the
# deterministic ranker's top-2 is precision 0.60, and it is only what the loop
# opens on -- ``prefetch.py`` swaps in a longer, sharper list when its background
# re-rank lands (see MAX_RETURN there).
SHORTLIST_MAX_FILES = 2
# Total content budget across every file whose body is embedded. Content is only
# ever embedded for task-named files, so this is not a cap on the shortlist.
PRELOAD_MAX_CHARS = 8000

_FILE_TOKEN_RE = re.compile(
    r"`?([\w./-]+\.(?:py|tsx|ts|jsx|json|js|go|rs|java|cs|rb|php|vue|html|"
    r"css|yaml|yml|md|cpp|hpp|h|c|toml|xml|sql|sh|txt))(?![\w-])`?",
    re.I,
)


def build_context_task(issue, repo_dir, limit=SHORTLIST_MAX_FILES, ranked=None):
    """The full task prompt: repo summary + the shortlist.

    ``ranked`` overrides the file ordering the shortlist is filled from --
    ``prefetch.py`` passes the model's re-ranked list when it lands, so the swap
    reuses this exact builder instead of formatting a second, divergent block."""
    if not repo_dir or not os.path.isdir(repo_dir):
        return ""
    paths = list_repo_files(repo_dir)
    summary = repo_digest(paths) if len(paths) > DIGEST_MIN_FILES else "\n".join(paths)
    shortlist = shortlist_files(issue, repo_dir, paths, limit=limit, ranked=ranked)
    return build_task_prompt(task_text=(issue or "").strip(),
                             repo_summary=summary,
                             preloaded_context=_render(shortlist, repo_dir))


def shortlist_files(issue, repo_dir, paths=None, limit=SHORTLIST_MAX_FILES, ranked=None):
    """The shortlist as ``[(path, named_by_task)]``, most relevant first.

    Task-named files ALWAYS come first and are never displaced (they are the one
    precise signal: precision 1.00 against reference patches, and they are the
    only files whose content is worth embedding). The remaining slots are filled
    from ``ranked`` -- the model's re-rank when prefetch supplies one, otherwise
    the local keyword ranker, which is best-effort: if it raises, the task-named
    files still stand on their own.

    ``scripts/prefetch_ab_cli.py`` scores exactly this function, so the offline
    sweep measures what actually ships."""
    named = _resolve_issue_files(issue, repo_dir, paths if paths is not None else list_repo_files(repo_dir))
    out = [(rel, True) for rel in named[:limit]]
    if ranked is None:
        try:
            ranked = [p for p, _score in rank_files(issue or "", repo_dir)]
        except Exception:  # noqa: BLE001 - a shortlist is an optimization, never fatal
            ranked = []
    for rel in ranked:
        if len(out) >= limit:
            break
        if rel not in named:
            out.append((rel, False))
    return out


def _render(shortlist, repo_dir):
    """The shortlist as prompt text: one entry per file, body only for the
    task-named ones (and only while the content budget lasts)."""
    blocks, used = [], 0
    for rel, named in shortlist:
        content = _read_file(repo_dir, rel) if named else ""
        room = PRELOAD_MAX_CHARS - used
        if not content or room <= 200:
            blocks.append(
                f"-----\nFILE NAME: {rel}\n"
                "NOTE: ranked as likely relevant to the task; content not shown "
                "-- read the file before editing it.\n-----"
            )
            continue
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
    by_base = {}
    for p in paths:
        by_base.setdefault(os.path.basename(p), []).append(p)
    out = []
    for m in _FILE_TOKEN_RE.finditer(issue or ""):
        rel = (m.group(1) or "").strip().lstrip("./")
        if not rel:
            continue
        pick = None
        if os.path.isfile(os.path.join(repo_dir, rel)):
            pick = rel
        else:
            hits = by_base.get(os.path.basename(rel), [])
            if len(hits) == 1 and os.path.isfile(os.path.join(repo_dir, hits[0])):
                pick = hits[0]
        if pick and pick not in out:
            out.append(pick)
    return out


def _read_file(repo_dir, rel):
    try:
        with open(os.path.join(repo_dir, rel), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(PRELOAD_MAX_CHARS + 200)
    except OSError:
        return ""
