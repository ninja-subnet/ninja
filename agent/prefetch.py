"""Background LLM re-rank of the file shortlist. Self-contained: nothing else in
the bundle depends on this module, and the agent runs unchanged without it.

The deterministic ranker (``repo_analyser``) answers in ~0.06s (p95 2.0s), so the
loop opens on its top-2 immediately. It is blunt, though -- measured against
reference patches on the primary pool, its top-2 is precision 0.60 / recall 0.42.
A single model call over the same candidates is far sharper -- precision 0.73 /
recall 0.75 at ~5 files, which is the top-10 keyword list's recall at twice its
precision -- but it costs 40-50s: paid up front that is ~8% of the solve budget,
and every step would start that much later.

So it is paid on a daemon thread. The loop runs on the deterministic shortlist
from step 1; when the re-rank lands (typically step 3-6) the task message is
rebuilt around the model's own file list and swapped in. The model is stateless,
so from its next call onward the task simply always carried those files.

The re-rank REPLACES the deterministic list rather than extending it. That does
occasionally discard a file the ranker had right, but the alternative (keep both,
fill the remaining slots from the ranker) measures worse where it counts: +0.06
recall for -0.32 precision, at double the files shown. Wrong files mislead.
The ONE exception is the deterministic top-2: the loop already opened on those,
so re-appending an omitted anchor can never show the model a file it hasn't
seen -- it only prevents the swap from deleting one. On duel-20260716174441 a
reply dropped the det-#0 file that contained the issue's own quoted strings
(judge-decisive); the keep A/B'd at +0.024 recall for -0.044 precision.

Failure is silent by construction: no creds, a timeout, an unparseable reply, any
exception at all -> the updater keeps returning "" and the deterministic
shortlist stands. A wrong file list is worse than a blunt one, so nothing is ever
shown that did not parse cleanly.

Ported from deltax/agent/file_retrieval.py; scored by scripts/prefetch_ab_cli.py
and visible per duel as profile-duel's PfP / PfR columns.
"""

from __future__ import annotations

import json
import re
import threading

from agent.context import build_context_task
from agent.model import ChatModel
from agent.repo_analyser import rank_files

# Candidate pool handed to the model: the keyword ranker's top-N.
CANDIDATE_POOL_N = 50
# Upper bound on files the model may return, and on the shortlist we swap in.
MAX_RETURN = 10
# The reply is a tiny JSON array, but the solver model is a thinking model that
# spirals stochastically; a generous cap keeps the reply intact on the turns
# where it thinks hard (reasoning counts against max_tokens).
_MAX_TOKENS = 8192
# Sized from what the call actually costs IN A DUEL, not on an idle endpoint. A
# solo call answers in ~35-50s, but the validator runs the pool in parallel and
# every solve hits the proxy at once: measured over duel-20260714110910, the
# re-rank took ~102s when it landed. A 90s timeout therefore expired BELOW the
# typical successful latency -- it timed out on 11 of 20 tasks (visible as a
# retry: two recorded side calls and no swap) and delivered on 8. This runs off
# the loop's critical path, so a generous bound costs the solve nothing; all it
# bounds is how late a still-useful answer may arrive. Raised 180 -> 300 after a
# 187.7s reply was discarded just above the old bound: under duel load the tail
# runs longer than 180s often enough that we were throwing away good, in-time
# answers, so we widen the window rather than lose the swap.
_REQUEST_TIMEOUT_S = 300.0
_MAX_ATTEMPTS = 2

# Coverage-demanding wording, A/B'd against the old "Prefer precision: return
# only files you would actually edit" on duel-20260716174441's 50 tasks: that
# clause made the model return 3-9 files with slots free while known targets
# sat in the pool (83 of 180 in-pool drops), skip DI/factory wiring, skip
# deletion targets, and once substitute an existing module for a named-but-
# absent project (ee505f89, -0.55).
_SYSTEM = (
    "You are a code-retrieval assistant for an automated bug-fix agent. Given a task "
    "and a list of candidate files from a repository, select the files that most "
    "likely need to be EDITED to implement the task. Include both the files that own "
    "the described behavior AND the structural/wiring files a change must touch "
    "(routers, layouts, entrypoints, navigation, config, DI/factory registrations) "
    "even when they don't mention the task's words. Files the task says to delete, "
    "rename, or fold into another module are edit targets too. If the task names a "
    "file or project that does not exist in the candidate list, do NOT substitute a "
    "similarly-named existing module -- select the structural files where the new one "
    "must be registered instead. Cover EVERY requirement or acceptance criterion in "
    "the task with at least one file, most-relevant first: a multi-requirement task "
    "needs many files, so use the full budget rather than a minimal set. Skip "
    "documentation files (*.md, changelogs) -- they are not scored -- unless the "
    "task explicitly asks to update one. Return ONLY "
    "a JSON array of file-path strings copied exactly from the candidate list -- no "
    "prose, no comments."
)

_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def start(*, issue, repo_dir, model_name, base_url, auth_token):
    """Kick off the re-rank and return the loop's ``task_updater``.

    The updater returns "" on every call until the re-rank lands, then ONCE
    returns the rebuilt task prompt, then "" forever after. One shot: deliver or
    drop, never re-check."""
    state = {"files": None, "done": False, "delivered": False}

    def _work():
        try:
            state["files"] = _select_files(
                issue=issue, repo_dir=repo_dir, model_name=model_name,
                base_url=base_url, auth_token=auth_token,
            )
        except Exception:  # noqa: BLE001 - the re-rank is best-effort, never fatal
            state["files"] = None
        state["done"] = True

    threading.Thread(target=_work, name="prefetch-rerank", daemon=True).start()

    def task_updater() -> str:
        if state["delivered"] or not state["done"]:
            return ""
        state["delivered"] = True
        files = state["files"]
        if not files:
            return ""
        try:
            # len(files), not MAX_RETURN: kept anchors may push the list to
            # MAX_RETURN + 2, and a tighter limit would silently truncate them.
            return build_context_task(issue, repo_dir, limit=len(files), ranked=files)
        except Exception:  # noqa: BLE001
            return ""

    return task_updater


def _select_files(*, issue, repo_dir, model_name, base_url, auth_token):
    """The model's chosen edit targets, or None if it gave us nothing usable."""
    pool = [path for path, _score in rank_files(issue or "", repo_dir)][:CANDIDATE_POOL_N]
    if not pool:
        return None
    model = ChatModel(
        model_name=model_name,
        base_url=base_url,
        auth_token=auth_token,
        max_completion_tokens=_MAX_TOKENS,
        request_timeout=_REQUEST_TIMEOUT_S,
        max_attempts=_MAX_ATTEMPTS,
    )
    listing = "\n".join(f"{i}. {path}" for i, path in enumerate(pool))
    user = (
        f"TASK:\n{(issue or '').strip()[:2500]}\n\n"
        f"CANDIDATE FILES:\n{listing}\n\n"
        f"Return a JSON array of the files to edit (at most {MAX_RETURN}), most-relevant first."
    )
    reply = model.query([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ])
    picked = _parse(reply, pool)
    return _keep_anchors(picked, pool) if picked else None


def _keep_anchors(picked, pool):
    """``picked`` plus any omitted deterministic top-2 anchor, appended last.

    The loop opened on the ranker's top-2, so this can only re-add files the
    model was already shown -- it stops a reply from deleting the one file the
    ranker had right. Bounded at ``MAX_RETURN + 2``."""
    kept = list(picked)
    for anchor in pool[:2]:
        if anchor not in kept:
            kept.append(anchor)
    return kept[:MAX_RETURN + 2]


def _parse(reply, candidates):
    """The ordered, de-duplicated subset of ``candidates`` the model returned.

    Accepts either format the model emits for a numbered list: exact path strings,
    or the integer indices of the numbered candidates (``[0, 4, 24]``). Robust to
    think-tags, code fences and trailing prose; empty if nothing valid parsed."""
    if not reply:
        return []
    text = _THINK_RE.sub("", reply)
    picked = []
    for block in reversed(_ARRAY_RE.findall(text)):  # last well-formed array wins
        try:
            parsed = json.loads(block)
        except ValueError:
            continue
        if isinstance(parsed, list) and parsed:
            picked = parsed
            break
    if not picked:  # model ignored the JSON contract -- salvage verbatim mentions
        picked = [path for path in candidates if path in text]
    allowed = set(candidates)
    ordered = []
    for item in picked:
        path = None
        if isinstance(item, str):
            candidate = item.strip()
            if candidate in allowed:
                path = candidate
            elif candidate.isdigit() and int(candidate) < len(candidates):
                path = candidates[int(candidate)]
        elif isinstance(item, bool):  # json true/false -- bool is an int subclass
            continue
        elif isinstance(item, int) and 0 <= item < len(candidates):
            path = candidates[item]
        if path and path not in ordered:
            ordered.append(path)
    return ordered[:MAX_RETURN]
