#!/usr/bin/env python3
"""
Multi-file SWE coding agent for the tau subnet.

Contract (unchanged from the public single-file base agent):
    The validator imports this file and calls:

        solve(
            repo_path="/tmp/task_repo",
            issue="Fix the bug...",
            model="validator-managed-model",
            api_base="http://validator-proxy/v1",
            api_key="per-run-proxy-token"
        )

    It returns a dict with patch, logs, steps, cost, and success.

Layout:
    agent.py             validator-owned contract + thin solve() wiring
    agent/prompts.py     system/instance templates tuned for diff-match scoring
    agent/model.py       stdlib OpenAI-compatible chat client with retries
    agent/environment.py fresh-subshell bash executor
    agent/agent_loop.py  the query -> act -> observe step loop
    agent/repo_diff.py   harness-compatible patch collection

All inference uses only the validator-provided api_base/api_key; there are no
third-party dependencies and no sampling overrides (the validator proxy owns
sampling).
"""

from __future__ import annotations

import os
import re
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from agent.agent_loop import AgentRunConfig, run_agent_loop
from agent.prompts import build_task_prompt
from agent.repo_diff import collect_repo_patch

try:
    # Multi-site COMPANION injector: when the issue ADDS a new public symbol
    # and the package curates a public export surface (__all__ / re-exports)
    # that does not yet list it, surface that export file as <context> so the
    # model wires the new symbol into BOTH sites. High-precision + fail-open:
    # if nothing qualifies (or on any error) it yields "" and the preload is
    # byte-identical to the base, so a non-multi-site round is an exact tie.
    from agent.companion import build_companion_context as _build_companion_context
except Exception:  # pragma: no cover
    _build_companion_context = None

try:
    # DEF-SITE LOCATOR: when the issue names NO owning file (king preload empty)
    # but mentions a distinctive symbol/error-string that a high-precision repo
    # scan resolves to 1-2 DEFINITION files, fill the empty <context> slot with
    # that definition site so the deterministic model starts its trajectory from
    # the true owning file instead of locking onto a decoy. Targets the king's
    # AMBIGUOUS-COLD-START-LOCALIZATION wrong-file (~0) failure. Fires only when
    # the named-file preload is empty AND the anchor resolves uniquely; otherwise
    # returns "" -> byte-identical to the base. Runs once, before the loop, off
    # the loop clock, hard-capped by a 6s scan budget. Fail-open import.
    from agent.locate import build_located_context as _build_located_context
except Exception:  # pragma: no cover
    _build_located_context = None

# Combined <context> ceiling. Sized ABOVE any reachable sum of the component
# budgets INCLUDING their block wrappers/notes -- named-file baseline (8000
# content + ~800 wrappers) + whole-block appendix (6000 content + ~800
# wrappers) + companion (~5000 incl. wrappers); the primary slot holds EITHER
# the named-file preload (with its optional whole-block appendix) OR the
# located context (<=8000+wrappers), never both, and the companion block is
# purely additive. Each injector already enforces its own budget, so this cap
# NEVER truncates in practice: it is a defensive invariant (guards a
# pathological oversized block), not an active truncator -- a whole-block
# appendix can never be chopped mid-block by a co-firing companion. The pinned
# first message stays far under the 90000-char message cap.
_MAX_COMBINED_PRELOAD_CHARS = 24000

# The def-site locate scan only runs in the generous-budget (live ~600s) regime,
# where its ~6s pre-loop scan is negligible against the round budget. In a tight
# regime it is skipped, so the preload stays byte-identical to the base.
_LOCATE_MIN_WALL_CLOCK_SECONDS = 400.0

# -----------------------------
# Config
# -----------------------------

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "50"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "15"))

# VALIDATOR CONTRACT: These defaults are only fallbacks for local testing and
# validator wiring. During real validation the validator passes model, api_base,
# and api_key into solve(). Keep this code compatible with that path.
DEFAULT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("NINJA_MODEL", "")
DEFAULT_API_BASE = (
    os.environ.get("AGENT_API_BASE")
    or os.environ.get("NINJA_INFERENCE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL", "")
)
DEFAULT_API_KEY = (
    os.environ.get("AGENT_API_KEY")
    or os.environ.get("NINJA_INFERENCE_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "8000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))
MAX_MESSAGE_CHARS = int(os.environ.get("AGENT_MAX_MESSAGE_CHARS", "90000"))


# Stay under the validator's per-round budget so the loop can finish gracefully
# and report its own patch instead of relying on the kill path. The validator
# now exports its real per-round budget as TAU_AGENT_TIMEOUT_SECONDS; honor it
# (leaving a margin for diff collection) so a looser budget actually lets the
# agent keep working. Falls back to the conservative 280s when unset.
def _wall_clock_limit_seconds() -> float:
    budget = os.environ.get("TAU_AGENT_TIMEOUT_SECONDS")
    if budget:
        try:
            return max(60.0, float(int(budget)) - 20.0)
        except ValueError:
            pass
    return 280.0


WALL_CLOCK_LIMIT_SECONDS = _wall_clock_limit_seconds()
_REPO_SUMMARY_ITEM_LIMIT = 80
_PRELOAD_MAX_CHARS = 8000
_SKIP_DIR_NAMES = frozenset({
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    ".next",
    "dist",
    "build",
    "target",
    "vendor",
    "coverage",
    ".gradle",
})
_FILE_IN_ISSUE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|json|yaml|yml|md|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
    re.I,
)


def _normalize_api_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _resolve_inference_config(
    model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, str]:
    model_name = (model or DEFAULT_MODEL).strip()
    base = (api_base or DEFAULT_API_BASE).strip()
    key = (api_key if api_key is not None else DEFAULT_API_KEY).strip()

    if not model_name:
        raise ValueError("model is required; validators must pass the centrally managed model id")
    if not base:
        raise ValueError("api_base is required; validators must pass the managed inference proxy URL")
    if not key:
        raise ValueError("api_key is required; validators must pass the per-run proxy token")

    return model_name, _normalize_api_base(base), key


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    return build_task_prompt(task_text=issue, repo_summary=repo_summary, preloaded_context=preloaded_context)


def build_repo_summary(repo_dir: str) -> str:
    if not repo_dir or not os.path.isdir(repo_dir):
        return ""
    paths = _repo_paths(repo_dir)
    shown = paths[:_REPO_SUMMARY_ITEM_LIMIT]
    note = "" if len(paths) <= len(shown) else f"\n... ({len(paths) - len(shown)} more items)"
    return "\n".join(shown) + note


def _repo_paths(repo_dir: str) -> list[str]:
    root_dir = os.path.abspath(repo_dir)
    paths: list[str] = []
    for root, dir_names, file_names in os.walk(root_dir, topdown=True, followlinks=False):
        dir_names[:] = sorted(name for name in dir_names if name not in _SKIP_DIR_NAMES)
        rel_root = os.path.relpath(root, root_dir)
        rel_prefix = "" if rel_root == "." else rel_root.replace("\\", "/")
        for name in dir_names:
            rel = f"{rel_prefix}/{name}" if rel_prefix else name
            paths.append(rel + "/")
        for name in sorted(file_names):
            rel = f"{rel_prefix}/{name}" if rel_prefix else name
            paths.append(rel)
        if len(paths) >= _REPO_SUMMARY_ITEM_LIMIT * 3:
            break
    return sorted(paths)


def issue_named_context(issue: str, repo_dir: str) -> str:
    """Preload the current content of files named by the task.

    Whole-block-aware, STRICTLY ADDITIVE refinement of the baseline preload.
    The baseline shows the TOP `budget` characters of each named file. When a
    named file is larger than that budget, the baseline truncates away whatever
    lives past the clip -- frequently the very function/class the task asks
    about (owning block lives further down). This refinement keeps the baseline
    clip byte-for-byte (so the model never sees LESS than the baseline) and,
    only when the baseline truncated a def/class the task names, APPENDS the
    complete text of that block so the model also gets a coherent view of the
    code that owns the requested behavior. When nothing was truncated, or no
    owning block can be confidently identified, the output is byte-for-byte
    identical to the baseline; any error falls back to the baseline. This
    changes only the model's input context; it never adds anything to the
    produced diff, and its output always contains the baseline as an exact
    prefix (it can only add context, never remove it).
    """
    try:
        return _issue_named_context_focused(issue, repo_dir)
    except Exception:
        return _issue_named_context_baseline(issue, repo_dir)


def _baseline_blocks(issue_text: str, repo_dir: str) -> Tuple[list, list]:
    """King-identical preload blocks plus a per-file display record.

    Returns (blocks, displayed): `"\\n".join(blocks)` is EXACTLY the baseline
    preload string, and `displayed` is a list of (path, content, clip_len) for
    each file the baseline actually showed, so the focused pass can tell what
    the baseline truncated. This loop must stay byte-for-byte equivalent to the
    historical baseline preload.
    """
    blocks: list = []
    displayed: list = []
    used = 0
    for path in _existing_issue_files(issue_text, repo_dir, limit=3):
        content = _read_repo_file(repo_dir, path)
        if not content:
            continue
        budget = _PRELOAD_MAX_CHARS - used
        if budget <= 200:
            break
        clipped = content[:budget]
        suffix = "\n... (truncated)" if len(content) > len(clipped) else ""
        block = (
            f"-----\nFILE NAME: {path}\n"
            "NOTE: current content of a file named by the task; use it as context.\n"
            f"FILE CONTENT:\n```\n{clipped}{suffix}\n```\n-----"
        )
        blocks.append(block)
        displayed.append((path, content, len(clipped)))
        used += len(clipped)
    return blocks, displayed


def _issue_named_context_baseline(issue_text: str, repo_dir: str) -> str:
    blocks, _ = _baseline_blocks(issue_text, repo_dir)
    return "\n".join(blocks)


_APPENDIX_NOTE = (
    "complete definition(s) of symbol(s) the task names that were truncated "
    "from the clipped view above; shown here in full as extra context. Confirm "
    "against the task and read the file directly if you need more."
)


def _issue_named_context_focused(issue_text: str, repo_dir: str) -> str:
    """Baseline preload, plus an appendix of complete owning blocks it truncated.

    The returned string is the baseline preload verbatim, optionally followed by
    the complete def/class blocks the task names that the baseline clipped away.
    It therefore always contains the baseline as an exact prefix -- the model
    can never see less context than the baseline gives it.
    """
    blocks, displayed = _baseline_blocks(issue_text, repo_dir)
    baseline_out = "\n".join(blocks)
    symbols = _issue_symbols(issue_text)
    if not symbols:
        return baseline_out
    appendix: list = []
    extra_used = 0
    for path, content, clip_len in displayed:
        if len(content) <= clip_len:
            continue  # the baseline already showed this file in full
        room = _FOCUS_EXTRA_MAX_CHARS - extra_used
        if room <= 200:
            break
        deep = _extract_truncated_owning_blocks(content, symbols, clip_len, room)
        if not deep:
            continue
        appendix.append(
            f"-----\nADDITIONAL CONTEXT FOR FILE: {path}\n"
            f"NOTE: {_APPENDIX_NOTE}\n"
            f"FILE CONTENT (continued):\n```\n{deep}\n```\n-----"
        )
        extra_used += len(deep)
    if not appendix:
        return baseline_out
    if baseline_out:
        return baseline_out + "\n" + "\n".join(appendix)
    return "\n".join(appendix)


_ISSUE_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_BACKTICK_SPAN_RE = re.compile(r"`([^`]+)`")
_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF_LINE_RE = re.compile(r"^([ \t]*)(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)")
_MAX_BLOCK_LINES = 400
_FOCUS_EXTRA_MAX_CHARS = 6000

# A backtick span that is exactly one identifier, optionally with an empty call
# suffix -- `parse_config` or `parse_config()`. Such a lone-identifier span is a
# deliberate, unambiguous code reference and is trusted verbatim.
_LONE_SYMBOL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(\s*\))?$")
# An identifier written as a call head inside a larger expression span --
# `x.get(` -> get, `parse_config(` -> parse_config.
_CALL_HEAD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(")

# Pronoun/receiver noise: appears as a dotted head but never names an owning
# def/class worth surfacing.
_NOISE_TOKENS = frozenset({"self", "cls"})

# Ultra-generic accessor/verb heads. Even written with call syntax
# (`settings.get(...)`) these collide with unrelated `def get`/`def set` blocks
# in a large file, so they are never admitted from the bare or call-head paths.
# Explicit snake_case/mixedCase names and lone-identifier backtick spans are
# unaffected (a token like `get_value` still qualifies by its identifier shape,
# and a deliberate lone `` `get` `` backtick is still honored).
_GENERIC_ACCESSORS = frozenset({
    "get", "set", "add", "pop", "put", "run", "save", "load", "call", "make",
    "build", "update", "delete", "remove", "init", "parse", "read", "write",
    "fetch", "create", "append", "insert", "send", "open", "close",
})


def _has_ident_shape(token: str) -> bool:
    """True when the token is written like a code identifier: snake_case
    (contains '_') or mixedCase (has both an upper- and a lower-case letter)."""
    if "_" in token:
        return True
    return any(c.isupper() for c in token) and any(c.islower() for c in token)


def _collect_backtick_symbols(span_text: str, out: set) -> None:
    """Harvest symbols from one backtick span into `out`.

    A lone-identifier span (`name` or `name()`) is an explicit reference and is
    trusted verbatim when at least 3 characters. A larger expression span
    (`settings.get('x')`, `a.b.c`) contributes only its identifier-shaped tokens
    and its explicit, non-generic call heads (`x.get(` -> dropped as generic;
    `x.parse_config(` -> parse_config), never bare dotted receivers/sub-words
    like self/key/value.
    """
    stripped = span_text.strip()
    lone = _LONE_SYMBOL_RE.match(stripped)
    if lone:
        name = lone.group(1)
        if len(name) >= 3:
            out.add(name)
        return
    for match in _CALL_HEAD_RE.finditer(span_text):
        head = match.group(1)
        if len(head) >= 3 and head not in _GENERIC_ACCESSORS:
            out.add(head)
    for token in _CODE_TOKEN_RE.findall(span_text):
        if len(token) >= 3 and _has_ident_shape(token):
            out.add(token)


def _is_named_code_symbol(token: str, issue_text: str, start: int, end: int) -> bool:
    """Admit a bare (non-backtick) prose token only as an unambiguous code
    reference: an identifier shape (snake_case/mixedCase), or written as a
    non-generic `name(` call. The loose length>=7 prose heuristic and the bare
    dotted-adjacency branches are deliberately dropped -- they were the source
    of unrelated-block mislead in the un-hardened filter.
    """
    if _has_ident_shape(token):
        return True
    if (
        end < len(issue_text)
        and issue_text[end] == "("
        and token not in _GENERIC_ACCESSORS
    ):
        return True
    return False


def _issue_symbols(issue_text: str) -> "frozenset[str]":
    """Code-shaped identifiers the task names, high precision.

    Only unambiguous code references are admitted so ordinary prose cannot
    surface an unrelated owning block:

      * backtick spans go through `_collect_backtick_symbols` (lone identifier
        trusted; a larger expression contributes only shaped tokens / call heads);
      * bare prose tokens go through `_is_named_code_symbol` (identifier shape or
        a non-generic `name(` call).

    Generic accessor heads and pronoun receivers are excluded. This is strictly
    a precision tightening of the historical filter: every token yielded here was
    also yielded by that looser filter, so the whole-block appendix can only
    shrink, never grow, relative to it. Because the refinement is additive,
    dropping a stray symbol at worst forgoes one appended block; it never removes
    baseline context.
    """
    if not issue_text:
        return frozenset()
    out: set = set()
    for span in _BACKTICK_SPAN_RE.finditer(issue_text):
        _collect_backtick_symbols(span.group(1), out)
    for match in _ISSUE_IDENT_RE.finditer(issue_text):
        token = match.group(0)
        if _is_named_code_symbol(token, issue_text, match.start(), match.end()):
            out.add(token)
    out -= _NOISE_TOKENS
    return frozenset(out)


def _leading_indent(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def _block_end(lines: list, start: int, indent: int) -> int:
    end = start
    total = len(lines)
    cursor = start + 1
    while cursor < total:
        line = lines[cursor]
        if line.strip() == "":
            cursor += 1
            continue
        if _leading_indent(line) <= indent:
            break
        end = cursor
        cursor += 1
        if end - start >= _MAX_BLOCK_LINES:
            break
    return end


def _extract_truncated_owning_blocks(
    content: str, symbols: "frozenset[str]", clip_len: int, budget: int
) -> str:
    """Complete def/class blocks the task names that the baseline clip truncated.

    Only blocks whose text extends past `clip_len` (i.e. the baseline did NOT
    already show them in full) are emitted, in file order, each with a
    `# lines A-B:` header. Returns "" when nothing qualifies, so the caller adds
    no appendix and the preload stays byte-identical to the baseline.
    """
    if not symbols:
        return ""
    lines = content.splitlines()
    starts: list = []  # approx char offset of each line start (exact for \n-only files)
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    spans: list = []  # (start_index, end_index_inclusive)
    for index, line in enumerate(lines):
        match = _DEF_LINE_RE.match(line)
        if not match or match.group(2) not in symbols:
            continue
        indent = _leading_indent(line)
        end = _block_end(lines, index, indent)
        end_offset = starts[end + 1] if end + 1 < len(starts) else pos
        if end_offset <= clip_len:
            continue  # the baseline clip already showed this block in full
        spans.append((index, end))
    if not spans:
        return ""
    spans.sort()
    merged: list = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    parts: list = []
    total = 0
    prev_end = -1
    for start, end in merged:
        if prev_end >= 0 and start > prev_end + 1:
            parts.append("# ... (unrelated code omitted) ...")
            total += 34
        header = f"# lines {start + 1}-{end + 1}:"
        segment = header + "\n" + "\n".join(lines[start : end + 1])
        if total + len(segment) + 1 > budget:
            if not parts:
                room = budget - (len(header) + 2)
                if room > 200:
                    partial = "\n".join(lines[start : end + 1])[:room]
                    parts.append(header + "\n" + partial + "\n# ... (block truncated) ...")
            break
        parts.append(segment)
        total += len(segment) + 1
        prev_end = end
    return "\n".join(parts)


def _existing_issue_files(issue: str, repo_dir: str, *, limit: int) -> list[str]:
    out: list[str] = []
    for match in _FILE_IN_ISSUE_RE.finditer(issue or ""):
        rel = match.group(1).strip().lstrip("./")
        if rel and rel not in out and os.path.isfile(os.path.join(repo_dir, rel)):
            out.append(rel)
        if len(out) >= limit:
            break
    return out


def _read_repo_file(repo_dir: str, relative_path: str) -> str:
    try:
        with open(os.path.join(repo_dir, relative_path), "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _fill_empty_with_located(preloaded_context: str, issue: str, repo_path: str) -> str:
    """When the literal-named-file preload is EMPTY, fill that slot with the
    DEFINITION site of a distinctive symbol/error-string the issue mentions
    (high-precision, 1-2 files, or "" if ambiguous). Never displaces a named
    file (only fires on the empty slot) -- and because the whole-block focused
    preload is empty EXACTLY when the baseline named-file preload is empty
    (its appendix only ever extends a non-empty baseline), this gate is
    byte-equivalent to the king's empty-slot condition: the locate path and the
    whole-block path are mutually exclusive by construction. Gated to the
    generous-budget regime. Fail-open: any problem returns the base preload
    unchanged -> exact tie."""
    if _build_located_context is None:
        return preloaded_context
    if WALL_CLOCK_LIMIT_SECONDS < _LOCATE_MIN_WALL_CLOCK_SECONDS:
        return preloaded_context
    if preloaded_context.strip():
        return preloaded_context
    try:
        located = _build_located_context(repo_path, issue)
    except Exception:
        return preloaded_context
    if located and located.strip():
        return located
    return preloaded_context


def _augment_with_companion(preloaded_context: str, issue: str, repo_path: str) -> str:
    """Append the multi-site companion export block to the primary preload
    (named-file/whole-block or located) when a near-certain companion is
    detected. Additive (a multi-site task may also name a file, so the primary
    preload is kept in full) and hard-capped by a defensive ceiling that is
    never reached in normal operation. Fail-open: any problem returns the base
    preload unchanged -> exact tie."""
    if _build_companion_context is None:
        return preloaded_context
    try:
        companion = _build_companion_context(repo_path, issue)
    except Exception:
        return preloaded_context
    if not companion or not companion.strip():
        return preloaded_context
    if not preloaded_context.strip():
        combined = companion
    else:
        combined = preloaded_context.rstrip() + "\n" + companion
    return combined[:_MAX_COMBINED_PRELOAD_CHARS]


def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        model_name, base_url, proxy_token = _resolve_inference_config(model, api_base, api_key)
        repo_summary = build_repo_summary(repo_path)
        preloaded_context = issue_named_context(issue, repo_path)
        # 1) Fill an EMPTY named-file slot with the located definition site
        #    (targets wrong-file cold-start). Keyed on the raw named context so a
        #    later companion append cannot suppress it. Mutually exclusive with
        #    the whole-block appendix above: that appendix exists only when a
        #    named file was preloaded (non-empty), and locate fires only when
        #    nothing was preloaded (empty).
        preloaded_context = _fill_empty_with_located(preloaded_context, issue, repo_path)
        # 2) Additively append the multi-site companion export block when the
        #    issue adds a new public symbol (targets incomplete-multi-site).
        preloaded_context = _augment_with_companion(preloaded_context, issue, repo_path)
        run_config = AgentRunConfig(
            repo_dir=repo_path,
            model_name=model_name,
            base_url=base_url,
            auth_token=proxy_token,
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_tokens=max_tokens,
            max_observation_chars=MAX_OBSERVATION_CHARS,
            max_log_chars=MAX_TOTAL_LOG_CHARS,
            max_message_chars=MAX_MESSAGE_CHARS,
            wall_clock_limit=WALL_CLOCK_LIMIT_SECONDS,
        )
        outcome = run_agent_loop(
            config=run_config,
            task=build_initial_user_prompt(issue, repo_summary, preloaded_context),
        )
        elapsed = time.monotonic() - started
        return {
            "patch": outcome.patch,
            "logs": outcome.logs,
            "steps": outcome.steps,
            "cost": outcome.cost,
            "success": outcome.success,
            "message": f"{outcome.exit_status}: {outcome.message} in {elapsed:.1f}s",
        }
    except Exception:
        fallback_patch = collect_repo_patch(repo_path)
        return {
            "patch": fallback_patch,
            "logs": traceback.format_exc()[-8000:],
            "steps": 0,
            "cost": None,
            "success": bool(fallback_patch.strip()),
            "message": "agent crashed; returning the on-disk repository diff",
        }
