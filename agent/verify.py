"""Pre-submit verification, syntax checks, and patch sanitization."""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Optional

from agent.prompts import COMPLETION_SENTINEL

try:
    from agent.criteria import MULTI_CRITERIA_MIN, extract_criteria, format_checklist
except Exception:
    MULTI_CRITERIA_MIN = 2

    def extract_criteria(_issue: str) -> list:
        return []

    def format_checklist(_criteria: list[str]) -> str:
        return ""

WALL_CLOCK_RESERVE_SECONDS = 10.0
VERIFY_REPAIR_SYNTAX_MIN_BUDGET_SECONDS = 30.0
VERIFY_REPAIR_MAX_STEPS = 16

_MUNGE_PATH_RE = re.compile(
    r"^(?:fix|clean|cleanup|replace|update|patch|apply|munge|modify|gen|generate|"
    r"rewrite|migrate|refactor)_[\w.-]+$",
    re.I,
)
_MUNGE_FILE_RE = re.compile(
    r"^(?:fix|update|replace|refactor|patch|apply|clean|generate|rewrite|migrate|"
    r"modify)_[\w.-]+\.(?:py|sh|js|ts|rb|pl)$",
    re.I,
)
_REFACTOR_ISSUE_RE = re.compile(
    r"\b(refactor|rename|restructur|convert|migrate|reorganiz)\b",
    re.I,
)
_FILE_IN_ISSUE_RE = re.compile(
    r"`?([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cs|rb|php|vue|html|css|json|yaml|yml|md|cpp|h|c|hpp|toml|xml|sql|sh|txt))`?",
    re.I,
)

_BRACE_BALANCE_EXTS = (".php", ".cs", ".kt", ".java", ".swift", ".scala")
_DELIM_OPEN = {")": "(", "]": "[", "}": "{"}
_DUP_DEF_EXTS = (
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".php", ".cs",
    ".kt", ".java", ".go", ".swift", ".scala", ".rs",
)
_CS_REPEATED_BASE_RE = re.compile(
    r"\b(?:class|interface|struct|record)\s+[A-Za-z_]\w*(?:\s*<[^>]*>)?"
    r"\s*:\s*([A-Za-z_][\w.]*)(?:\s*:\s*\1\b)+"
)
_DUP_DEF_RE = re.compile(
    r"^[ \t]*"
    r"(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|final\s+|abstract\s+|async\s+)*"
    r"(?:"
    r"(?:function|def|class|interface|struct|enum|type)\s+(\w+)"
    r"|(?:const|let|var)\s+(\w+)\s*="
    r")",
    re.M,
)
_JAVA_METHOD_RE = re.compile(
    r"(?:public|private|protected|static|final|abstract|synchronized|native|default)"
    r"[\w<>\[\],\s.@]*?\b(\w+)\s*\(([^;{}]*)\)\s*(?:throws[\w\s,.]*)?\{"
)
_KOTLIN_FUN_RE = re.compile(r"\bfun\s+(?:<[^>]*>\s*)?(\w+)\s*\(([^)]*)\)")
_N_ARTIFACT_RE = re.compile(r"^n(?:[ \t]{2,}(?://|#|/\*|[A-Za-z{}])|//|/\*)")
_DOLLAR_PH_RE = re.compile(r"(?<![\w$])\$[1-9]\b")
_DOLLAR_PH_EXTS = (
    ".rs", ".kt", ".java", ".go", ".cpp", ".cc", ".cxx", ".hpp",
    ".h", ".c", ".cs", ".swift", ".scala", ".py",
)
_STRINGY_RE = re.compile(r"\"([^\"\\]|\\.)*\"|'([^'\\]|\\.)*'|`[^`]*`")
_AUTOFAIL_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bas an ai (?:language )?model\b",
        r"\bi(?:'m| am) (?:sorry|unable|not able)\b",
        r"\bi cannot (?:assist|help|comply|complete|fulfill)\b",
        r"\bi can(?:'|no)t (?:assist|help|comply|complete|fulfill) with\b",
        r"\bi['\u2019]?m sorry,? but\b",
        r"\bplaceholder (?:value|logic|implementation)\b",
        r"\bto[_ ]be[_ ]determined\b",
        r"#\s*todo:\s*implement\b",
        r"\bnot implemented\b.*\bplaceholder\b",
    )
]


def patch_acceptable(patch_text: str) -> bool:
    if not patch_text.strip():
        return False
    if destructive_patch_reason(patch_text) or munge_artifact_reason(patch_text):
        return False
    return True


def destructive_patch_reason(patch_text: str) -> Optional[str]:
    added, removed = _line_stats(patch_text)
    if removed >= 60 and added < max(5, removed // 4):
        return (
            f"the patch removes far more than it adds ({removed} deletions vs {added} additions); "
            "restore required logic instead of gutting the codebase"
        )
    return None


def refactor_delete_reason(issue_text: str, patch_text: str) -> Optional[str]:
    if not _REFACTOR_ISSUE_RE.search(issue_text or ""):
        return None
    added, removed = _line_stats(patch_text)
    if removed >= 30 and added < max(8, removed // 3):
        return (
            f"refactor/rename task but patch mostly deletes code "
            f"({removed} deletions vs {added} additions); implement the change in place"
        )
    return None


def task_coverage_reason(
    issue_text: str,
    patch_text: str,
    repo_dir: Optional[str] = None,
) -> Optional[str]:
    mentioned: list[str] = []
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text or ""):
        path = match.group(1).strip().lstrip("./")
        if path and path not in mentioned:
            mentioned.append(path)
    if not mentioned:
        return None
    touched = _all_changed_files(patch_text)
    if not touched:
        return None
    if repo_dir is not None:
        valid: list[str] = []
        for rel in mentioned:
            exists = os.path.isfile(os.path.join(repo_dir, rel))
            hit = any(t == rel or t.endswith("/" + rel) or rel.endswith("/" + t) for t in touched)
            if exists or hit:
                valid.append(rel)
        mentioned = valid
    if not mentioned:
        return None
    hit_count = sum(
        1
        for rel in mentioned
        if any(t == rel or t.endswith("/" + rel) or rel.endswith("/" + t) for t in touched)
    )
    if hit_count == 0:
        sample = ", ".join(mentioned[:6])
        return (
            f"the task names specific files ({sample}) but the patch does not touch any of them; "
            "find and edit the correct targets"
        )
    return None


def munge_artifact_reason(patch_text: str) -> Optional[str]:
    for path in _all_changed_files(patch_text):
        base = path.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0] if "." in base else base
        if (
            _MUNGE_PATH_RE.match(stem)
            or _MUNGE_FILE_RE.match(base)
            or base.endswith((".new", ".bak", ".orig", ".tmp", ".rej"))
        ):
            return (
                f"the patch adds scratch or munge artifact `{path}`; "
                "edit source files directly and remove helper/backup files"
            )
    return None


def _line_stats(patch_text: str) -> tuple[int, int]:
    added = removed = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def submit_readiness_message(
    repo_dir: str,
    patch_text: str,
    issue_text: str = "",
) -> Optional[str]:
    blocked = submit_readiness_light(repo_dir, patch_text)
    if blocked:
        return blocked
    if not source_files(patch_text):
        impl_hint = re.search(
            r"\b(fix|implement|add|update|change|handle|support|wire|create|modify|refactor)\b",
            issue_text or "",
            re.I,
        )
        if impl_hint or not added_test_files(patch_text):
            return _source_only_submit_guard_message()
    return None


def submit_readiness_light(repo_dir: str, patch_text: str) -> Optional[str]:
    """Empty, destructive, corruption, and syntax gates — no source-only rejection."""
    if not patch_text.strip():
        return _empty_submit_guard_message()
    if not patch_acceptable(patch_text):
        return (
            "[Submit rejected: patch contains destructive or corrupted edits.]\n\n"
            "Narrow the change to a minimal targeted fix, then submit again."
        )
    corrupt = patch_corruption_error(patch_text, repo_dir)
    if corrupt:
        return (
            "[Submit rejected: mechanical patch corruption detected.]\n\n"
            f"{corrupt}\n\nRewrite the affected region with a heredoc, then submit again."
        )
    broken = syntax_errors(repo_dir, patch_text)
    if broken:
        return _syntax_submit_guard_message(broken)
    return None


_RUNTIME_VERIFY_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:python3?\s+-m\s+py_compile\b|python3?\s+-c\b|pytest\b|"
    r"node\s+--check\b|php\s+-l\b|go\s+(?:test|build)\b|cargo\s+(?:test|check)\b)",
    re.I,
)
_INVOKE_CALL_RE = re.compile(r"`(\w+)\(([^)]*)\)`")
_ASSERT_CALL_RE = re.compile(
    r"\b(\w+)\(([^)]*)\)\s*(?:==|must(?:\s+return)?|should(?:\s+return)?)\s*([^\n`.]+)",
    re.I,
)


def issue_named_file_count(issue_text: str) -> int:
    seen: set[str] = set()
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text or ""):
        rel = (match.group(1) or "").strip().lstrip("./")
        if rel:
            seen.add(rel)
    return len(seen)


def is_multifile_issue(issue_text: str) -> bool:
    return issue_named_file_count(issue_text) >= 2


def command_is_runtime_verify(command: str) -> bool:
    return bool(_RUNTIME_VERIFY_RE.search((command or "").strip()))


def runtime_smoke_errors(repo_dir: str, patch_text: str, issue_text: str = "") -> list[str]:
    if not (patch_text or "").strip() or submit_readiness_light(repo_dir, patch_text):
        return []
    errors: list[str] = []
    for rel in _changed_source_files(patch_text, (".py",)):
        err = _python_import_smoke(repo_dir, rel)
        if err:
            errors.append(err)
    invoke = _build_issue_invoke_smoke(repo_dir, patch_text, issue_text)
    if invoke:
        err = _run_check(["python3", "-c", invoke], repo_dir)
        if err:
            errors.append(f"runtime smoke failed: {err}")
    return errors


def runtime_submit_guard_message(
    repo_dir: str,
    patch_text: str,
    issue_text: str = "",
    *,
    runtime_verified: bool = False,
) -> Optional[str]:
    if runtime_verified:
        return None
    errors = runtime_smoke_errors(repo_dir, patch_text, issue_text)
    if not errors:
        return None
    detail = "\n- ".join(errors[:6])
    return (
        "[Submit rejected: runtime smoke failed.]\n\n"
        f"- {detail}\n\n"
        "Fix the logic error in the owning source file, run a quick runtime check "
        f"(for example `python3 -c '...'` or `python3 -m py_compile <file>`), then submit with "
        f"`echo {COMPLETION_SENTINEL}`."
    )


def submit_readiness_for_submit(
    repo_dir: str,
    patch_text: str,
    issue_text: str = "",
    *,
    runtime_verified: bool = False,
) -> Optional[str]:
    blocked = submit_readiness_light(repo_dir, patch_text) or criteria_submit_message(
        issue_text, patch_text
    )
    if blocked:
        return blocked
    return runtime_submit_guard_message(
        repo_dir, patch_text, issue_text, runtime_verified=runtime_verified
    )


def patch_passes_runtime(repo_dir: str, patch_text: str, issue_text: str = "") -> bool:
    return not runtime_smoke_errors(repo_dir, patch_text, issue_text)


def _python_import_smoke(repo_dir: str, rel: str) -> Optional[str]:
    code = f"import runpy; runpy.run_path({rel!r}, run_name='__agent_smoke__')"
    err = _run_check(["python3", "-c", code], repo_dir)
    return f"{rel}: import failed: {err}" if err else None


def _build_issue_invoke_smoke(repo_dir: str, patch_text: str, issue_text: str) -> str:
    py_files = _changed_source_files(patch_text, (".py",))
    if not py_files:
        return ""
    primary = _primary_changed_file(py_files, issue_text)
    modname = os.path.splitext(os.path.basename(primary))[0]
    stmts: list[str] = []
    for match in _INVOKE_CALL_RE.finditer(issue_text or ""):
        func, args = match.group(1), match.group(2)
        stmts.append(f"from {modname} import {func}; {func}({args})")
    for match in _ASSERT_CALL_RE.finditer(issue_text or ""):
        func, args, expected = match.group(1), match.group(2), match.group(3).strip()
        if expected:
            stmts.append(f"from {modname} import {func}; assert {func}({args}) == {expected}")
    if not stmts:
        for sym in _issue_symbols(issue_text)[:6]:
            if sym == modname or "." in sym:
                continue
            stmts.append(
                f"from {modname} import {sym}\n"
                f"if callable({sym}):\n"
                f"    {sym}(0)\n"
                f"    {sym}(1)\n"
                f"    {sym}(0, 1)\n"
            )
    return "\n".join(dict.fromkeys(stmts))


def _primary_changed_file(py_files: list[str], issue_text: str) -> str:
    named = []
    for match in _FILE_IN_ISSUE_RE.finditer(issue_text or ""):
        rel = (match.group(1) or "").strip().lstrip("./")
        if rel in py_files:
            return rel
        if rel:
            named.append(rel)
    for rel in py_files:
        if any(rel.endswith(os.path.basename(name)) or rel == name for name in named):
            return rel
    return py_files[0]


def repair_reason(
    repo_dir: str,
    patch_text: str,
    issue_text: str = "",
) -> Optional[tuple[str, str]]:
    if not (patch_text or "").strip():
        return ("empty", "the current change set is empty; no fix was produced yet")
    corrupt = patch_corruption_error(patch_text, repo_dir)
    if corrupt:
        return ("corruption", corrupt)
    broken = syntax_errors(repo_dir, patch_text)
    if broken:
        return (
            "syntax",
            "the edited files contain syntax errors that must be fixed:\n- "
            + "\n- ".join(broken[:8]),
        )
    quality = destructive_patch_reason(patch_text) or munge_artifact_reason(patch_text)
    if quality:
        return ("quality", quality)
    refactor = refactor_delete_reason(issue_text, patch_text)
    if refactor:
        return ("quality", refactor)
    coverage = task_coverage_reason(issue_text, patch_text, repo_dir)
    if coverage:
        return ("coverage", coverage)
    return None


def build_repair_task(issue_text: str, reason: str) -> str:
    return (
        "A previous attempt to solve the task below left the repository in an "
        "incomplete or broken state. " + reason + "\n\n"
        "Inspect the current state of the repository, then finish and correct "
        "the change so it fully and correctly solves the task. Re-read each "
        "edited region to confirm it is syntactically valid before submitting.\n\n"
        "Original task:\n" + issue_text
    )


def build_completion_task(issue_text: str) -> str:
    checklist = format_checklist(extract_criteria(issue_text))
    body = (
        "An initial patch exists for the task below, but it may only cover part "
        "of the requirements. Re-read the task and every acceptance checklist item. "
        "Implement any missing behavior in the files you already edited, run a "
        "quick syntax check on each edited file, then submit the complete patch.\n\n"
        "Do not restart from scratch, do not refactor unrelated code, and do not "
        "remove working changes.\n\n"
        "Original task:\n" + issue_text
    )
    return body + checklist if checklist else body


def patch_change_lines(patch_text: str) -> int:
    return sum(
        1
        for line in (patch_text or "").splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )


def criteria_submit_message(issue_text: str, patch_text: str) -> Optional[str]:
    """Lightweight pre-submit gate: named symbols from the task should appear in the diff."""
    criteria = extract_criteria(issue_text)
    if len(criteria) < MULTI_CRITERIA_MIN:
        return None
    symbols = _issue_symbols(issue_text)
    if len(symbols) < 2:
        return None
    added = "\n".join(
        line[1:]
        for line in (patch_text or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    hits = sum(1 for sym in symbols if re.search(r"\b" + re.escape(sym) + r"\b", added))
    if hits >= 1 or len(symbols) < 3:
        return None
    missing = ", ".join(f"`{sym}`" for sym in symbols if sym not in added[:8000])[:240]
    return (
        "[Submit rejected: the diff may not yet implement named task symbols.]\n\n"
        f"These symbols from the task are not clearly present in your added lines: {missing}.\n\n"
        "Implement the missing requirement(s) in the owning source file, then submit again."
    )


def _issue_symbols(issue_text: str) -> list[str]:
    out: list[str] = []
    for match in re.finditer(r"`([A-Za-z_][\w.]*)`", issue_text or ""):
        sym = match.group(1)
        if len(sym) >= 3 and sym not in out:
            out.append(sym)
    for match in re.finditer(
        r"\b(?:class|function|def|method|module)\s+([A-Za-z_]\w+)",
        issue_text or "",
        re.I,
    ):
        sym = match.group(1)
        if sym not in out:
            out.append(sym)
    return out[:8]


def partial_submit_message(issue_text: str, criteria_count: int) -> str:
    return (
        "[Submit rejected: the current diff looks too small for a multi-requirement task.]\n\n"
        f"The task lists about {criteria_count} concrete requirements, but the working "
        "tree diff is still very small. Implement the remaining requirements in the "
        "owning source file(s), verify each checklist item, then submit again."
    )


def recovery_prompt(issue: str) -> str:
    issue_lower = issue.lower()
    if any(x in issue_lower for x in [".go", "golang", " go ", "goroutine", "sync.", "chan "]):
        lang_hint = (
            "This is a Go task. In 3 steps: "
            "(1) grep for the most relevant .go source file, "
            "(2) read that file, "
            "(3) make ONE minimal edit to address the core issue and submit. "
            "Single file, single logical change only."
        )
    elif any(x in issue_lower for x in [".cpp", ".hpp", "c++", "cmake"]):
        lang_hint = (
            "This is a C++ task. In 3 steps: "
            "(1) grep for the relevant .cpp/.h file, "
            "(2) read it, "
            "(3) make ONE targeted change and submit."
        )
    elif any(x in issue_lower for x in [".ts", ".tsx", "typescript"]):
        lang_hint = (
            "This is a TypeScript task. In 3 steps: "
            "(1) find the relevant .ts file, "
            "(2) read the affected class/function, "
            "(3) make ONE precise change and submit."
        )
    else:
        lang_hint = (
            "In 3 steps: (1) find the most relevant file, "
            "(2) read it, (3) make ONE targeted fix and submit."
        )
    return "The repository has no changes yet. " + lang_hint + "\n\nOriginal task:\n" + issue


def sanitize_patch(patch_text: str) -> str:
    try:
        if not patch_text or not patch_text.strip():
            return patch_text
        lines = patch_text.splitlines(keepends=True)
        out = []
        removed_any = False
        kept_real_addition = False
        for ln in lines:
            body = ln.rstrip("\n")
            if body.startswith("+") and not body.startswith("+++"):
                added_content = body[1:]
                if _line_is_autofail(added_content):
                    removed_any = True
                    continue
                if added_content.strip():
                    kept_real_addition = True
            out.append(ln)
        if not removed_any:
            return patch_text
        if not kept_real_addition:
            return patch_text
        return "".join(out)
    except Exception:
        return patch_text


def syntax_errors(repo_dir: str, patch_text: str) -> list:
    broken = []
    for rel in _changed_source_files(patch_text, (".py",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            broken.append(f"{rel}: line {exc.lineno}: {exc.msg}")
        except (ValueError, TypeError):
            broken.append(f"{rel}: could not be parsed")
    for rel in _changed_source_files(patch_text, (".json",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        try:
            json.loads(content)
        except ValueError as exc:
            broken.append(f"{rel}: invalid JSON: {str(exc)[:120]}")
    for rel in _changed_source_files(patch_text, (".js", ".mjs", ".cjs")):
        err = _run_check(["node", "--check", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    for rel in _changed_source_files(patch_text, (".go",)):
        err = _run_check(["gofmt", "-e", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    for rel in _changed_source_files(patch_text, _BRACE_BALANCE_EXTS):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        err = _delimiter_balance_error(text, rel)
        if err:
            broken.append(err)
    for rel in _changed_source_files(patch_text, _DUP_DEF_EXTS):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        err = _duplicate_definition_error(text, rel)
        if err:
            broken.append(err)
    for rel in _changed_source_files(patch_text, (".php",)):
        err = _run_check(["php", "-l", rel], repo_dir)
        if err:
            broken.append(f"{rel}: {err}")
    for rel in _changed_source_files(patch_text, (".cs",)):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if _CS_REPEATED_BASE_RE.search(_strip_code_noise(text)):
            broken.append(f"{rel}: malformed repeated base type (e.g. ': X : X')")
    for rel in _changed_source_files(patch_text, (".java", ".kt")):
        full = os.path.join(repo_dir, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        err = _duplicate_method_error(text, rel)
        if err:
            broken.append(err)
    return broken


def patch_corruption_error(patch_text: str, repo_dir: str) -> Optional[str]:
    try:
        cur_path = ""
        cur_ext = ""
        blank_run = 0
        per_file_add = {}
        per_file_rem = {}
        for raw in patch_text.splitlines():
            if raw.startswith("+++ b/"):
                cur_path = raw[len("+++ b/"):].strip()
                cur_ext = os.path.splitext(cur_path)[1].lower()
                blank_run = 0
                continue
            if raw.startswith("+") and not raw.startswith("+++"):
                body = raw[1:]
                if _N_ARTIFACT_RE.match(body):
                    return (
                        "a stray 'n' (a mangled newline from a broken sed/heredoc) "
                        "starts an added line in " + (cur_path or "the patch")
                    )
                if cur_ext in _DOLLAR_PH_EXTS and _DOLLAR_PH_RE.search(_STRINGY_RE.sub('""', body)):
                    return (
                        "a leaked sed backreference placeholder ($1/$2) is in an added "
                        "line of " + (cur_path or "the patch")
                    )
                if cur_path:
                    per_file_add[cur_path] = per_file_add.get(cur_path, 0) + (1 if body.strip() else 0)
                if body.strip():
                    blank_run = 0
                else:
                    blank_run += 1
                    if blank_run >= 10:
                        return (
                            "a block of " + str(blank_run) + "+ blank lines was added to "
                            + (cur_path or "the patch")
                        )
                continue
            if raw.startswith("-") and not raw.startswith("---"):
                if cur_path:
                    per_file_rem[cur_path] = per_file_rem.get(cur_path, 0) + 1
            blank_run = 0
        for path, rem in per_file_rem.items():
            if rem >= 3 and per_file_add.get(path, 0) == 0:
                full = os.path.join(repo_dir, path)
                try:
                    if os.path.isfile(full) and os.path.getsize(full) == 0:
                        return "the file " + path + " was emptied (all content removed)"
                except OSError:
                    continue
    except Exception:
        return None
    return None


def source_files(patch_text: str) -> set:
    return {p for p in _all_changed_files(patch_text) if not _is_test_path(p)}


def added_test_files(patch_text: str) -> list:
    return [p for p in _all_changed_files(patch_text) if _is_test_path(p)]


def _empty_submit_guard_message() -> str:
    return (
        "[Submit rejected: the repository has no changes on disk yet.]\n\n"
        "You ran the completion command but the working tree diff is empty -- no file was "
        "created or modified. Edit or create at least one real source file for this task, "
        f"then run `echo {COMPLETION_SENTINEL}` again when the fix is on disk."
    )


def _syntax_submit_guard_message(errors: list) -> str:
    joined = "\n- ".join(errors[:6])
    return (
        "[Submit rejected: edited files have syntax or parse errors.]\n\n"
        "Fix every error below before submitting. Run compile/check on each edited file, "
        f"correct the code, then run `echo {COMPLETION_SENTINEL}` again.\n\n- {joined}"
    )


def _source_only_submit_guard_message() -> str:
    return (
        "[Submit rejected: the diff only touches tests or non-source files.]\n\n"
        "This task needs a real implementation change. Edit the source file(s) that own "
        f"the required behavior, then run `echo {COMPLETION_SENTINEL}` again."
    )


def _all_changed_files(patch_text: str) -> list:
    out = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            p = line[len("+++ b/"):].strip()
            if p and p != "/dev/null" and p not in out:
                out.append(p)
    return out


def _is_test_path(path: str) -> bool:
    p = path.lower()
    base = p.rsplit("/", 1)[-1]
    if any(seg in ("test", "tests", "spec", "specs", "__tests__") for seg in p.split("/")[:-1]):
        return True
    if base.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py") or base.startswith("test")):
        return True
    if ".test." in base or ".spec." in base or base.endswith("_spec.rb") or base.endswith("_test.go"):
        return True
    return False


def _changed_source_files(patch_text: str, exts: tuple) -> list:
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path.endswith(exts) and path not in paths:
                paths.append(path)
    return paths


def _run_check(cmd: list, cwd: str) -> Optional[str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return None
    msg = (proc.stderr or proc.stdout or "").strip()
    return msg.splitlines()[0][:200] if msg else "failed syntax check"


def _line_is_autofail(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    for pat in _AUTOFAIL_PATTERNS:
        m = pat.search(stripped)
        if m and (m.end() - m.start()) >= max(8, int(0.4 * len(stripped))):
            return True
    return False


def _strip_code_noise(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                return ""
            i = j + 2
            continue
        if c in "'\"`":
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            else:
                return ""
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _delimiter_balance_error(text: str, rel: str):
    if "<<<" in text:
        return None
    code = _strip_code_noise(text)
    if not code:
        return None
    stack = []
    for ch in code:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            want = _DELIM_OPEN[ch]
            if not stack:
                return f"{rel}: unexpected closing '{ch}' (extra/dangling delimiter)"
            top = stack.pop()
            if top != want:
                return f"{rel}: mismatched '{ch}' (expected close for '{top}')"
    if stack:
        return f"{rel}: {len(stack)} unclosed '{stack[-1]}' delimiter(s) (missing close brace/paren)"
    return None


def _duplicate_definition_error(text: str, rel: str):
    code = _strip_code_noise(text)
    if not code:
        return None
    seen = {}
    for mobj in _DUP_DEF_RE.finditer(code):
        name = mobj.group(1) or mobj.group(2)
        if not name:
            continue
        seen[name] = seen.get(name, 0) + 1
    dups = sorted(n for n, c in seen.items() if c > 1)
    if dups:
        return (
            f"{rel}: duplicate top-level definition(s): {', '.join(dups[:4])} "
            "(defined more than once -> compile error)"
        )
    return None


def _duplicate_method_error(text: str, rel: str):
    code = _strip_code_noise(text)
    if not code:
        return None
    rex = _KOTLIN_FUN_RE if rel.endswith(".kt") else _JAVA_METHOD_RE
    seen = {}
    for m in rex.finditer(code):
        params = re.sub(r"\s+", "", m.group(2) or "")
        if not params:
            continue
        key = m.group(1) + "(" + params + ")"
        seen[key] = seen.get(key, 0) + 1
    dups = sorted(k.split("(")[0] for k, c in seen.items() if c > 1)
    if dups:
        return (
            f"{rel}: duplicate method definition(s): {', '.join(dups[:4])} "
            "(same signature defined more than once -> compile error)"
        )
    return None
