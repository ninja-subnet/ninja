"""Post-loop REMOVE-ONLY patch hygiene (standard library only).

Runs ONCE after the agent step loop has fully exited. It takes no model turn and
adds no per-step cost, so it cannot change the model's trajectory, and it is a
byte-identity no-op on a clean, source-only patch.

The final patch is collected as the tracked ``git diff --binary`` PLUS a
no-index diff for every path in ``git ls-files --others --exclude-standard``
(see ``repo_diff.collect_repo_patch``; the validator's in-sandbox harness uses
byte-identical git commands, so those untracked files really do land in the
judged patch). A focused ``python -m py_compile`` or a ``pytest`` collection run
during the loop can leave ``__pycache__/*.pyc`` / cache state on disk even under
``PYTHONDONTWRITEBYTECODE=1`` (that var stops a plain ``python -c`` import from
writing bytecode, but ``py_compile`` / some tools still do); in a task repo whose
``.gitignore`` does not cover them, those files enter the judged diff as
unrelated "GIT binary patch" churn -- dead weight to the diff-only coverage
judge and against the patch-size cap.

This guard deletes, ONLY from that exact untracked list, files that PROVABLY
cannot be a task deliverable: compiled bytecode (``*.pyc`` / ``*.pyo``) and
anything living under a tool-cache directory (``__pycache__`` /
``.pytest_cache`` / ``.mypy_cache``). It has ZERO mis-fire surface: a
hand-authored deliverable is never compiled bytecode and never lives under a
bytecode-cache dir, a committed test/repro file (which the judge REWARDS) is
neither, and a tracked file can never appear in the untracked list -- so on a
clean tree the collected patch is byte-identical. It deliberately does NOT do
hunk-level trimming inside tracked files (a hunk cannot be proven unrelated
without the reference) and does NOT touch any source edit or test. Deterministic
(pure function of on-disk state; no time / randomness / network), stdlib-only,
fail-open: any error skips that file or aborts the whole cleanup, leaving the
tree exactly as the loop left it and the collected patch unchanged.
"""

import os
import subprocess

_GIT_TIMEOUT = 60

# Directory names that only ever hold tool-generated cache state.
_ARTIFACT_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
# Compiled-bytecode suffixes: never a deliverable, always generated.
_ARTIFACT_SUFFIXES = (".pyc", ".pyo")


def remove_untracked_artifacts(repo_dir):
    """Delete untracked bytecode/cache artifacts so they never enter the final
    diff. Only ever removes paths that appear in
    ``git ls-files --others --exclude-standard`` (the exact set
    ``collect_repo_patch`` diffs) AND are compiled bytecode (``*.pyc`` /
    ``*.pyo``) or live under a tool-cache directory. Fail-open per file and
    overall; never raises into the agent loop."""
    try:
        for rel in _untracked_files(repo_dir):
            try:
                if _is_artifact(rel):
                    os.remove(os.path.join(repo_dir, rel))
            except OSError:
                continue  # fail-open per file: leave it in the patch
    except Exception:
        return None  # fail-open: never raise into the agent loop
    return None


def _is_artifact(rel):
    parts = rel.replace("\\", "/").split("/")
    # (a) anything living under a cache directory
    if any(part in _ARTIFACT_DIR_NAMES for part in parts[:-1]):
        return True
    # (b) compiled bytecode anywhere
    return parts[-1].lower().endswith(_ARTIFACT_SUFFIXES)


def _untracked_files(repo_dir):
    try:
        done = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = done.stdout or ""
    return [item for item in out.split("\0") if item]
