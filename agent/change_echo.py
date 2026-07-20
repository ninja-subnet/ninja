"""Echo what a command changed on disk back into its observation.

RCA (duel-20260720123807): on rounds the king wins big, the edit phase
degenerates into blind one-line ``sed -i`` probes followed by verification
greps -- ~10 LLM calls per losing task whose entire observation is an empty
``<output>``. ``sed -i`` prints nothing on success AND nothing when its
pattern matched nothing, so the model pays several round-trips per edit just
to learn whether it landed.

This closes that void deterministically: the loop snapshots the repo patch
before and after every potentially-writing command and appends the current
on-disk content of each changed region (with line numbers) to the
observation -- or an explicit "changed nothing" note when a write-shaped
command was a no-op. The patch delta is used only to locate the file and the
covering line ranges; the echoed lines are read back from the file itself, so
the model sees the ground truth that later tools will see -- including any
stray characters or mangled bytes an edit left behind -- not git's rendering
of the hunk.
"""

import re
import time
from pathlib import Path

from .repo_diff import collect_repo_patch

# Bounds the echo, not the change: a whole-file rewrite echoes its head and
# tail, which is enough to confirm the write landed intact.
_ECHO_CAP_CHARS = 2000
# One-way latency fuse for the pre/post snapshots. Measured over 54,818 real
# snapshot invocations across the task pools: p50 10ms, p99 311ms -- but three
# tasks run 3-247s per snapshot, where two snapshots per command would dwarf
# the LLM calls themselves. One slow snapshot disables the echo for the rest
# of the run (the loop simply behaves as before this feature existed).
_SNAPSHOT_BUDGET_SECONDS = 2.0
# Changed ranges in one file closer than this merge into one covering block.
_MERGE_GAP_LINES = 2

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

ECHO_HEADER = "[Changed on disk -- current content of the affected lines:]"

NOTHING_CHANGED_NOTE = (
    "[This command changed nothing on disk. If it was meant to edit, its "
    "pattern or anchor did not match the file; re-read the exact target "
    "lines and retry with a corrected match.]"
)

# Sent to the model ONCE when the fuse trips: the system prompt promises an
# echo after every file-modifying command, so going silent would read as
# "nothing changed". The note hands verification back to the model.
FUSE_NOTE = (
    "[Change echoes are disabled for the rest of this run: this repository is "
    "too slow to snapshot. You will no longer be told what your commands "
    "changed; verify each edit by re-reading the edited region.]"
)


class PatchSnapshotter:
    """``collect_repo_patch`` behind the latency fuse: ``snapshot`` returns
    the patch, or None forever once any snapshot exceeded the budget."""

    def __init__(self, budget_seconds: float = _SNAPSHOT_BUDGET_SECONDS,
                 clock=time.monotonic, collect=collect_repo_patch):
        self._budget = budget_seconds
        self._clock = clock
        self._collect = collect
        self.fused = False

    def snapshot(self, repo_dir: str):
        if self.fused:
            return None
        started = self._clock()
        patch = self._collect(repo_dir)
        if self._clock() - started > self._budget:
            self.fused = True
            return None
        return patch


def build_change_echo(pre_patch: str, post_patch: str, *, write_shaped: bool,
                      repo_dir: str) -> str:
    """Render what changed between two repo-patch snapshots as the current
    on-disk content of the affected line ranges. Returns "" for a no-op by a
    command that never looked like a write (e.g. a python script that only
    read)."""
    if pre_patch == post_patch:
        return NOTHING_CHANGED_NOTE if write_shaped else ""
    pre_files = _split_file_sections(pre_patch)
    post_files = _split_file_sections(post_patch)
    parts = []
    for path, section in post_files.items():
        if pre_files.get(path) == section:
            continue
        parts.extend(
            _render_file_change(path, pre_files.get(path, ""), section, repo_dir)
        )
    for path in pre_files:
        if path not in post_files:
            parts.append(f"{path}: restored to its original content")
    if not parts:
        return NOTHING_CHANGED_NOTE if write_shaped else ""
    return ECHO_HEADER + "\n" + _cap_lines(parts, _ECHO_CAP_CHARS)


def _cap_lines(parts: list, cap: int) -> str:
    """Head/tail elision on WHOLE lines: a numbered line cut mid-text would
    itself read as a stray-character defect, defeating the echo's purpose."""
    text = "\n".join(parts)
    if len(text) <= cap:
        return text
    head, used = [], 0
    for line in parts:
        if used + len(line) > cap // 2:
            break
        head.append(line)
        used += len(line) + 1
    tail, used = [], 0
    for line in reversed(parts[len(head):]):
        if used + len(line) > cap // 2:
            break
        tail.append(line)
        used += len(line) + 1
    elided = len(parts) - len(head) - len(tail)
    return "\n".join(head + [f"[... {elided} lines elided ...]"] + list(reversed(tail)))


def _split_file_sections(patch: str) -> dict:
    """Split a git patch into {path: its full per-file section}."""
    sections = {}
    path, lines = None, []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            if path is not None:
                sections[path] = "\n".join(lines)
            path = _diff_header_path(line)
            lines = []
        lines.append(line)
    if path is not None:
        sections[path] = "\n".join(lines)
    return sections


def _diff_header_path(line: str) -> str:
    """b-side path of a ``diff --git`` header. Non-ASCII paths arrive
    quoted with octal escapes (``"b/Pesta\\303\\261as/x.cs"``); unquote
    them or the disk read behind the echo fails."""
    if ' "b/' in line:
        quoted = line.rsplit(' "b/', 1)[-1].rstrip()
        if quoted.endswith('"'):
            quoted = quoted[:-1]
        try:
            return (quoted.encode("ascii").decode("unicode_escape")
                    .encode("latin-1").decode("utf-8"))
        except (UnicodeError, ValueError):
            return quoted
    if " b/" in line:
        return line.rsplit(" b/", 1)[-1]
    return line


def _render_file_change(path: str, pre_section: str, post_section: str,
                        repo_dir: str) -> list:
    if "GIT binary patch" in post_section:
        return [f"{path}: binary file changed"]
    if "\n+++ /dev/null" in post_section:
        return [f"{path}: deleted"]
    # Hunks whose body already existed before this command are unchanged and
    # merely line-shifted by an edit earlier in the file; echoing them again
    # would misreport old edits as this command's work.
    pre_bodies = {tuple(body) for _, body in _hunks(pre_section)}
    ranges = [
        _new_side_range(start, body)
        for start, body in _hunks(post_section)
        if tuple(body) not in pre_bodies
    ]
    if not ranges:
        return [f"{path}: changed (an earlier edit was reverted or moved)"]
    return _read_ranges(path, _merge_ranges(ranges), repo_dir)


def _hunks(section: str) -> list:
    """[(new_side_start_line, [raw body lines]), ...] for one file section."""
    hunks = []
    start, body = None, []
    for line in section.splitlines():
        match = _HUNK_HEADER_RE.match(line)
        if match:
            if start is not None:
                hunks.append((start, body))
            start, body = int(match.group(1)), []
        elif start is not None and line[:1] in (" ", "+", "-", "\\"):
            body.append(line)
    if start is not None:
        hunks.append((start, body))
    return hunks


def _new_side_range(new_start: int, body: list) -> tuple:
    length = sum(1 for raw in body if raw[:1] in (" ", "+"))
    return new_start, max(new_start, new_start + length - 1)


def _merge_ranges(ranges: list) -> list:
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + _MERGE_GAP_LINES + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _read_ranges(path: str, ranges: list, repo_dir: str) -> list:
    """The affected ranges as they exist on disk right now, numbered like
    ``nl -ba`` output. Undecodable bytes render as U+FFFD so a corrupted
    write is visible instead of silently prettified."""
    try:
        file_lines = (Path(repo_dir) / path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return [f"{path}:{start}-{end}: (file unreadable)" for start, end in ranges]
    out = []
    for start, end in ranges:
        end = min(end, len(file_lines))
        out.append(f"{path}:{start}-{end}")
        out.extend(
            f"{lineno:>6}\t{file_lines[lineno - 1]}"
            for lineno in range(start, end + 1)
        )
    return out
