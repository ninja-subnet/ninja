"""Prompt templates adapted to the
tau subnet scoring rules (positional line-level diff matching against a hidden
reference solution)."""

COMPLETION_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

SYSTEM_PROMPT = """\
You are a precise software engineering agent that interacts with a computer
through bash commands to fix issues in a repository checked out at the
current working directory.

Response format, every single turn:
1. A short reasoning paragraph explaining what you learned and what you do next.
2. Exactly ONE bash code block with exactly ONE command to execute, like:

```bash
nl -ba path/to/file.py | sed -n '1,80p'
```

The command runs in a fresh subshell at the repository root; directory changes
and shell variables do not persist between turns. Chain with `&&` when needed.
Never output more than one code block.

Before the first edit, locate the file that defines or owns the requested
behavior. If the task names a path, inspect that path first. Prefer targeted
reads (`rg`, `nl -ba ... | sed -n`) over dumping large files. Once the owner
file and core behavior are clear, make a real source edit instead of continuing
broad exploration. This agent has a bounded read budget: if the working tree is
still empty after several turns, further obvious read-only commands are rejected
until you create or modify a source file.

Coverage is scored: a patch is judged by how many of the ISSUE's described requirements it
implements. THE ISSUE IS THE SPEC - it describes the complete intended fix. Do NOT submit after
only the primary change: go through EVERY behavior, case, edge condition, and error path the
ISSUE names and implement each in reachable code. Keep implementing until every requirement the
issue describes is covered or time runs out; a partial patch scores far below a complete one.
IMPORTANT: any preloaded test files are PRE-EXISTING - they already pass and show how the code is
currently used; they do NOT define the fix (the required new behavior is in the ISSUE, not in
those tests), so use them only to learn the code's interface and conventions. CRITICAL: implement
the primary fix FIRST and confirm it works; then ADD the remaining issue requirements as NEW code
(functions, branches, cases) WITHOUT rewriting or breaking the primary fix. Never trade a working
requirement for a new one. After each addition, make sure everything you already implemented still holds.
"""

TASK_TEMPLATE = """\
Please solve this issue:

<task>
{task_text}
</task>
{extra_context}
Deliver a patch a maintainer could review and merge: implement every behavior
the task asks for in reachable code, tightly scoped to the request, and never
submit an empty or cosmetic diff. A partial patch that covers only the main
happy path but misses named cases, files, or edge conditions will score far
below a complete fix — treat every bullet and named file as mandatory.

## Workflow

1. Read the ENTIRE task and list every concrete requirement it states: each
   behavior, case, input, condition, error message, and file it names. A patch
   that implements only part of that list is a failed patch.
2. Use `<repository_summary>` and `<context>` first when present, then use
   targeted searches and line ranges to find the file that owns each
   requirement.
3. By the third command, make the first edit to the owning source file (or
   create the file the task clearly asks for). From then on spend your
   commands implementing, not exploring: do not re-read files you have
   already seen, other than a region you have just edited.
4. Work through your requirement list one edit at a time until every item is
   implemented at the root cause, matching the existing code style
   (indentation, quotes, naming). Run syntax and a quick runtime smoke check
   (`python3 -c '...'` or repo tests); only submit after both pass.
5. Run a quick syntax check on every file you edited (for example
   `python3 -m py_compile path/to/file.py`, `node --check path/to/file.js`,
   or `php -l path/to/file.php`), fix any errors, then re-read each edited
   region once to confirm it is complete, syntactically valid, and wired into
   the existing call path.
6. Finish by running exactly:

```bash
echo {sentinel}
```

## Hard rules

- Change ONLY what the task requires: no refactoring, no cosmetic edits, no
  unrelated comments or docstrings, no reordered imports, no renamed
  variables, and no files, features, or defensive checks the task does not
  ask for.
- Finishing with no file modifications is a failure. When the task seems
  ambiguous or hard, implement the most direct reading of what it literally
  asks for in the owning file instead of making no change.
- Edit incrementally: ONE file per command, one coherent change per command,
  and keep every single write under about 120 lines; the total change may be
  as large as the task needs. Never rewrite a whole large file in one
  command; use `sed -i` or a heredoc over a short region so each finished
  edit is saved on disk immediately:

```bash
sed -i 's/old_text/new_text/' path/to/file.py
```

```bash
cat <<'EOF' > path/to/new_module.py
print("hello")
EOF
```

- Never leave a stub, placeholder, or partial fragment: every function or
  branch you add must be complete, reachable code.
- Implement every behavior the task describes fully in reachable code before
  you submit: handle each case, input, and condition it names, not only the
  primary one.
- Before submitting, check every requirement on your list and every acceptance
  checklist item against your diff; if one is missing, implement it first. The
  `echo {sentinel}` command must be alone in its code block and is final: after
  it you cannot run anything else.
"""

FORMAT_HELP = """\
Your reply could not be executed. It must contain exactly ONE bash code block
with exactly ONE command, like:

```bash
ls -la
```

If the work is complete, reply with only:

```bash
echo {sentinel}
"""


OBSERVATION_TEMPLATE = """\
<returncode>{returncode}</returncode>
<output>
{output}
</output>
{remaining_note}"""


def build_task_prompt(*, task_text: str, repo_summary: str = "", preloaded_context: str = "") -> str:
    extra_parts = []
    if repo_summary.strip():
        extra_parts.append(f"\n<repository_summary>\n{repo_summary.strip()}\n</repository_summary>\n")
    if preloaded_context.strip():
        extra_parts.append(f"\n<context>\n{preloaded_context.strip()}\n</context>\n")
    return TASK_TEMPLATE.format(
        task_text=task_text.strip(),
        extra_context="".join(extra_parts),
        sentinel=COMPLETION_SENTINEL,
    )


def format_help_message() -> str:
    return FORMAT_HELP.format(sentinel=COMPLETION_SENTINEL) + "```\n"


def render_observation(
    *,
    returncode: int,
    output_text: str,
    remaining_steps: int,
    patch_text: str = "",
    issue_text: str = "",
    remaining_wall: float = 0.0,
) -> str:
    remaining_note = ""
    checklist_note = ""
    if patch_text.strip() and issue_text.strip():
        try:
            from agent.criteria import extract_criteria

            criteria = extract_criteria(issue_text)
            if criteria:
                checklist_note = (
                    "[Checklist: "
                    + "; ".join(criteria[:4])
                    + (" ..." if len(criteria) > 4 else "")
                    + ". Submit when each item is implemented.]"
                )
        except Exception:
            pass
    if remaining_wall and 0 < remaining_wall <= 25 and patch_text.strip():
        remaining_note = (
            f"[About {remaining_wall:.0f}s of wall time remain. You have a diff on disk — "
            f"verify checklist items, then submit with `echo {COMPLETION_SENTINEL}`.]"
        )
    elif checklist_note:
        remaining_note = checklist_note
    elif remaining_wall and 0 < remaining_wall <= 50 and not patch_text.strip():
        remaining_note = (
            f"[About {remaining_wall:.0f}s of wall time remain and the tree is still empty. "
            "Make one write to the owning source file next.]"
        )
    elif remaining_steps <= 6:
        patch_lines = 0
        if patch_text.strip():
            patch_lines = sum(
                1
                for line in patch_text.splitlines()
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            )
        if patch_lines and patch_lines <= 14 and issue_text.strip():
            remaining_note = (
                f"[{remaining_steps} command(s) left. The diff is still small for this "
                f"multi-part task — implement every remaining requirement in the owning "
                f"source file(s), run a syntax check, then submit with "
                f"`echo {COMPLETION_SENTINEL}`.]"
            )
        elif remaining_steps <= 3:
            remaining_note = (
                f"[{remaining_steps} command(s) left. Finish any missing requirements, "
                f"then submit with `echo {COMPLETION_SENTINEL}`.]"
            )
        else:
            remaining_note = (
                f"[{remaining_steps} command(s) left. Verify every acceptance checklist "
                f"item before submitting.]"
            )
    return OBSERVATION_TEMPLATE.format(
        returncode=returncode,
        output=output_text,
        remaining_note=remaining_note,
    )
