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

After any command that modifies files, the observation is followed by
`[Changed on disk ...]`: the new content of every changed region with line
numbers. Trust it -- a successful edit never needs a verification read. If it
instead reports the command changed nothing, your edit's pattern or anchor
missed the file.
"""

TASK_TEMPLATE = """\
Please solve this issue:

<task>
{task_text}
</task>
{extra_context}
Deliver a patch a maintainer could review and merge: implement every behavior
the task asks for in reachable code, tightly scoped to the request, and never
submit an empty or cosmetic diff.

## Workflow

1. Read the ENTIRE task and list every concrete requirement it states: each
   behavior, case, input, condition, error message, and file it names. A patch
   that implements only part of that list is a failed patch.
2. Use `<repository_summary>` and `<context>` first when present, then use
   targeted searches and line ranges to find the file that owns each
   requirement.
3. By the fourth command, make the first edit to the owning source file (or
   create the file the task clearly asks for). From then on spend your
   commands implementing, not exploring: do not re-read files you have
   already seen, other than a region you have just edited.
4. Work through your requirement list one edit at a time until every item is
   implemented at the root cause, matching the existing code style
   (indentation, quotes, naming). Do not stop at the first working state.
5. Confirm each edit from the `[Changed on disk ...]` echo that follows it --
   complete, syntactically valid, correctly placed -- instead of re-reading
   the file. If the echo says the command changed nothing, the edit missed:
   re-read the exact target lines and retry with a corrected match. Separately
   make sure new code is wired into an existing call path.
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
- Before submitting, check every requirement on your list against your diff;
  if one is missing, implement it first. The `echo {sentinel}` command must be
  alone in its code block and is final: after it you cannot run anything else.
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


def render_observation(*, returncode: int, output_text: str, time_note: str = "") -> str:
    """``time_note``: the wall-clock pressure note for this turn, or "" (see
    ``agent_loop._time_pressure_note``). Pressure is denominated in real seconds
    at fixed budget fractions, not in projected commands-left: the mean-step
    projection under-warned every run whose late steps ran several times the
    early mean, so those runs died mid-edit having never been told to wrap up."""
    return OBSERVATION_TEMPLATE.format(
        returncode=returncode,
        output=output_text,
        remaining_note=time_note,
    )


# Escalating wall-clock notes, fired once each by the loop. "half"/"late" push
# the switch from exploring to implementing; "final" allows one last
# self-contained edit.
_TIME_NOTES = {
    "half": (
        "[Time check: about {left}s of wall clock remain -- half the budget is "
        "gone. You must now start implementing the tasks you have figured out so far.]"
    ),
    "late": (
        "[Time check: about {left}s of wall clock remain. Stop exploring and start implementing. "
        "Start implementing, starting with the most important ones]"
    ),
    "final": (
        "[Final window: about {left}s of wall clock remain. You have time for at "
        "most ONE more self-contained edit: do not modify code that already "
        "works, and do not start a change that needs a follow-up command. Implement the single most important remaining item which can be implemented in one command]"
    ),
}


def time_pressure_note(*, phase: str, time_left_s: float) -> str:
    return _TIME_NOTES[phase].format(left=max(0, int(time_left_s)))
