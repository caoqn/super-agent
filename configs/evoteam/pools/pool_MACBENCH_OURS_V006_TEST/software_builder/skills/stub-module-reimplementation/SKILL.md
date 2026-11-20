---
name: stub-module-reimplementation
description: Workflow for reimplementing a module whose function bodies were replaced by NotImplementedError stubs, where a test file in the repo is the authoritative spec
trigger: When asked to fill in stubbed/NotImplementedError function bodies so that existing public tests pass
---

# Reimplementing a stubbed module

## 0. Generate the whole first pass in ONE mechanical edit
Do not hand-edit dozens of bodies. Instead:
1. `grep -n 'NotImplementedError' <file>` to enumerate every stub, and note the stub marker text
   (often the symbol name is embedded in the message — that makes the markers unique keys).
2. Read the file in a few large chunks (declarations, signatures, class attributes, docstrings,
   preserved `default_error_messages` / module constants / imports are all hints).
3. Write a throwaway generator script OUTSIDE the repo (`/tmp/...`) holding a table
   `marker -> replacement code block`, then replace markers sequentially (keep an occurrence
   counter for markers that legitimately repeat, e.g. a property getter+setter pair).
4. Run it, then `python -m compileall`/build and the focused test file. A runnable draft with
   failures beats a perfect plan.

## 1. Mine literal expectations BEFORE writing (not after failing)
The tests, not your memory of the upstream library, are the spec. Before coding a group of
related symbols, grep the test suite mechanically for the assertions that pin behavior:
- error/exception assertions: `pytest.raises(...)`, `match=`, `excinfo.value.args[0] ==`,
  `str(exc.value...) ==`, `assert "..." in`
- value assertions: `@pytest.mark.parametrize` tables feeding the symbol under test
- type assertions: `isinstance(...)` / `is ...` / `type(...) is`
Write the results as `input -> required output` / `raise -> required message fragment` rows.
This is the single biggest time saver: memory of a library's source silently drifts on edge
cases (NaN signs, int-vs-str coercion, empty-vs-None, hashing, Mapping vs iterable checks),
and each drift costs a full test cycle to discover.

## 2. Read the consumers, not the whole repo
Grep sibling modules and callers for attribute/return-value use to pin data shapes
(dict keys, str vs Path, list ordering, who sets `parent`/`root`/`name`). Ask the coordinating
agent to confirm any shared interface you cannot pin down, and state the assumption in your
first report.

## 3. Classify every failure into (a)/(b)/(c) before fixing
(a) my bug, (b) dependency still stubbed/buggy in a file another owner holds, (c) environment
(no pytest, package not installed). Only (a) is yours to edit.

**Attribution proof for (b) without touching the peer's file:** copy the failing behavior's
expected upstream semantics into a scratch pytest plugin outside the repo and load it with
`-p <module>` (e.g. monkeypatch one method on the class at import time). Re-run the same tests:
if they go green, you have proven the defect is in the other file, and you can hand the owner
the exact replacement snippet plus before/after pass counts. Never edit a file you were told
is off-limits "just to check", and never weaken a test.

## 4. Iterate on the focused test command
Run the smallest relevant test selection after each batch of fixes; widen only when it is green.
Always report the real summary line (`N passed, M failed`), never an inferred one.

## 5. Record deliberate deviations
Deviations from the reference implementation that are required to satisfy an assertion
(e.g. normalizing a value the reference would preserve) must be listed explicitly at the end,
together with the test that forces them. That is what lets an independent verifier judge
intent instead of guessing.

## 6. Report early and briefly
Send the requester an interim message (drafted/passing/blocked/ETA) well before the deadline.
Include: remaining stub count, per-file pass/fail counts, the exact commands used, and any
assumption you could not verify. An owner who hears nothing assumes failure.

## 7. Scope reassigned mid-task
Execute the new smallest deliverable first (scratch file + self-check + pasted output), report
it, then continue.
