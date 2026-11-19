---
name: spec-extraction-from-tests
description: Turn a stub-only implementation plus its test suite and fixtures into an exact behavioural contract table, flag conflicts, then verify the implementation without needing the project's test runner to be installed.
trigger: Assigned to extract or document the expected behaviour of unimplemented/stubbed functions from tests, docstrings and sample fixtures; or asked to verify an implementation against a test suite when pytest/the runner is unavailable.
---

# Spec extraction from tests

Use when the "spec" exists only as tests + docstrings + sample fixture directories, and
someone else will implement (or has implemented) from your report.

## Step 1 — Read the manifest first
Fetch the task description/README and any build/check script it names, in the SAME batch
as your first source listing. It tells you the sanctioned runner and the real scope, and
prevents you from guessing a test command later.

## Step 2 — Inventory the stubs
Grep for the placeholder pattern (`NotImplementedError`, `NotImplemented`, `pass  # TODO`,
`raise NotImplemented`). Count per file to get a live progress signal you can re-check
cheaply later.
Critical: in stub-only files the **preserved** parts are the spec — signatures, docstrings,
module-level constants, class attributes, exception hierarchies. Record them as authoritative;
recording only the missing bodies throws away free contract.

## Step 3 — Build the symbol table
For every symbol, collect two independent columns:
- **Consumers**: `grep -rn '\.attr\b'` across non-test source. This reveals undocumented
  attributes, return-shape conventions (basename vs full path, prefixed vs bare), and which
  args callers actually pass. A documented API with *no* consumer is low priority — say so
  rather than over-engineering.
- **Tests**: every `assert` touching the symbol.

## Step 4 — Harvest exact strings
Error text is usually pinned by regex, not by equality. Grep for `match=`, `pytest.raises`,
`assertRaisesRegex`, `caplog.text`, `str(exc_info.value)` and the harness's exception matcher.
Record the literal fragment that must appear (e.g. a `match=r'...'` substring), and note that
`match` is a *search*, so surrounding wording is free. Also capture log-message fragments
asserted via the log capture fixture.

## Step 5 — Turn each fixture directory into a case
Sample/fixture trees are executable spec cases. For each: list files, read the config, read
the code under test, and write down `input -> expected output` in one line. Fixtures are
often chosen to disambiguate branches the docstring hides (e.g. one sample that only parses,
another that must actually execute; one that must raise a *conflict* error because two
candidate locations both exist).

## Step 6 — Look for the traps
Naive implementations pass fixtures and fail tests when:
- the identifier in config differs from the display/public name (module name vs package name);
- a returned dict key differs from the parameter name (`for_fields=['description']` -> `'summary'`);
- a helper returns a bare digest while its caller adds the algorithm prefix;
- a property is exercised on `object.__new__(Cls)` with only one attribute set — so it must
  not touch anything `__init__` would have set;
- a return value is a **basename** while the caller joins it to a directory.
Call these out explicitly; they are the highest-value lines in the report.

## Step 7 — State conflicts with file:line
List contradictions between tests, docstrings and docs, each with a citation, and give a
recommendation. Typical: a constant duplicated with different values in two modules; a
docstring narrower than the behaviour the fixtures require; a doc that omits a field the
tests demand. Never silently pick one.

## Step 8 — Verify without the runner
If the test runner is missing, do NOT downgrade to static inspection. Re-execute the pinned
assertions yourself:
- run a short inline script against the real code paths, pointing at the real fixtures;
- compare exceptions and values on SEPARATE branches (catch -> compare type/substring; else
  compare value). Never use one sentinel tuple for both — mismatched sentinel shape produces
  fake failures;
- for parametrized tables, loop the table verbatim from the test file so the coverage is
  provably the test's, not yours;
- report `PASS n / FAIL n` plus only the failing rows.

## Step 9 — Report, then keep going
Send the dense partial (finding, source refs, conflicts, open gaps) as soon as the core
table exists. Then close your own flagged gaps and send an addendum — do not ship a report
titled "final" that still lists open items.

## Step 10 — Re-check the stub counter
After the implementer lands code, re-run the stub count and the verification sweep. This
produces a cheap, high-trust red/green signal for the whole team and identifies precisely
which remaining module is blocking which suite.
