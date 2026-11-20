---
name: stubbed-module-reimplementation
description: Workflow for reimplementing a library/module set whose function bodies were replaced by stub exceptions, where an existing test suite (public or shipped) defines the required behaviour
trigger: A source tree where function/method bodies are replaced by NotImplementedError/TODO/pass, and a shipped test suite must go green
---

# Stubbed-module reimplementation

Use when you must fill in many stubbed functions so that a shipped test suite (and later, artifact-level
grading) passes. The tests + downstream callers are the authoritative spec; the stubs' signatures,
defaults, docstrings and class attributes are contracts you must preserve.

## 1. Recon (fast, ≤5 tool calls)
1. Enumerate every stub with grep (`NotImplementedError|TODO|pass  # implement`), count per module, and list
   module line counts. **Test count per module = priority order.**
2. **Cheaply prove whether an unstubbed copy of the original source already exists** before reconstructing
   anything from memory: search for sibling/installed copies, source tarballs/archives, pip/wheel caches, and
   `__pycache__` artifacts. One `find`/`tar tf`/`grep -c` costs seconds; discovering late that a pristine copy
   was sitting in the image (or wasting hours reconstructing when it was not) costs the whole budget. Verify a
   `.pyc` is really from the *unstubbed* source by disassembling one stubbed method (marshalling the code
   object and inspecting `co_names`/`co_consts`) — a pyc compiled from stubs still contains the stub strings.
3. Read the test files that cover each module; note assertions per module.
4. Read fully-implemented sibling/consumer code (a frontend, CLI, subclass, or caller package) — it reveals
   attribute names and method signatures you must provide that no test mentions.
5. Read the local docs for documented keys/behaviours; skip unrelated doc chapters.

## 2. Build the spec table before writing code
For each stub, record: signature + defaults, return shape, exact exception type, and the **literal message
fragments** the tests assert. Tests usually match messages with `re.search`, so a message must contain the
expected words **in the same order** as the pattern. Watch for implicit key mappings that only tests reveal
(an input field name that must appear under a different output key), for "accept several types" helpers whose
error text quotes Python types verbatim, and for **empty/zero-input branches** (empty list/empty set/`only=()`
must take the documented branch instead of raising).

**Do not invent guard clauses.** When reconstructing from memory you will be tempted to add `isinstance`
checks or "must be a X" rejections that the original never had. Guards and error paths are exactly where
remembered behavior diverges; implement the permissive merge/accept path and let the tests tell you which
rejections are mandated. An invented guard typically manifests as a cluster of tests that fail *before* any
of the code under test runs (during construction/binding).

## 3. Ownership decision
- If the whole grade gates on one module (largest test count, imported by everything), **own that module
  yourself and write it first**; delegate peripheral modules.
- If you delegate the critical module, require an early intermediate artifact (file skeleton + first passing
  test count) within a fixed time box, then take it over if the artifact does not appear.
- One owner per file. Never let two writers touch the same file. For two tightly-coupled modules, split by
  file, not by class, and immediately exchange the **cross-module contracts** (who sets/reads which
  attribute, which side normalizes dotted/qualified names, which side binds children) so neither writer has to
  guess the other's behavior.

## 4. Writing the code
- Prefer **surgical body replacement** over rewriting a large file: a small script that regex-matches the stub
  line (`^(\s*)raise NotImplementedError(...)$`) and substitutes a pre-written body with
  `textwrap.indent(body, matched_indent)` preserves every docstring, decorator and declaration exactly.
  Keep that script (and a backup of the stub) so a bad patch is one command to revert.
- **Indentation is load-bearing**: `textwrap.dedent` strips the common prefix, so re-indenting by the matched
  prefix is mandatory — an off-by-four body still passes `ast.parse` while meaning something different. Always
  print/read the patched region back and eyeball its nesting before running tests.
- Write large files in **chunks of ≤200 lines per tool call** (append with a quoted heredoc); a single
  oversized write can be rejected and lose the whole attempt.
- Implement bottom-up: primitives (paths, hashing, permission/format helpers) → domain object → builders →
  public API hooks.
- Preserve stub docstrings, defaults (`mode=0o644`), return types (Path vs str vs basename), and legacy
  semantics such as "this helper returns the basename only".

## 5. Iterate per test file, not per whole suite
Run the module's own test file after each module, then the full suite. Triage each failure by reading the
assertion, not by re-reading the whole test file. **Attribute every failure to a file/owner before acting** on
it: read the bottom frame of the traceback, not the top one (a failure "in" a shared entry point is usually
raised inside the module under test). Expect 2-3 iteration rounds; most residual failures are
message-wording, ordering, or empty-input edge cases.

## 6. Verify at artifact level, not just unit level
Unit tests rarely cover real products. Write a throwaway script that builds/generates the real artifacts and
asserts the invariants: completeness of manifests/records (recompute hashes and sizes from bytes), permission
bits, timestamps/reproducibility, resource/data-file placement, and an end-to-end round trip
(build → unpack → rebuild → import).
When that script asserts behaviour whose exact representation you are unsure of, **probe first, assert
second**: print the real values for 2-3 primitives, then assert only observed values or
representation-independent invariants (round-trip equality, types, key sets). Guessed representations create
false failures that masquerade as implementation bugs.
If a stricter sibling test suite cannot run (missing deps/metadata), emulate its assertions through the
low-level API rather than skipping verification — and when a single test fails only due to missing packaging
metadata, prove it cheaply by placing a minimal `<dist>.dist-info/METADATA` on `PYTHONPATH` and re-running it,
then report it as environmental rather than papering over it with source edits.

## 7. Report honestly
State which checks you executed, which you emulated, and which stayed unverified (missing deps, blocked
agents). Never upgrade an unverified claim to "complete", and never let a "cannot pass here" failure be
silently dropped — classify it (source defect vs environment) with the command output that proves it.
