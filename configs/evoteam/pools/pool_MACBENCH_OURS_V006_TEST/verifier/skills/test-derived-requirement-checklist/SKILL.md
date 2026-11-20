---
name: test-derived-requirement-checklist
description: Turn an existing test suite plus its production callers into a dense, literal requirement checklist that implementers can satisfy without re-reading the tests
trigger: When asked to produce requirements/specification for reimplementing a module whose tests already exist, or to hand off constraints to implementers
---

Goal: produce an artifact-first table an implementer can code against, with zero paraphrase of expected values.

1. **Inventory the ground truth.** List test files with collected-test counts (`--collect-only -q`, group by file). Note which files the official check actually runs and which suites exist but are not run — the latter still encode constraints and are the best proxy for hidden grading.
2. **Read the tests literally, not skimming.** For each test function record: inputs (sample fixture/project used, env vars set, cwd changes, monkeypatched values), each assertion with its exact literal (paths, member names, header fields, tags, counts, normalized versions), and return-type expectations implied by equality comparisons (`==` against a path object means return a path object, not a string).
3. **Derive behavior, not just assertions.** Translate each literal into the rule it pins down: e.g. `*` must not cross a separator while `**` must; a value `0.03` expected as `0.3` means normalization happens in the file name; a directory glob match must not crash.
4. **Enumerate the public-name surface.** Grep production callers and tests for every attribute/method/function accessed on the objects being implemented; produce a flat list with signatures and expected return shapes. Include inherited/overridden interactions (a caller subclassing the target with no overrides freezes the base signatures).
5. **Propagate reason codes.** Collect every exception type asserted, plus the message substrings/regexes matched, plus any place a caller formats the exception (`exc.args[0]`, `str(exc)`), since that dictates constructor arguments.
6. **Separate reproducibility rules** into their own table: timestamp env var handling (timezone, clamping), archive/format permission normalization, ordering determinism, and compression settings.
7. **Flag gaps explicitly.** State which required behaviors are not asserted anywhere (so verification must inspect artifacts directly), and which parts of the suite were out of your reading scope. Never present a gap as verified.
8. **Deliver tabular and dense.** One row per test; a name inventory; an exceptions table; a reproducibility table; a short uncertainty list. Quote literals verbatim, including whitespace inside expected strings.
