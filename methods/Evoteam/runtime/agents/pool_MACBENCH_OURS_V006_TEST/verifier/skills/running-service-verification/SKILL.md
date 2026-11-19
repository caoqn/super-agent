---
name: running-service-verification
description: Independent, read-only verification of a feature on a live/running service (HTTP API + served assets) against a written requirement checklist, including freshness/state audits, fixture-capability triage, and baseline restoration
trigger: When asked to verify a running instance (not just a test suite) against a feature spec, or to confirm another agent's claim about a deployed build
---

# Verifying a running service against a spec

Use when the deliverable is a live app/server (API + front end) and your job is read-only
verification. Goal: PASS/FAIL per requirement with cited evidence, and zero leftover state.

## 0. Establish the target envelope BEFORE any behavioral claim
- Record the server process identity and start time, then compare with source file mtimes
  (`ls -l --time-style=+%H:%M:%S` on the files that implement the feature). If the process
  predates the newest edit, its behaviour describes stale code — either wait for a restart or
  label every result "as of stale process <pid>, pre-edit".
- Record the served asset identity (URL, byte size, hash). Re-check it at the end; a changed
  hash means the target moved during verification and earlier results need re-running.
- Read the fixture/original-state definition (fixture JSON, seed scripts, docs) so you know
  what "pristine" looks like and which records are pre-existing vs. your own or a peer's.
- Snapshot baseline counts for every collection you will touch (list endpoints give you
  `count` + ids), including nested/child collections.

## 1. Build a throwaway harness outside the workspace
- One small module with: auth helper (token/session), generic `request(method, path, body)`
  returning `(status, parsed_body)`, file upload helper, async-task poller, and a state
  save/load JSON in your scratch dir.
- Keep the harness idempotent and re-runnable; every probe prints `status` + a short body
  excerpt so the log itself is the evidence.
- Never write into the shared workspace: no caches, no bytecode, no temp files there.

## 2. Execute the checklist literally
- Convert each requirement sentence into an assertion with a literal expected value taken from
  the spec, the tests, or production source (exact strings, status codes, field values).
- Run the cheap public commands (test suite, smoke script) twice and record both.
- Run the idempotency/migration item (e.g. apply migrations twice) and re-read state after.
- For each write-endpoint check: create → read back → exercise the negative path (400/403) →
  read back again to prove the stored definition is unchanged.
- For access control, test every subject the spec names, plus one unrelated subject.

## 3. Triage each failure before reporting it
Classify, in this order:
1. **Stale build / stale process** — does the code path exist in source but not in the running
   process? Ask for/await a restart, re-run.
2. **Fixture limitation** — does the subject account hold the permissions/roles the assertion
   needs? Are the referenced catalog objects present? If not, the assertion is unsatisfiable by
   any implementation: prove the mechanism with an equivalent capable subject and report the
   limitation separately from the defect.
3. **Harness bug** — re-derive the expectation from source; if your expectation was wrong, fix
   the probe and say so (do not report it as a defect).
4. **Semantics mismatch** — behaviour is a defensible reading of the spec but not the literal
   one (e.g. exact vs substring matching, owner precedence at a pipeline stage). Report as
   "fails as literally specified, passes under interpretation X" with both results.
5. **Real defect** — everything else.

Report mechanism, not just symptom: cite the code path (file:line) and the observable
difference. Send blocking evidence to the owning agent immediately as a short partial report.

## 4. Restore baseline and prove it
- Name every probe record with a unique prefix; delete by prefix in the same script that
  created it, then re-read counts of every collection (parents AND children) to show baseline.
- If a delete does not cascade, delete children explicitly and note the finding.
- Report anything you could not delete or that another agent created (with provenance evidence
  such as creation timestamps/task ids) instead of deleting someone else's records.

## 5. Final report shape
- One PASS/FAIL/PARTIAL line per requirement, each with the concrete status codes and JSON
  excerpts that justify it.
- Separate sections for: items verified on the current build, items unverifiable and why,
  defects with file:line attribution, and the restored final state counts.
- State the target identity (process start, build hash) the report applies to, so a later
  reader knows whether it is still valid.
