---
name: end-to-end-api-feature-verification
description: Checklist for proving a newly implemented HTTP/REST feature actually works through the application's own build, service start-up and public entry points rather than only through unit-level calls
trigger: After implementing a feature that ships behind an HTTP API (with or without a UI), before declaring the task complete
---

# End-to-end API feature verification

Unit tests passing on direct function calls prove very little about a feature whose
acceptance criteria mention endpoints, status codes, permissions or UI controls.
Run this checklist against a **running instance** and keep a script so each run is
reproducible.

## 0. Build with the project's own command
- Run the exact build command the project documents / the grader will run (not your
  hand-assembled equivalent). Note its exit code; also note the *deploy* step
  (collect / copy static assets) and any documented caching gotcha — stale
  pre-compressed assets served with a mismatched `Content-Length` silently break a
  single-page frontend while every API call still works.
- Re-run the project's public check suite with the same interpreter/env as the build.

## 1. Authenticate exactly as a client would
- Obtain a token/session through the real login endpoint; never bypass auth by calling
  the ORM. Note the header format the project expects.

## 2. Exercise the documented surface, in the documented order
For each resource: list (verify pagination envelope and ordering param), create
(expect 201), read (200), update by full PUT **and** by PATCH of a single nested key,
delete (204), then list again. Verify:
- **Round-trip self-consistency**: send the exact retrieved representation back on PUT
  — it must be accepted. Any read representation the write path rejects (empty lists,
  null-vs-absent, normalised scalars) breaks whole-object clients and is invisible to
  create-only tests.
- **Nested row identity**: after an update, confirm counts did not duplicate and note
  whether nested ids are recreated; if they are, clients must re-read after save.
- **Cascade / orphan hygiene**: delete the parent, then confirm children are gone and
  no orphans accumulate in the child list endpoints (count back to baseline).

## 3. Negative paths — validation must precede writes
- Invalid enum value, nonexistent foreign key, constraint the task names explicitly
  (e.g. "requires at least one filter") → 400 with a useful message.
- After each rejected update, re-read the stored record and assert it is **unchanged**.
- Permission denial: authenticate as a user without the new permission → 403 for list,
  detail and write; as a granted non-superuser → 200. Superuser → 200.

## 4. Prove the feature has real effects, not just records
Drive the real pipeline end to end (upload through the real ingestion entry point,
trigger the real lifecycle event), then read the resulting business object and assert
every claimed effect — including negative cases (non-matching input must be untouched)
and disabled/edge configurations (disabled feature must not act).

## 5. If there is a UI
- Assert the required visible/accessible controls are present in the **served** bundle
  (fetch the JS asset and grep for the exact English strings, including aria-label text
  and button names); confirm `Content-Length` matches on-disk size for plain and gzip
  responses.
- Drive the browser when possible: create through the form, then re-read the definition
  via the API to prove the exact submitted values persisted, and prove a document
  processed afterwards picked up the change. Verify permission gating hides the nav /
  redirects an unprivileged user.
- Prefer real `<button>`/labelled controls for anything the requirements describe as a
  button or label; option text inside a `<select>` is discoverable but not button-like.

## 6. Leave the world clean and report state
- Delete every probe record you created (and verify the delete cascades); re-read counts
  for each resource and state them in the report so a stale leftover is obvious.
- Record the exact commands run, exit codes, and any unresolved item with its rationale,
  distinguishing environment artefacts from source defects.
