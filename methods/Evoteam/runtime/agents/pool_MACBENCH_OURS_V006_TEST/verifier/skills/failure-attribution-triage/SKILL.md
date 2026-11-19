---
name: failure-attribution-triage
description: Turn a batch of failing tests into an ownership-attributed defect table (which module/file owns each failure, and which failures are harness or environment artifacts) so parallel owners can fix in one pass
trigger: After a partial or full test run produces many failures spread across modules owned by different agents, or when a teammate claims an attribution for failures that you must confirm independently
---

# Failure attribution triage

Goal: every failing test gets exactly one owner and one cause category, proven by output — not by intuition about which file "feels" responsible.

## 1. Get a stable, complete failure inventory
- Run the suite with a machine-parseable reporter and save to a log outside the workspace: `... --tb=no -rf > /tmp/<dir>/full.log 2>&1`, then extract the `FAILED <nodeid>` lines. Keep the node ids (they encode the parametrization, which is usually the whole clue).
- Record the summary line verbatim; note pass/fail/skip counts per file by looping over test files one at a time (per-file counts localize the defect cluster faster than a raw diff of the full run).
- Run the batch a second time later: a team may still be editing, and a single run is only a snapshot.

## 2. Collapse failures into clusters by bottom frame
- Re-run with a one-line traceback (`--tb=line`), then normalize with `sed 's|^<repo>/||'` and `sort | uniq -c | sort -rn`. The counts immediately show how many failures share one code location and one exception.
- Cluster rule: same `file:line` + same exception message ⇒ one defect, one owner, regardless of how many parametrizations fail.

## 3. Attribute with the *raise site*, then sanity-check with the *call chain*
- The bottom frame is the raise/serialize site, not necessarily the owner. Classify:
  - raise site in module A and reached only via A's own logic ⇒ A owns it.
  - raise site in a shared/supplied helper (utils, validators, stdlib) but reached from A's call ⇒ A owns it; say so explicitly so the owner does not go hunting in the helper.
  - bottom frame in the test file itself (`Failed: DID NOT RAISE`, assertion diffs) ⇒ the owner is whatever production code should have raised/returned; identify it by reading the few lines above in the same traceback.
- Watch for wrapper/composition bugs: if a feature works for a direct object but fails when the same object is wrapped in a container field, the fix usually belongs in the wrapper's construction/normalization path (or in the normalization step that decides whether the wrapper qualifies) — check both sides before assigning.

## 4. Separately classify non-source failures
- Environment artifacts (missing dist metadata, missing optional tool, absent network/index, stub interpreter) are not defects. Prove them: reproduce the failure with a one-line interpreter command outside the test framework, and show where the failing attribute/metadata is read from a *supplied, unmodified* module.
- Confirm the file that reads it is byte-identical to the pristine baseline before calling it environmental.
- If you want to prove the environment theory, satisfy the missing prerequisite in a scratch location (e.g. a stub dist-info directory on the path outside the workspace) and show the test then passes; never install into the shared workspace.

## 5. Emit the table, then send early
- Table columns: `test node id | cluster | raise site (file:line) | exception/assertion | expected vs actual | owner | category (defect / harness / env)`.
- For any failure you attribute to a different owner than the teammate claimed, quote the traceback line that proves it, and send that row to the owning agent as soon as you have it — do not hold blocking evidence until the whole report is polished.
- Explicitly list what remains unverified (suites excluded, files not diffed, behaviours not asserted anywhere).

## 6. Before reporting a failure as a defect, rule out your own harness
- Any assertion you authored yourself must be traceable to production source or a public test. If a check fails, first grep the production code for the behaviour (e.g. does this serializer normalize the value at all?) and re-derive the expected value. Harnesses commonly encode *remembered* library behaviour that the actual code never promised.
- Only after the expectation is grounded do you report pass/fail.

## 7. Close the loop
- Re-run the full suite after fixes land and diff the failure list against your table: failures that disappear, new failures introduced, and clusters that split (one defect that revealed another) all belong in the final report.
