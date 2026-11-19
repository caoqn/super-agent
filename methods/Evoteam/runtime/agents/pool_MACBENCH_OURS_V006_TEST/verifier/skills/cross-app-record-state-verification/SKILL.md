---
name: cross-app-record-state-verification
description: Independently verify a multi-application business-state change (CRM records, calendar, HR documents, accounting ledgers) against a written policy, using baseline snapshots, live re-reads, independent recomputation and version fingerprints
trigger: When asked to verify that another agent's writes across several connected business applications satisfy a policy/requirement set, or when signing off a state-changing task whose outputs are records rather than code
---

# Cross-app record-state verification

## 0. Before the writes land (do this first, it is the cheapest hour you will spend)
1. Read the policy/spec sources and turn them into a checklist with EXACT expected values: counts, per-item money amounts, date arithmetic windows, and explicit "protected set" lists (records that must not change).
2. Derive the numeric model yourself from source (rates, fractions, per-item fees, minimums). Write the formula, not just the answer, e.g. `fee = rate * shortfall_units`, `total = unit_rate * trips`, `net = gross - credits`.
3. Snapshot a baseline inventory with immutable ids for every collection you will later verify: task/record ids, event ids, document names, invoice/bill/payment ids and their amounts. Store the raw read receipt so you can diff later.
4. For each artifact you read, record its version/fingerprint field (content_version, etag, modified timestamp). Later re-reads then PROVE stability instead of assuming it.
5. Pre-compute the risky edge cases and state them to the writer before the write: tight date windows (a notice date that is only just inside the 14-day cap for the earliest item), id-space constraints, single-resource collisions (one shared asset booked twice), and hard caps per owner.

## 1. Verify from live reads only
- Never accept the writer's summary as evidence. Re-read every collection yourself.
- Read the key collections TWICE, a couple of minutes apart, and compare. Identical reads = stable tree. Differing reads = someone is still editing; report the delta, not a verdict.
- Re-read the registers/notes and compare their version fingerprints with your earlier read; a changed version means the artifact moved after your first pass.

## 2. Recompute, don't compare
- Recompute every derived number from the live primitive records (assignments → per-owner counts → shortfall units → fees → ledger bills) and compare to the stored register value. Two independent paths agreeing is the actual evidence.
- Tie totals to the ledger from BOTH directions: sum of line/document amounts and sum of outstanding amounts. State the basis you used (whole-account vs period-only) and check the register used the same basis.
- Check that credits/refunds/voids are represented as separate applied documents, never as silent edits to the original amount.

## 3. Completeness and exclusivity
- For each item in the final set: exactly one artifact exists (one invoice, one assignment, one calendar entry) — no duplicates, no orphans, and none for excluded items.
- For each excluded item: the artifact is still present in a non-active/terminated state (cancelled/credited/void), not deleted.
- Every retained item's cross-referenced attributes (dates, location, asset, owner) must be identical across ALL artifacts: calendar, staffing, logistics, and registers.

## 4. Protected-set diff
- Diff the protected collections against the baseline: same ids, same amounts, same statuses, same document names. Report counts by title/type (e.g. N pre-existing + M new = total) — count arithmetic alone proves nothing was silently deleted.

## 5. Negative probes (state them explicitly as "all negative")
Duplicate per item, artifact for an excluded item, charge/liability with no matching shortfall, a completion/terminal status with no documented reason, missing required response records, arithmetic drift, more than one artifact per required slot, stale attribute left on a retained item.

## 6. Reporting
- One row per checklist item: PASS / FAIL / unverified, with the concrete id and value you observed.
- Separate real defects from application-side staleness (cached/denormalised aggregate fields that a write path does not refresh). Name the authoritative basis and warn against "fixing" a cached aggregate with a write.
- Flag interpretive ambiguities (e.g. whether a due date should be moved to the decision deadline) as notes, not failures.
- Close with what you could not verify, and why.
