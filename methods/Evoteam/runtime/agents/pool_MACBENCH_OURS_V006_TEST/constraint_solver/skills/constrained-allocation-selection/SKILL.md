---
name: constrained-allocation-selection
description: Choose which commitments of a larger original set to keep, release, or replace under a quota, per-unit prices, minimum-guarantee penalties, and scarce shared resources — then prove the choice is near-optimal and internally consistent.
trigger: When a task requires selecting a fixed number of items from a larger committed set (retain/release/replace/downgrade) while satisfying quotas, penalties for unfilled guarantees, per-item prices, and shared-resource calendars.
---

# Constrained allocation selection

Use this when a corpus of commitments must be cut down to a quota and every cut has contractual consequences. Worked pattern: original set -> exactly N kept, rest released with reasons and lead-time notices, penalties for unmet minimums, shared equipment/rooms that cannot be in two places at once.

## 1. Build the commitment table first (one pass)
One row per commitment: id, category/pool, originally assigned party, assigned resource, original dates, capacity quantity (participants/headcount/volume), monetary unit price. Read the authoritative source text once end-to-end (chunked reads beat crawling many small tables) and write the table down.

## 2. Separate the three decision classes before optimising
- **Hard-infeasible**: below a stated minimum, counterparty withdrew, resource unavailable on those dates. These can only be released or moved onto a *signed* alternative. Never "fix" a shortfall by editing the original report.
- **Movable**: has an explicit signed offer/option (new date, new venue). Accepting uses the offered value verbatim; declining preserves the original.
- **Clean**: keep candidates.
This step usually shrinks the search space to a handful of choices.

## 3. Model cost correctly (the usual source of error)
Total = sum(unit price of each kept item) + sum(per-party penalties) + fixed incurred costs + scarce-resource logistics.
- Penalty = max(0, guarantee − actual kept assignments) x penalty rate, computed **per originally contracted party**, not per item.
- Reassignment inside a pool redistributes guarantee satisfaction but does **not** change the unit price sum. So separate the two problems: (a) which items to keep, (b) how to assign them to satisfy guarantees.
- A release still counts against the original party's guarantee: substitution never erases the obligation.
- Fixed/irrecoverable costs stay in the total even when their item is released.

## 4. Optimise with a table, never in your head
Enumerate candidate selections in a small SQL/script table (kept count per pool x fee + recomputed penalties) rather than mental arithmetic. Two candidate sets within ~1% of each other are a tie: pick the one that is simpler to explain, not the marginally cheaper one.
Establish a **theoretical floor** as a stop signal: if the sum of guarantees exceeds the quota by k units, at least k penalty units are unavoidable, so the floor = simplest mix of cheapest penalty units. Once your answer is at/near the floor, stop searching — state the bound in the handover so nobody re-litigates the choice.

## 5. Evaluate optional/incremental additions with net deltas
For each candidate addition compute delta = added price + added logistics − penalties it removes (− any guarantee unit it satisfies). Include **decreasing** lines, not just new spending. Report each variant's total and remaining headroom against the ceiling; recommend zero additions when exercising them is permitted but not required and income does not offset a gross cost ceiling.

## 6. Consistency checks before handing off
1. Counts reconcile: kept + released + replaced = original total, each id exactly once.
2. Every retained item appears in every downstream artifact: staffing rows, resource/trip rows, calendar rows, invoices. Cross-check by set equality, not by spot-looking.
3. Per resource, list its dates and assert uniqueness; per party, assert no overlapping dates and no exceeded availability/absence/maximum.
4. Every release has a substantive reason and a notice date that satisfies both the lead-time rule (>= X days before the *original* date) and the response deadline.
5. Totals re-add exactly from the stored line items, and the ceiling check shows explicit headroom.
6. Recompute any number that depends on mutable application state from a fresh read, and say which read it came from.

## 7. Report shape that survives review
Give the plan as: ordered action list with owner per step, the decisive number and its arithmetic, the floor bound that justifies the choice, and named risks/interpretations. Make each derived number auditable from an explicit mapping (id -> assignment -> count), because a single reassignment can silently change every penalty figure.
