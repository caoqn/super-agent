---
name: baseline-preserving-record-migration
description: Workflow for tasks that must replan and then re-implement an existing programme of records (residencies, shifts, bookings, invoices, calendars) across several business applications, where pre-existing records already encode intended assignments
trigger: When a task says "amend / reduce / replan an existing programme and implement it in the live applications", especially when supplied records already carry per-event assignments (staff, venue, resource) that the plan must respect
---

1. Inventory the BASELINE OPERATING RECORDS before designing anything.
   The pre-existing rows (draft/provisional staffing, provisional calendars, existing bookings) usually encode the intended assignment of people/resources to each event. Read them and write down the full mapping (event -> person, event -> venue, event -> resource) before choosing what to add, move or drop.
   Do NOT invent an alternative assignment map of your own unless a documented constraint (leave, unavailability, capacity, qualification) forces that specific change.

2. Prefer the MINIMAL change set.
   Keep the existing assignment for every retained item; touch only the items where a constraint forces a move/transfer/replacement. Record the reason for each forced change. Gratuitous reassignment is not neutral: derived obligations (minimum guarantees, penalties, notice fees) are attached to the ORIGINAL person and re-attribute themselves when you reshuffle, even if the total happens to be invariant.

3. Build a decision table (item -> keep/release, reason, notice date, alternative used) and a separate arithmetic sheet (per-person counts vs guaranteed counts, unit rates, totals, ceiling). Check invariants that must hold under any reading (e.g. total penalty) AND the per-person attribution.

4. Probe lifecycle constraints with ONE throwaway call before bulk writes.
   Determine early, on a single record, whether an already-submitted/committed document can be edited, and whether cancel-then-delete works. Repairing after a bulk write costs 3-4x the original write (cancel + recreate + re-submit) and can leave cancelled residue that pollutes "active" counts.

5. Write in dependency order, one owner per write set: policy-facing registers/decisions first, then source-of-truth operational records (calendars updated IN PLACE, never replaced), then derived obligations (staffing submissions, invoices/credit notes, unpaid penalty bills), then follow-up tasks as done. Never delete or rewrite original evidence; neutralise superseded financial documents with a credit note rather than deletion.

6. Re-read after each phase with a field-limited list call (id, key, status, docstatus) and count the rows. Confirm: exactly one active row per retained item, none for released items, unchanged totals, protected records untouched. Then hand the list to an independent verifier for a fresh read.
