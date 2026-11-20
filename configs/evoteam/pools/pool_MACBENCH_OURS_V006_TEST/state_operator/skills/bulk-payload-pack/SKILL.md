---
name: bulk-payload-pack
description: Build a complete, handoff-ready set of many similar write payloads (dozens of creates/updates/cancels) from read-only source evidence, with validated shapes, resolved identifiers and recomputed invariants, when you cannot or should not execute the writes yourself.
trigger: When a task requires many similar application writes (bulk creates, bulk cancellations, roster/record population) and the payloads must be handed to an executor, or executed after careful design.
---

# Bulk payload pack

Goal: the executor can paste and run without re-deriving anything, and a reviewer can catch a wrong count before it becomes wrong records.

## 1. Read the schema of record, not prose
- Call the operation-card / schema projection for the exact operation; record required keys, nullability, min/max lengths, enum values, numeric formats, and the address path vs body split.
- Note which fields are *derived by the system* if omitted (e.g. an auto-issued identifier). Prefer omitting an optional code the native layer owns unless the task explicitly requires a specific code.

## 2. Resolve every identifier to its native form
- For each target object, take the native locator from the record directory/database, not the business label. Labels, slugs and human references are display-only.
- If a directory row shows `business_label -> native_id`, assume they differ everywhere and confirm per object.
- Resolve child/row identity by its stable id, then join the attributes you need (names, contact addresses) on that id — never on a name that repeats.
- Look at one existing object of the same kind to learn conventions the schema cannot express: the style of codes, what actually sits in a name/attendee field, what a zero amount looks like.

## 3. Recompute the invariants before emitting
- Row count = expected count; distinct keys = row count; no key missing from the expected range and none outside it.
- Per-group tallies sum to the total; per-group value stays inside any stated cap/limit.
- Value ranges: zero prices stay zero, no fees/refunds/payments invented, no derived totals that contradict the source.
Quote these numbers in the handoff — they are what catches a wrong plan before it becomes a wrong write.

## 4. Emit compactly, then a template
- One line per object: `key|target-native-locator|numeric-id|required-name-field|contact-email`.
- Then one copy-paste template with placeholders, and a rule for unique request ids (`create-<key>`), so the executor cannot accidentally replay one id across objects.
- Group lines by target group so the executor can stop after a clean band.

## 5. State status honestly
- Report executed/verified counts separately from planned counts; never describe an unexecuted payload as a change.
- Say explicitly whether any state was touched, and list what the executor must confirm live before writing (identifiers the schema could not confirm, item ids, caps).
- Offer read-only reconciliation: if the executor returns created ids, map them back to the plan lines and check exactly one active object per key.
