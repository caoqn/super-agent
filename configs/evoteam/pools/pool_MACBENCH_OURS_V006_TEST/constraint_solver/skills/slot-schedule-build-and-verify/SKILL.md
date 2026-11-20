---
name: slot-schedule-build-and-verify
description: Build a complete assignment of items to (date, time-slot, resource) under qualification, capacity, travel-buffer, absence and availability constraints — then prove it with mechanical checks instead of eyeballing.
trigger: Task asks for a full timetable/schedule/roster (every item gets exactly one slot) where resources have qualifications, some slots are blocked, and two things cannot occupy the same slot.
---

# Slot schedule build + verify

## 1. Turn the problem into four literal tables
Load these as rows (JSON literal rows in a query, or a small script) before assigning anything:
- `items`  — item_id, group (items sharing a group must share a placement), required_attributes.
- `slots`  — date, start_time, end_time.
- `resources` — resource_id, held_attributes, unavailable_windows.
- `hard_blocks` — resource_id, date, start, end (leave, protected duty, other booking).
- `eligibility` — item_id -> allowed dates / allowed slots (only for items that are restricted; default = all).
Materialize restricted-eligibility exceptions with ONE query (e.g. `WHERE accepted <> 'all'` + `GROUP BY value`), not by paging the whole collection.

## 2. Compute the capacity budget before assigning
- usable_slots = |dates| x |start_times| minus slots blocked for every candidate resource.
- Earliest-finish/earliest-slot ordering: place the most-constrained items first (fewest allowed dates, rarest required attribute, fewest qualified resources). Sort by (eligible_slot_count ASC, required_attribute_rarity DESC).
- Assert budget feasibility: for each required-attribute class, count qualified resources x available slots >= number of items needing it; for each centre/room, distinct slots >= families at that centre.

## 3. Assign greedily, with an explicit tie-break
Iterate items in the order above; for each, take the first slot in a fixed deterministic order (date ASC, time ASC, resource_id ASC) that satisfies ALL of:
1. slot ∈ item's allowed set;
2. the group's placement (centre/room) has that slot free — track a `used_placements` set keyed by (placement, date, time);
3. the resource has that slot free — track `used_resources` keyed by (resource, date, time);
4. resource holds every required attribute;
5. the resource's previous/next booking allows the travel/transition buffer (e.g. same time-slot at a different location is illegal; a 30-min slot then an immediate start elsewhere is illegal) — check the resources' bookings at the adjacent time index, not just the same time;
6. slot does not fall in a hard block or outside the resource's availability window.
Never break rule 7 of "no deferral": if the greedy pass leaves items unplaced, RE-PASS with a different resource ordering (rotate resource choice) before declaring failure — infeasibility claims need at least two independent passes.

## 4. Verify with set queries, not by reading the table
Run these as SQL/aggregates over the finished assignment and print the counts:
- `COUNT(DISTINCT item_id) = N` and no item_id appears twice; every required group appears once.
- `GROUP BY placement, date, time HAVING COUNT(*)>1` -> must be empty.
- `GROUP BY resource, date, time HAVING COUNT(*)>1` -> must be empty.
- Self-join consecutive slots per resource with a location change and gap < buffer -> must be empty.
- Join assignment to resource attributes and item requirements; any row where required attribute not held -> must be empty.
- Join assignment to hard blocks/availability on interval overlap -> must be empty.
- Print per-placement counts, per-date counts, and per-requirement-class counts; each must sum to N.

## 5. Report
Deliver one line per item in a fixed format, then the verification block (line count, the aggregate counts, and the explicit "empty result" checks with the query that produced them). If any rule cannot be met, name that rule, still deliver the best complete table, and never silently drop an item.
