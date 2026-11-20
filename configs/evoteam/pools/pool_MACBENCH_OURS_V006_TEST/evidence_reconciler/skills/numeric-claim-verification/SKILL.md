---
name: numeric-claim-verification
description: How to check another party's computed numbers (totals, counts, fees, allocations) and report discrepancies without publishing false corrections.
trigger: When asked to verify, audit, or correct numbers produced by another agent/person/plan, or before telling someone their figure is wrong.
---

# Verifying someone else's numbers

Goal: be the reliable checker. A false "you have an error" costs more trust than a missed
discrepancy, because the recipient may act on it and break a correct plan.

## Procedure

1. **Restate their inputs, not yours.**
   Before comparing outputs, write down the exact input mapping the other party is working
   from: which items are in scope, which attribute of each item is used, and any
   re-assignment/selection they made. If you cannot infer it, ask for one line: "give me
   your item→category table". Comparing your mapping against their total is the single most
   common source of false corrections.

2. **Reproduce their number from their inputs.**
   Recompute mechanically. If you can reproduce it → it is right, stop.
   If you cannot → you do not yet have a finding; you have a hypothesis about a mapping
   mismatch. Say "this differs under mapping A; under mapping B it matches" and name both.

3. **State selection-dependent rules as conditionals.**
   "If X is selected/released/retained, then the deadline is D" — never "your plan violates D".
   A rule whose premise depends on a decision not yet fixed is not a defect in their plan.

4. **Recheck every multiplication and sum with a tool or twice by hand.**
   Confirm the factor list itself (how many items, at what unit price) before asserting a
   total. A single mis-multiplied factor invalidates an entire correction message.

5. **Separate the three kinds of finding.** Label each explicitly:
   - `CONFIRMED` — reproduced from their stated inputs, or read from the live source.
   - `CONDITIONAL` — true only if a stated premise holds; name the premise.
   - `ANOMALY` — two sources disagree and you cannot resolve which is authoritative; give
     both values, the evidence path for each, and which one you would defend and why.

6. **Retract fast, precisely, and without defensiveness.**
   On discovering a false claim: open with "I was wrong, withdraw it, do not change it", then
   one paragraph on the exact cause (which mapping/step), the corrected figure, and a short
   list of anything else that leaned on the same mistake. Do not bury it under new analysis.

7. **Check whether the fix changes anything downstream.** If the corrected figure feeds other
   totals, registers, or documents, list each dependent value with its new number so the
   recipient does not have to re-derive them.

## Anti-patterns

- Declaring an error because your recomputation differs — differences usually mean different
  inputs, not bad arithmetic.
- Publishing a headline "REAL ERROR" before you have reproduced their figure.
- Treating a derived aggregate field as authoritative when item-level rows are available; when
  a cached/summary field disagrees with the sum of the underlying documents, report the
  disagreement rather than "fixing" the cache.
- Silently dropping your own earlier wrong numbers from the record; name them in the retraction.
