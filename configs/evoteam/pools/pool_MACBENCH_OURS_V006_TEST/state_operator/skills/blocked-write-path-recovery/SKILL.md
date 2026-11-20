---
name: blocked-write-path-recovery
description: Procedure for when your agent cannot execute mutation tools (bridge/validation failures) but the task still needs state changed — diagnose fast, prove zero writes, and deliver paste-ready payloads to the write owner.
trigger: A mutating tool call fails with a schema/validation/dispatch error, or a peer reports the same call succeeding while yours fails.
---

# When you own the writes but the write path is dead

## 1. Diagnose in at most 3 calls (never permute payloads)
1. Send the exact documented payload once.
2. Send an empty/minimal payload to the same tool.
3. Send the fully-conforming payload once more, only if steps 1-2 differed.

Decision rule: **identical error for an empty and a valid payload ⇒ the fault is the call path or this agent's capability, not your arguments.** Stop. Each additional variant is wasted budget; twelve variants teach you nothing that three did not.

Also record, in one line each, what DID work (other read-only tools in the same family), so the owner can see the failure is scoped.

## 2. Prove the state, in the same message
- Say explicitly "zero writes performed" and list what you attempted (all rejected pre-dispatch).
- Confirm no partial object exists, so nobody has to reconcile a half-done write later.
- Never retry a create with a new request id for the same logical object without reading provider state first.

## 3. Escalate with evidence, not with questions
Send the write owner: the verbatim payload, the verbatim error per shape, the working tools, and a clear request ("confirm you can execute; otherwise tell me which non-write role I should take"). Ask for ONE verbatim working call from them — a peer's copy-paste payload is the cheapest possible disambiguator.

## 4. Pivot to payload builder (highest value you can still add)
1. Read the machine-readable schema of record for every object you were meant to write (the source file/API card, not prose). Extract required keys, enums, nullability, and date/time formats verbatim.
2. Emit one compact single-line JSON argument object per object, ready to paste, with no commentary inside the JSON.
3. Recompute and state the invariants the payload must satisfy: row counts, per-group tallies, sums/totals, enum coverage, id sets. This is what catches a wrong total before it becomes a wrong record.
4. If the schema allows extra keys, offer an optional audit key (e.g. the assignment list behind a derived number) as a separate alternative line, so the owner can choose traceability without risking the required shape.

## 5. Defend the numbers
When a teammate reports an error in your figures, re-derive from the authoritative assignment table before accepting. Reply with per-row evidence (counts per group, sums) and state exactly what changed or why the report is conditioned on a different mapping. Accept corrections promptly and say so — and warn the owner if a wrong correction is heading their way.

## 6. Stay in your lane
If another agent takes over writes, do not touch those objects, even if you later find a working path; a duplicate write is worse than none. Offer read-only verification instead.
