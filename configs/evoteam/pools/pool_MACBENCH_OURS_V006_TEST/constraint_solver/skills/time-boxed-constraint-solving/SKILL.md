---
name: time-boxed-constraint-solving
description: A disciplined loop for turning task requirements plus evidence into a concrete, ordered action plan under a hard wall-clock budget, without over-exploring.
trigger: When you are asked to solve constraints, compare candidate solutions, or plan a multi-step application change under a task time/budget limit.
---

1. **Frame before probing (1 pass).** Restate the deliverable, the entities involved, and the pass/fail postcondition in at most a few lines. If the requirement is ambiguous, pick the literal reading and record it as an assumption instead of asking.
2. **Cap evidence gathering.** Define up front the minimum facts needed to choose between options (often 2-5 facts). Stop as soon as those facts exist. Do NOT exhaustively enumerate the environment.
3. **Sketch candidates cheaply.** List the 2-3 plausible plans in one message/note; discard the ones already incompatible with a known constraint. Do not deep-dive a candidate you will not recommend.
4. **Compute only the decisive quantities.** For numeric comparison, calculate the single number that separates the top options (total, net, balance, delta). Avoid calculating values that cannot change the pick.
5. **Decide with explicit assumptions.** Produce one recommended plan with: ordered steps, owner, inputs each step needs, and check to run after each step. Flag every unresolved constraint as a named risk.
6. **Self-check in-file, not by round-trip.** Embed the verification command/expected state into the plan so the executor does not need to message you back.
7. **Escalate early if blocked.** If a required fact is unavailable after ~2 attempts, report the blocker plus your best-guess plan in a single concise handoff rather than retrying.
8. **Budget tripwire.** Track a rough call/time budget; when half the budget is spent and no plan has been handed off, forcefully pivot from analysis to a written decision-with-risks.
