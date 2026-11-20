---
name: budget-gated-team-execution
description: Time-boxed dispatch, checkpoint, and forced-finalize protocol so a multi-agent run always produces a submitted result instead of exhausting the wall clock
trigger: Any task with a wall-clock or cost budget, especially long reconciliation/analysis tasks with multiple sub-deliverables
---
## Purpose
Prevent forced termination with zero output. The dominant failure mode is heavy upfront analysis + serial clarification round-trips, which consumes the budget before any verdict is written.

## Protocol
1. **Recon (<=5 calls, <=10% of budget).** Call the tool catalogue, then `describe` only the 1-3 tools you will actually invoke. Do not inspect tools unrelated to the deliverable.
2. **Dispatch once, dispatch complete (by ~15% of budget).** In a single batch, message every agent needed. Each message must contain: exact scope, the data/IDs to use, the output format you need back, the read/write boundary, and a self-check the agent can run without contacting you.
3. **Demand artifact-first reports.** Require each agent to write its numeric/structured result into the report itself (tables of computed values, not prose descriptions) so you never need a follow-up question.
4. **Checkpoint at 50%.** Collect partial results from everyone, even if incomplete. Note which sub-items are still open.
5. **Checkpoint at 70%.** Freeze all state changes. Run one verification pass on what has been written. Stop all further analysis.
6. **Finalize by 80%.** Submit the verified subset plus an explicit "unresolved" list. Never spend the reserve budget on a marginal sub-item when the main answer is unsubmitted.

## Anti-patterns
- Refining a plan or reading more schemas while teammates sit idle.
- One-message-per-question ping-pong instead of one complete specification.
- Attempting full correctness on every sub-deliverable when the total is graded and partial credit is possible.
- Continuing to debug after the freeze point.
