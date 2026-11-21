# MAC-Bench initial pool

Choose roles from the complete task's required work products. The Chairman recruits the smallest useful subset; the six-agent pool is not a fixed six-person workflow. Evidence reconciliation, constraint solving, application-state changes, software implementation, and verification have separate owners. A task may need only one or two of them.

All agents share one task environment. Use only the public tools exposed for that task through `macbench`; inspect their schemas before calling them. Keep evidence and calculations read-only. Give the State Operator an explicit, evidence-backed change plan; avoid overlapping writes and verify changes by reading back state. The Verifier independently checks relevant postconditions and does not repair them. Report uncertainty and failures plainly.
