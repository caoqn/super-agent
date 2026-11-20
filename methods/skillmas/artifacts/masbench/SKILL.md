---
name: unified_meta_agent_skill
description: "A foundational meta-agent skill for generating Multi-Agent Systems (MAS). It systematically drives the process from conceptual task decomposition to agent engineering, and finally to workflow orchestration."
tags:
  - meta-agent
  - task-decomposition
  - agent-engineering
  - workflow-orchestration
inputs:
  - user_query
---

## 1. Task Decomposition Module (The "What")
*Core Objective: Analyze the user query and break it down into a logical blueprint whose requirements are explicitly resolvable and demonstrably covered.*

- **Intent & Scope Analysis:** Understand the macro objective, identify core requirements, and define the boundaries of the task.
- **Sub-task Breakdown:** Decompose the high-level request into a set of discrete, manageable, and logically cohesive sub-tasks.
- **Logical Dependency Mapping:** Identify the business-logic relationships between sub-tasks (e.g., prerequisite, parallel, or iterative). *Note: This focuses on logical order, not system dataflow.* **Calibrate effort to the score driver:** classify each sub-task as *decisive* (its outcome changes whether the deliverable is correct) or *optional/cosmetic* (it improves polish or exploration). Decisive sub-tasks own the budget; optional ones must never displace them.
- **Requirement Resolution & Evidence Matrix (Core Framework):** For every requirement in the query, allocate one row containing three coupled slots: (1) the *decision variable* to be determined, (2) the *convention or assumption* that disambiguates how it is to be interpreted, and (3) the *evidentiary source* that must attest to it. Governing rules:
  - **Resolve conventions once, globally.** An ambiguity that applies to the whole task (semantics of a label, unit of measure, inclusiveness of a boundary, horizon of validity) must be settled explicitly in the plan and propagated as a shared constant. It must never be re-decided independently inside each sub-task, and no sub-task may silently adopt the reading that makes its own local answer convenient.
  - **Assumptions carry their falsifier.** When the source material underdetermines a convention, record both the chosen assumption and the observation that would overturn it. An assumption without a falsifier is a guess wearing a plan's clothing.
  - **No unanswered rows at delivery.** Every row must terminate in either an attributed piece of evidence or an explicit, declared gap. A row that is merely "not looked at" is a planning defect, not a finding.
- **Success Criteria & Critical Path:** Define objective, checkable outcomes per sub-task, and mark which matrix rows are decisive. The decisive set forms the critical path: it must be fully evidenced before any answer is emitted, whereas optional rows may be reported as gaps.

## 2. Agent Engineering Module (The "Who")
*Core Objective: Design specialized sub-agents whose identities, authorities, and deliverable contracts are unambiguous to both the runtime and each other.*

- **The Agent Charter — Identity, Authority, Contract (Core Framework):** Every sub-agent receives a charter that keeps three things explicitly separate:
  - **Function vs. Identity.** *Function* is what the agent contributes and may be described in free, invented language. *Identity* is the handle by which the runtime and the environment recognize the agent, and it may **only** be drawn from — and exactly match — the system's registered roster of addressable participants. Any agent that must observe the world, act upon it, or exchange messages with peers must be chartered with a roster identity; a functional description must never be promoted into an identity. Agents whose work is purely aggregative, comparative, or supervisory are chartered as *identity-less coordinators*: they hold no environment-facing authority and can only delegate to actor-bound agents.
  - **Authority scoping.** Grant each agent the smallest set of capabilities its function requires, and make the distinction between "may act on the world" and "may only reason, request, and delegate" an explicit planning decision rather than an emergent accident. A role that lacks authority for an action must never be the designated path for that action.
  - **Boundary conditions / escalation.** An agent must stop and escalate to the orchestrator — rather than invent a workaround, silently degrade, or declare the task impossible — when it is denied an authority it believes it needs, when it cannot reach evidence that its matrix row demands, or when its instructions conflict with an observed payload. Escalation carries a diagnosis and a proposed repair, so a detected fault becomes a one-step correction instead of a terminal narrative.
- **Instruction Design (durable charter vs. ephemeral assignment):** Separate the stable charter (identity, authority, output form, register, escalation rules) from the per-round or per-sub-task assignment (the specific target, the specific question). Charters stay invariant across the whole run so behavior cannot drift; assignments are freely re-issued. Keep register requirements in the charter: terse, declarative claims — not hedged narration, not provenance annotations, not commentary about the agent's own confidence.
- **Input Context Framing:** Specify precisely what each agent needs (global objective, its matrix rows, upstream artifacts) and what it must not depend on, so no agent's work silently presupposes an artifact that no one has produced.
- **Deliverable Shape:** Require every agent to return a structured record naming its own identity, the claims it asserts, the evidence behind each claim, and an explicit *no-evidence* marker for anything it could not observe. Narrative prose is a byproduct, never the deliverable.

## 3. Workflow & Orchestration Module (The "How")
*Core Objective: Wire the distinct agents into a system that runs to a verified conclusion and hands over one canonical answer.*

- **Architectural Topology:** Choose the topology that matches Stage 1's logical dependencies — Sequential Pipeline, Router-based, Hierarchical, or Blackboard. Complex decisive sub-tasks may host a localized inner topology: an iterative quality loop for outputs that need repeated checking, or a redundant fan-out whose diverse outputs are all forwarded downstream. Topology is chosen for the *decisive* rows, not for the cosmetic ones.
- **Round-Driven Execution for Long-Horizon Tasks (Core Framework):** Whenever the task requires repeated interaction with a stateful environment, the topology must be a *sustained loop*, not a one-shot fan-out. Instantiate: a round counter; a persistent per-participant ledger recording last known state, holdings, and the latest returned status; and a fixed invocation order in which each actor performs exactly one world-affecting action per round. After each fan-out, read the environment's returned status, persist it to the ledger, then repeat. Three disciplines make the loop robust:
  - **Progress assertion.** If the objective's measurable quantity has not moved after a bounded number of rounds, the orchestrator must *change strategy* — not re-run an identical round. Repetition is not progress.
  - **Delivery bias.** Actions that convert accumulated holdings into scored outcomes take precedence over further exploration. An agent that is holding the means to deliver and continues gathering is malfunctioning.
  - **Watchdog with partial evidence.** Bound each round so that a stall leaves a populated ledger as a diagnosable partial trace, never an empty one.
- **Termination Authority & Canonical Emission (Core Framework):** The workflow ends only for two legitimate reasons: a termination signal owned by the environment/verifier, or complete satisfaction of the critical path. Self-narrated completion, a round count the orchestrator reports about itself, and a refusal caused by an infrastructure failure are all *invalid* terminations and must be blocked. Before hand-over, run an emission gate:
  - **Coverage:** every decisive matrix row holds evidence or a declared, justified gap; no upstream participant's contribution is missing, empty, or truncated.
  - **Agreement:** the last observed environment/verifier status actually indicates the task is over. A stated "done" that no payload corroborates is a fabrication.
  - **Container:** exactly one canonical, machine-parseable payload in the precise form the evaluator consumes — without decorative wrappers, without a preamble, without the correct content buried inside a narrative.
  - **Register:** terse and declarative. No hedging blocks, no provenance tags, no confidence commentary, no alternative answers offered "just in case." Supporting numeric values that were derived must appear in the answer, not remain upstream.
- **Dataflow & State Management:** Define the exact I/O mapping — how one agent's output becomes another's input payload, and how global state is maintained and compressed so hand-offs do not overflow or lose context. Every evidence path must have a degradation path: if a peer-to-peer transport yields nothing, the orchestrator must be able to obtain the same evidence by requesting each participant's verbatim private view and merging centrally. No result may depend on a single channel succeeding.
- **Executable Generation:** Output the final orchestration logic that binds agents, tools, and dataflow into a ready-to-run system.