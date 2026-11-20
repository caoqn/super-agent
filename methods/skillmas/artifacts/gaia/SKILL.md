---
name: unified_meta_agent_skill
description: "A foundational meta-agent skill for generating Multi-Agent Systems (MAS). It systematically drives the process from constraint-aware task decomposition to contract-driven agent engineering, and finally to resilient workflow orchestration with explicit commitment and recovery gates."
tags:
  - meta-agent
  - task-decomposition
  - agent-engineering
  - workflow-orchestration
inputs:
  - user_query
---

## 1. Task Decomposition Module (The "What")
*Core Objective: Analyze the user query and break it down into a logical blueprint.*

- **Intent & Scope Analysis:** Understand the macro objective, identify core requirements, and define the boundaries of the task.
- **Sub-task Breakdown:** Decompose the high-level request into a set of discrete, manageable, and logically cohesive sub-tasks.
- **Logical Dependency Mapping:** Identify the business-logic relationships between sub-tasks (e.g., prerequisite, parallel, or iterative). *Note: This focuses on logical order, not system dataflow.*
- **Constraint & Interpretation Ledger:** Before finalizing the plan, surface explicit and implicit constraints, qualifiers, exclusions, temporal scopes, and output-normalization rules. For each ambiguous term, record an interpretation commitment and its downstream planning consequence; distinguish critical-path obligations from optional enrichments; define the observable signals that would falsify an interpretation. This prevents a single upstream misreading from poisoning all downstream work.
- **Success Criteria:** Define clear objective outcomes and verification signals for each sub-task to ensure evaluability.

## 2. Agent Engineering Module (The "Who")
*Core Objective: Design specialized sub-agents tailored for the sub-tasks defined in Stage 1.*

- **Role Profiling:** Assign a unique identity and specialized role to each sub-agent based on its target sub-task.
- **Instruction Design:** Draft precise system prompts/instructions. Define the agent's specific goals, behavioral boundaries, and output expectations.
- **Agent Operating Contract:** Replace ad-hoc input framing with an explicit contract governing each agent's inputs, outputs, evidence standards, and escalation behavior. The contract must define: required context and source-of-truth hierarchy, with authoritative supplied artifacts taking precedence over inferred or retrieved substitutes; structured output expectations including evidence provenance, confidence, and unresolved assumptions; boundary conditions for stopping and escalating when required inputs are absent, evidence conflicts, or source access remains degraded after bounded retries; and a self-correction obligation requiring the agent to audit its output against the contract and flag contradictions before handoff.

## 3. Workflow & Orchestration Module (The "How")
*Core Objective: Wire the distinct agents from Stage 2 into a functional, executable Multi-Agent System (MAS).*

- **Architectural Topology:** Design the optimal MAS architecture (e.g., Sequential Pipeline, Router-based, Hierarchical, or Blackboard) based on Stage 1's logical dependencies. For complex but important sub-tasks, design localized sub-topologies using the instantiated agents, such as iterative critique loops for quality checks, parallel generation for diverse candidate outputs, or ensemble aggregation before downstream consumption.
- **Dataflow & State Management:** Define the exact I/O mapping. Specify how the output schema of one agent transforms into the input payload/prompt of downstream agents. Determine how global context and local state are maintained, what is preserved across handoffs, and what must be compressed or summarized to prevent context loss or overload.
- **Resilience & Commitment Layer:** Treat execution as a budgeted, fault-tolerant lifecycle with explicit commitment checkpoints.
  - **Budget-aware commitment:** Reserve a terminal synthesis window; at predefined consumption thresholds, stop exploration and force synthesis of the best-supported answer. A workflow must never terminate with an empty, placeholder, or unresolved sentinel; unresolved cases must still emit a declared best candidate with confidence and rationale in the pre-terminal region.
  - **Evidence-sufficiency gate:** Define in advance what constitutes decisive evidence. When that pattern appears, freeze further exploration, consolidate context, and hand off to synthesis.
  - **Source-health and fallback routing:** Maintain per-source and per-channel health; cap retries; mark degraded channels and route through a bounded fallback ladder to authoritative structured sources or alternate extraction paths; keep blocking waits below the system timeout.
  - **Terminal output contract:** Normalize the final deliverable to the required shape before emission; keep the terminal field atomic and minimal; validate against format constraints; place no post-answer prose after the terminal marker.
- **Executable Synthesis:** Produce the final orchestration specification that binds roles, contracts, topology, dataflow, and resilience gates into a runnable system design.