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
*Core Objective: Analyze the user query and break it down into a logical blueprint.*

- **Intent & Scope Analysis:** Understand the macro objective, identify core requirements, and define the boundaries of the task.
- **Sub-task Breakdown:** Decompose the high-level request into a set of discrete, manageable, and logically cohesive sub-tasks.
- **Logical Dependency Mapping:** Identify the business-logic relationships between sub-tasks (e.g., prerequisite, parallel, or iterative). Distinguish critical-path work from optional enhancements. *Note: This focuses on logical order, not system dataflow.*
- **Verification-Contract Matrix:** For each sub-task, define the observable acceptance evidence, the verification method that will establish it, and its criticality. Treat conjunctive success conditions as separately verifiable commitments rather than a single aggregate goal, so partial progress remains visible, attributable, and repairable.

## 2. Agent Engineering Module (The "Who")
*Core Objective: Design specialized sub-agents tailored for the sub-tasks defined in Stage 1.*

- **Role Profiling:** Assign a unique identity and specialized role to each sub-agent based on its target sub-task.
- **Agent Contract & Boundary Design:** Draft precise instructions that separate stable role and behavioral policy from task-specific objectives. Define the agent's goal, output expectations, decision authority, required self-verification evidence, escalation triggers, and safe stop conditions; an agent must know when it has enough evidence to declare completion and when it must request clarification or hand off a bounded unresolved issue instead of continuing an open-ended loop.
- **Input Context Framing:** Specify what contextual information this agent requires from the user or the global task to begin its work.

## 3. Workflow & Orchestration Module (The "How")
*Core Objective: Wire the distinct agents from Stage 2 into a functional, executable Multi-Agent System (MAS).*

- **Architectural Topology:** Design the optimal MAS architecture (e.g., Sequential Pipeline, Router-based, Hierarchical, or Blackboard) based on Stage 1's logical dependencies. For complex but important sub-tasks, instantiate localized topology patterns using the available agents—for example, iterative quality loops, or repeated sampling of one agent's outputs followed by synthesis for downstream agents.
- **Verification-Gated Dataflow & Budget Governance:** Define the exact I/O mapping, context transformations, and durable shared state, including the separation between global memory and agent-local context. Establish explicit resource-budget checkpoints that force a transition from exploration to consolidation, reserve capacity for end-to-end acceptance verification, and require every handoff or finalization to carry the evidence defined in Stage 1. The workflow must distinguish "still exploring", "verifiably complete", and "bounded partial completion", and must never terminate solely because resources were exhausted without a deliberate submission decision.
- **Executable Generation:** Output the final orchestration specification that binds the agents, tools, dataflow, and verification gates into a ready-to-run system.