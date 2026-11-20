---
name: bulk-business-app-write-triage
description: Plan and execute tasks where a source dataset must be turned into many live application records (one per entity) across several business apps, including deciding early how much of the write volume is affordable
trigger: A task asks you to reconcile source evidence into live application state and the deliverable is one record/arrangement per entity (per child, per family, per order, per shift, per invoice) rather than a single edit
---

# Bulk business-app write triage

## 1. Fix the invocation path once (max 3 attempts)
- Discovery/documentation tools are usually called directly; business operations usually go through a dispatch/bridge operation. If the catalogue returns a name like `app.entity.operation` but calling it at top level gives "unknown tool", try the documented dispatch wrapper once: `{tool_name:"call_tool", arguments:{tool_name:"<op>", arguments:{...}, request_id:"<unique>"}}`. Also try `arguments_ref:{result_ref, path}` for large argument objects.
- Do NOT permute more than ~2 shapes. Read the environment's worked-example/tutorial document instead; it states the wrapper verbatim.
- Verify the path with a READ on a real record before any write. A validation error coming back *from the target operation* proves the path works.

## 2. Derive every decision in SQL, not by eye
- Load the source-linked fact tables/collections as compute sources; join entity, evidence, consent, capacity and roster tables in one CTE.
- Never infer per-item attributes from visual patterns (even/odd ids, position in a list). Join the actual column.
- Validate with aggregate checks before writing: counts per status, per programme/capacity bucket, accessible/limited-slot usage, sibling/group coherence, and that every entity appears exactly once.

## 3. Emit register/document content directly from SQL
- Build the whole argument object in one query: `json_object('content', json_object('<rows_key>', json_group_array(json_object(...))))` with `output="json_value"`, order rows in a subquery, and use `json(col)` when a nested array would otherwise be emitted as a quoted string.
- Submit with `arguments_ref:{result_ref, path:"/data/value"}` so a 100+ row record costs one call.
- Accept that the compute receipt echoes the payload once; budget for it.

## 4. Triage the native write volume BEFORE analysis
- Count units: entities placeable, holds to cancel, appointments to create, duties to submit. Multiply by calls-per-unit (often 2: calendar + staffing duty).
- Probe ONE write and record its true cost (each successful write typically echoes the full object — often ~1k tokens). Multiply.
- If the product exceeds budget, decide explicitly and early:
  - Prefer the register/document layer (cheap, one call each) and complete ALL of it.
  - Then spend the remainder on coherent slices of native state — slices that leave each entity in a *self-consistent* condition (e.g. cancel-old-then-create-new per entity, or only entities that currently have no conflicting record).
  - Never leave an entity with a wrong old record plus a new correct one (duplicate), and never strip an entity's only record while its replacement is unwritten.
- Start the mechanical write stream as early as possible; it is the part that cannot be compressed, and it must not be left behind while cheap analysis continues.

## 5. Copy conventions from one existing record
Before creating anything in a live app, retrieve one existing record of the same type and mirror it: identifier charset (hyphens can be rejected even when the length is valid), which field carries the stable business key (often the entity id, not the display name), amount/price representation, status values, and whether the platform issues its own codes. One read is cheaper than a 400.

## 6. Honest closure
Report per-artefact: what was written (with the native ids returned), what was verified by read-back, and what is unresolved with the exact state left behind. A register row must never contain an invented native id (event UUID, order code, duty name) — omit the register and mark it unresolved instead.
