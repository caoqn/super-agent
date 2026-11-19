"""Runner — minimal task execution engine that manages Agent lifecycles and reflection."""

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from core.agent import Agent, AgentResult
from core.api_resilience import APIReliabilityTracker
from core.message_store import MessageStore
from core.output_contract import OutputContract
from core.reflection_runner import ReflectionMixin
from core.submission_bridge import SubmissionBridge
from core.tool_registry import ToolRegistry
from core.utils import load_prompt_template as _load_prompt_template

logger = logging.getLogger(__name__)


@dataclass
class RunnerResult:
    """Result of a Runner execution."""
    output: str
    agents: dict[str, Agent] = field(default_factory=dict)
    cost_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class Runner(ReflectionMixin):
    """Minimal task execution engine. Starts Chairman, waits for terminate or timeout, collects results."""

    REFLECTION_MAX_STEPS = 50
    REFLECTION_PHASE_TIMEOUT = 300.0
    _CHAIRMAN_EXIT_GRACE_SECONDS = 2.0
    _CHAIRMAN_SUBMISSION_WARNING_FRACTION = 0.10
    _CHAIRMAN_SUBMISSION_WARNING_MAX_SECONDS = 120.0
    GLOBAL_ANSWER_AGENT_NAME = "answer_agent"

    def __init__(
        self,
        pool_dir: str | Path,
        chairman_name: str,
        constitution: str = "",
        tool_registry: ToolRegistry | None = None,
        max_seconds: float = 600.0,
        max_messages: int = 200,
        idle_timeout: float = 120.0,
        enable_reflection: bool = False,
        max_cost_usd: float = 0.0,
        reflection_phase_timeout: float = 0.0,
        wall_clock_seconds: float | None = None,
        api_tracker: APIReliabilityTracker | None = None,
        budget_started_at: float | None = None,
        chairman_forced_submission_seconds: float | None = None,
        allowed_agent_ids: set[str] | None = None,
        team_selection_metadata: dict | None = None,
        output_contract: OutputContract | None = None,
        execution_policy: Any | None = None,
        teammate_profiles_enabled: bool = False,
        chairman_teammate_profiles_enabled: bool | None = None,
        member_teammate_profiles_enabled: bool | None = None,
        enforce_handoff_selection_protocol: bool = False,
        fixed_team_mode: bool = False,
        no_chairman_mode: bool = False,
    ):
        self.pool_dir = Path(pool_dir)
        self.chairman_name = chairman_name
        self.constitution = constitution
        self.tool_registry = tool_registry or ToolRegistry(
            str(Path(__file__).parent.parent / "tools")
        )
        # ``max_seconds`` is the effective execution budget. Failed,
        # retryable provider requests are discounted from it; the independent
        # wall-clock cap stops an unhealthy upstream from extending a case
        # forever.
        self.max_seconds = max_seconds
        self.wall_clock_seconds = (
            max_seconds if wall_clock_seconds is None else wall_clock_seconds
        )
        self.max_messages = max_messages
        self.idle_timeout = idle_timeout
        self.enable_reflection = enable_reflection
        self.max_cost_usd = max_cost_usd
        self.reflection_phase_timeout = (
            reflection_phase_timeout if reflection_phase_timeout > 0
            else self.REFLECTION_PHASE_TIMEOUT
        )
        self.allowed_agent_ids = (
            frozenset(str(agent_id) for agent_id in allowed_agent_ids)
            if allowed_agent_ids is not None
            else None
        )
        if (
            self.allowed_agent_ids is not None
            and self.chairman_name in self.allowed_agent_ids
        ):
            raise ValueError(
                "allowed_agent_ids contains recruitable members only; "
                "the chairman must not be duplicated"
            )
        self.team_selection_metadata = dict(team_selection_metadata or {})
        # Adapters pass TeamSelection.to_dict(), where evaluation handoff
        # menus live under the selection's metadata field. Normalize that
        # shape once so messaging tools and context builders read the same
        # exact rule set.
        nested_selection_metadata = self.team_selection_metadata.get("metadata")
        if isinstance(nested_selection_metadata, dict):
            for key in ("handoff_family_id", "handoff_rules", "handoff_rule_details"):
                if key not in self.team_selection_metadata and key in nested_selection_metadata:
                    self.team_selection_metadata[key] = nested_selection_metadata[key]
        self.output_contract = output_contract or OutputContract()
        self.execution_policy = execution_policy
        self.native_message_validator = None
        self.native_tool_schemas: list[dict] = []
        self.native_tool_handler = None
        self.task_budget = None
        # Keep the legacy flag as the fallback for existing pools, but allow
        # MIX-COOP to give only the Chairman a profile view.  Execution
        # members should use their own skills and outgoing handoff rules,
        # rather than peer-profile summaries.
        self.chairman_teammate_profiles_enabled = bool(
            teammate_profiles_enabled
            if chairman_teammate_profiles_enabled is None
            else chairman_teammate_profiles_enabled
        )
        self.member_teammate_profiles_enabled = bool(
            teammate_profiles_enabled
            if member_teammate_profiles_enabled is None
            else member_teammate_profiles_enabled
        )
        self.teammate_profiles_enabled = (
            self.chairman_teammate_profiles_enabled
            or self.member_teammate_profiles_enabled
        )
        self.enforce_handoff_selection_protocol = bool(
            enforce_handoff_selection_protocol
        )
        self.fixed_team_mode = fixed_team_mode
        self.no_chairman_mode = no_chairman_mode

        self.runner_id: str = uuid.uuid4().hex

        self._pool_agents_cache: list[dict] | None = None
        self.api_tracker = api_tracker or APIReliabilityTracker()
        self._budget_started_at = budget_started_at
        self.chairman_forced_submission_seconds = (
            float(chairman_forced_submission_seconds)
            if chairman_forced_submission_seconds is not None
            and float(chairman_forced_submission_seconds) > 0
            else None
        )

        self.message_store = MessageStore()
        self.submission_bridge = SubmissionBridge(self)
        self.agents: dict[str, Agent] = {}
        # Lifecycle state and historical participation are different facts.
        # ``self.agents`` contains only currently active agents, while these
        # append-only ledgers preserve everyone who actually participated even
        # if ``stop_agent`` later removes them from the active set.
        self._participating_agent_ids: list[str] = []
        self._recruited_agent_ids: list[str] = []
        self._agent_tasks: dict[str, asyncio.Task] = {}  # name → asyncio.Task
        self._result: str | None = None
        self._final_output_locked: bool = False
        self._finalization_mode: str = "none"
        self._submission_bridge_used: bool = False
        self._global_service_agent_ids: frozenset[str] = (
            frozenset({self.output_contract.answer_agent_name})
            if self.output_contract.answer_agent_enabled
            else frozenset()
        )
        self._done = asyncio.Event()
        self._forced_termination: str | None = None
        self._chairman_submission_warning_sent = False
        self._deadline_submission_requested = False

        self._stopped_agents_cost: float = 0.0

        self._phase: str = "task_execution"
        self._all_reflection_agents: list[str] = []
        self._l2_reflection_agents: list[str] = []
        self._l1_completed: set[str] = set()
        self._l2_completed: set[str] = set()
        self._reflection_phase_start: float = 0.0
        self._reflection_phase_guard: asyncio.Task | None = None
        self._agent_dirs: dict[str, str] = {}
        self._task_validation = None
        self._task_validator = None
        self._pre_reflection_hook = None
        self._evolution_description_key: str = ""
        self._run_start_time: float = 0.0
        self._event_log = None
        self._cwd: str | None = None
        self._task: str = ""


    # ------------------------------------------------------------------

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)

    def _record_agent_participation(
        self,
        agent_name: str,
        *,
        recruited: bool,
    ) -> None:
        self._append_unique(self._participating_agent_ids, agent_name)
        if recruited:
            self._append_unique(self._recruited_agent_ids, agent_name)

    def _agent_usage_snapshot(self) -> dict[str, list[str]]:
        active_agent_ids = list(self.agents.keys())
        return {
            "agents_used": list(self._participating_agent_ids),
            "recruited_agent_ids": list(self._recruited_agent_ids),
            "active_agent_ids": active_agent_ids,
            "stopped_agent_ids": [
                agent_id
                for agent_id in self._participating_agent_ids
                if agent_id not in active_agent_ids
            ],
        }

    # ------------------------------------------------------------------

    def set_task_validator(self, validator) -> None:
        """Set task validation callback (injected by adapter)."""
        self._task_validator = validator

    def set_pre_reflection_hook(self, hook) -> None:
        """Set pre-reflection callback (injected by adapter for eval snapshot)."""
        self._pre_reflection_hook = hook

    # ------------------------------------------------------------------

    def load_agent_from_pool(self, agent_name: str) -> Agent:
        """Load an Agent from the Pool directory (does not start run_loop)."""
        if not self._is_agent_allowed(agent_name) and not self._is_global_service(agent_name):
            raise PermissionError(
                f"Agent '{agent_name}' is outside the selected team boundary."
            )
        agent_dir = self.pool_dir / agent_name
        if not agent_dir.exists():
            raise FileNotFoundError(f"Agent '{agent_name}' not found in pool: {agent_dir}")
        config_path = agent_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Agent '{agent_name}' has no config.yaml: {config_path}")
        load_profiles = (
            self.chairman_teammate_profiles_enabled
            if agent_name == self.chairman_name
            else self.member_teammate_profiles_enabled
        )
        agent = Agent(
            str(agent_dir),
            self.tool_registry,
            load_teammate_profiles_for_agent=load_profiles,
        )
        if agent_name == self.chairman_name:
            if self.chairman_teammate_profiles_enabled:
                # Chairman profile files are keyed by the teammate being
                # profiled. Scope the read-only view to the selected family;
                # profile content must never expand the frozen recruitable set.
                agent.teammate_profiles = self._scope_chairman_profiles(
                    agent.teammate_profiles,
                )
            else:
                agent.teammate_profiles = ""
        elif not self.member_teammate_profiles_enabled:
            agent.teammate_profiles = ""
        elif agent_name in (self.allowed_agent_ids or ()):
            # Recruited members receive concise summaries of selected-family
            # peers. Their own behavior remains governed by prompt/skills,
            # rather than by teammates' potentially self-reinforcing opinions.
            agent.teammate_profiles = self._build_family_agent_profile_view(
                agent_name,
            )
        return agent

    def l2_reflection_agents(self) -> list[str]:
        """Return the agents allowed to create teammate-profile observations."""
        if self.member_teammate_profiles_enabled:
            # Legacy symmetric L2 behavior for pools that have not opted into
            # Chairman-only profiles.
            return list(self._all_reflection_agents)
        if (
            self.chairman_teammate_profiles_enabled
            and self.chairman_name in self.agents
        ):
            return [self.chairman_name]
        return []

    def teammate_profile_persistence_scope(self) -> set[str] | None:
        """Return the profile writers whose artifacts may enter a new version.

        ``None`` preserves legacy symmetric-profile pools.  The MIX-COOP
        Chairman-only mode returns the Chairman singleton, so persistence
        derives its authority from the same runtime policy as L2.
        """
        if not self.teammate_profiles_enabled:
            return set()
        if self.member_teammate_profiles_enabled:
            return None
        if self.chairman_teammate_profiles_enabled:
            return {self.chairman_name}
        return set()

    def chairman_profile_targets(self) -> list[str]:
        """Only actual recruited workers may be profiled by the Chairman."""
        return [
            agent_id for agent_id in self._recruited_agent_ids
            if agent_id not in self._global_service_agent_ids
            and agent_id != self.chairman_name
        ]

    def _scope_chairman_profiles(self, profile_text: str) -> str:
        """Return only profiles belonging to the selected family members."""
        if self.allowed_agent_ids is None:
            return profile_text
        if not profile_text.strip() or not self.allowed_agent_ids:
            return ""

        try:
            import yaml
            profiles = yaml.safe_load(profile_text)
        except Exception:
            # A malformed profile must not leak an unrestricted historical
            # profile into the Chairman context.
            return ""
        if not isinstance(profiles, dict):
            return ""

        scoped = {
            str(agent_id): profiles[agent_id]
            for agent_id in profiles
            if str(agent_id) in self.allowed_agent_ids
        }
        if not scoped:
            return ""
        try:
            return yaml.safe_dump(
                scoped,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ).strip()
        except Exception:
            return ""

    def _build_family_agent_profile_view(self, agent_name: str) -> str:
        """Aggregate only selected-family observations for a recruited Agent."""
        if self.allowed_agent_ids is None or not self.allowed_agent_ids:
            return ""

        import yaml

        family_ids = [self.chairman_name, *sorted(self.allowed_agent_ids)]
        observations: dict[str, list[dict[str, Any]]] = {}
        for observer_id in dict.fromkeys(family_ids):
            profile_path = (
                self.pool_dir / observer_id / "evolution" / "teammate_profiles.yaml"
            )
            if not profile_path.exists():
                continue
            try:
                raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            for target_id, profile in raw.items():
                target_id = str(target_id)
                if target_id not in self.allowed_agent_ids:
                    continue
                if not isinstance(profile, dict):
                    continue
                observations.setdefault(target_id, []).append({
                    "observer": observer_id,
                    "profile": profile,
                })

        def compact(entries: list[dict[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for field_name in (
                "reliability", "strengths", "weaknesses",
                "communication_style", "notes",
            ):
                values = [
                    item["profile"].get(field_name)
                    for item in entries
                    if item["profile"].get(field_name) not in (None, "", [])
                ]
                if not values:
                    continue
                if field_name in {"strengths", "weaknesses", "notes"}:
                    merged: list[str] = []
                    for value in values:
                        items = value if isinstance(value, list) else [value]
                        for item in items:
                            item = str(item).strip()
                            if item and item not in merged:
                                merged.append(item)
                    if merged:
                        # Profiles are curated during L2 reflection; retain
                        # the complete deduplicated view rather than applying
                        # an arbitrary item-count limit at injection time.
                        result[field_name] = merged
                else:
                    result[field_name] = str(values[-1])
            return result

        peer_profiles = {
            target_id: compact(entries)
            for target_id, entries in observations.items()
            if target_id != agent_name and compact(entries)
        }

        if not peer_profiles:
            return ""
        return (
            "Relevant profiles for other selected family members:\n"
            + yaml.safe_dump(
                peer_profiles,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ).strip()
        )

    def start_agent(
        self,
        agent: Agent,
        event_log=None,
        cwd: str | None = None,
        global_service: bool = False,
    ) -> None:
        """Start an agent from the pool and begin its run loop."""
        name = agent.config.name
        if name == self.chairman_name or (
            not self._is_agent_allowed(name)
            and not (global_service and self._is_global_service(name))
        ):
            raise PermissionError(
                f"Agent '{name}' is outside the selected recruitable team."
            )
        if name in self.agents:
            return

        agent.bind_message_store(self.message_store)
        agent._runner_ref = self
        agent._reflection_enabled = self.enable_reflection
        self.agents[name] = agent
        self._record_agent_participation(
            name,
            recruited=not global_service,
        )

        if self.enable_reflection:
            from tools.primitives import rebuild_primitive_schemas
            agent._primitive_tools_refresher = lambda: rebuild_primitive_schemas(
                self, include_chairman_tools=False, agent_name=agent.config.name,
            )

        if hasattr(agent, 'agent_dir'):
            self._agent_dirs[name] = str(agent.agent_dir)

        # 🔥 Build context based on mode
        if self.no_chairman_mode:
            system_context = self._build_peer_agent_context(agent)
        else:
            system_context = self._build_agent_context(agent)

        task = asyncio.create_task(
            agent.run_loop(
                event_log=event_log,
                cwd=cwd,
                initial_task=None,
                system_context=system_context,
                idle_timeout=self.idle_timeout,
                # The global AnswerAgent must remain available until the
                # Chairman has finished evidence gathering and asks for the
                # required final synthesis.
                max_idle_rounds=(
                    max(5, int(self.max_seconds / max(self.idle_timeout, 1)) + 2)
                    if global_service and self.max_seconds > 0
                    else 5
                ),
            ),
            name=f"agent-{name}",
        )
        self._agent_tasks[name] = task

    def stop_agent(self, agent_name: str) -> bool:
        """Stop a running agent gracefully."""
        task = self._agent_tasks.get(agent_name)
        if task and not task.done():
            task.cancel()
            agent = self.agents.get(agent_name)
            if agent:
                self._stopped_agents_cost += getattr(agent, "_total_cost_usd", 0.0)
            self.message_store.unregister(agent_name)
            self.agents.pop(agent_name, None)
            self._agent_tasks.pop(agent_name, None)
            return True
        return False

    def get_agent_states(self) -> dict[str, dict]:
        """Return current state of all agents in the pool."""
        states = {}
        for name, agent in self.agents.items():
            states[name] = {
                "state": agent._state,
                "state_since": agent._state_since,
            }
        return states

    def list_pool_agents(self) -> list[dict]:
        """List available agents in the pool directory."""
        if self._pool_agents_cache is None:
            self._pool_agents_cache = self._scan_pool_agents()

        for entry in self._pool_agents_cache:
            entry["started"] = entry["name"] in self.agents
        return self._pool_agents_cache

    def _is_agent_allowed(self, agent_name: str) -> bool:
        if agent_name == self.chairman_name or self._is_global_service(agent_name):
            return True
        return (
            self.allowed_agent_ids is None
            or agent_name in self.allowed_agent_ids
        )

    def _is_global_service(self, agent_name: str) -> bool:
        return (
            self.output_contract.answer_agent_enabled
            and agent_name == self.output_contract.answer_agent_name
        )

    def _answer_agent_final_answer(self) -> str:
        """Return AnswerAgent's most recent exact answer, if it supplied one."""
        return self.submission_bridge.extract_answer()

    def validate_answer_agent_submission(self, output: str) -> str | None:
        return self.submission_bridge.validate(output)

    def _scan_pool_agents(self) -> list[dict]:
        import os
        import yaml
        model_override = os.environ.get("META_TEAM_MODEL")

        result = []
        for agent_dir in sorted(self.pool_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            config_path = agent_dir / "config.yaml"
            if not config_path.exists():
                continue
            try:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            name = raw.get("name", agent_dir.name)
            if name == self.chairman_name or self._is_global_service(name):
                continue
            if not self._is_agent_allowed(name):
                continue
            result.append({
                "name": name,
                "description": raw.get("description", ""),
                "model": model_override or raw.get("model", ""),
                "tools": raw.get("tools", []),
                "skills": raw.get("skills", []),
                "started": False,
            })
        return result

    # ------------------------------------------------------------------

    async def run(self, task: str, session=None) -> RunnerResult:
        """Execute the full task lifecycle: start chairman, wait for completion, run reflection."""
        start = self._budget_started_at or time.time()
        self._run_start_time = start
        event_log = session.event_log if session else None
        cwd = session.workspace if session else None
        self._event_log = event_log
        self._cwd = cwd
        self._task = task

        logger.info(
            "Runner[%s] starting: chairman=%s, pool=%s, effective_budget=%.0fs, "
            "wall_clock_cap=%.0fs, max_messages=%d, reflection=%s",
            self.runner_id[:8], self.chairman_name, self.pool_dir.name,
            self.max_seconds, self.wall_clock_seconds, self.max_messages,
            self.enable_reflection,
        )

        if event_log:
            event_log.log("runner.start", data={
                "task": task[:500],
                "chairman": self.chairman_name,
                "pool_dir": str(self.pool_dir),
                "effective_task_budget_seconds": self.max_seconds,
                "wall_clock_seconds": self.wall_clock_seconds,
                "enable_reflection": self.enable_reflection,
                "teammate_profiles_enabled": self.teammate_profiles_enabled,
                "chairman_teammate_profiles_enabled": (
                    self.chairman_teammate_profiles_enabled
                ),
                "member_teammate_profiles_enabled": (
                    self.member_teammate_profiles_enabled
                ),
                "output_contract": self.output_contract.name,
                "task_budget": self.task_budget.snapshot() if self.task_budget else None,
                "execution_policy": (
                    self.execution_policy.to_dict()
                    if self.execution_policy is not None else None
                ),
            })

        # 🔥 Load chairman/coordinator based on mode
        chairman = self.load_agent_from_pool(self.chairman_name)
        chairman.bind_message_store(self.message_store)
        chairman._runner_ref = self
        chairman._reflection_enabled = self.enable_reflection
        self.agents[self.chairman_name] = chairman
        self._record_agent_participation(
            self.chairman_name,
            recruited=False,
        )

        if hasattr(chairman, 'agent_dir'):
            self._agent_dirs[self.chairman_name] = str(chairman.agent_dir)

        # An AnswerAgent is benchmark-specific. Only contracts that request
        # one (currently GAIA) start it as a global final-answer service.
        if self.output_contract.answer_agent_enabled:
            answer_agent = self.load_agent_from_pool(
                self.output_contract.answer_agent_name,
            )
            self.start_agent(
                answer_agent, event_log=event_log, cwd=cwd, global_service=True,
            )

        # 🔥 Build context based on mode (coordinator vs chairman)
        if self.no_chairman_mode:
            chairman_context = self._build_coordinator_context()
        else:
            chairman_context = self._build_chairman_context()

        from tools.primitives import build_primitive_tools, rebuild_primitive_schemas, cleanup_handlers

        # 🔥 In no_chairman_mode, coordinator doesn't get primitive tools
        if self.no_chairman_mode:
            primitive_tools = []
            chairman._primitive_tools_refresher = None
        else:
            primitive_tools = build_primitive_tools(self, agent_name=self.chairman_name)
            chairman._primitive_tools_refresher = lambda: rebuild_primitive_schemas(
                self, include_chairman_tools=True, agent_name=self.chairman_name,
            )

        chairman_task = asyncio.create_task(
            chairman.run_loop(
                event_log=event_log,
                cwd=cwd,
                initial_task=f"## Task\n\n{task}",
                system_context=chairman_context,
                extra_tools=primitive_tools,
                idle_timeout=self.idle_timeout,
                max_idle_rounds=3,
            ),
            name=f"agent-{self.chairman_name}",
        )
        self._agent_tasks[self.chairman_name] = chairman_task

        # 🔥 Fixed team mode: pre-start all agents
        if self.fixed_team_mode:
            logger.info(
                "Runner[%s]: fixed_team_mode enabled, pre-starting all agents",
                self.runner_id[:8],
            )
            pool_agents = self.list_pool_agents()
            for agent_info in pool_agents:
                agent_name = agent_info["name"]
                if agent_name == self.chairman_name:
                    continue  # Chairman/coordinator already started
                if agent_name == self.GLOBAL_ANSWER_AGENT_NAME:
                    continue  # answer_agent handled separately
                try:
                    agent = self.load_agent_from_pool(agent_name)
                    self.start_agent(agent, event_log=event_log, cwd=cwd)
                    logger.info(
                        "Runner[%s]: pre-started agent %s",
                        self.runner_id[:8], agent_name,
                    )
                except Exception as e:
                    logger.warning(
                        "Runner[%s]: failed to pre-start %s: %s",
                        self.runner_id[:8], agent_name, e,
                    )

        guard_tasks = []
        if self.max_messages > 0:
            guard_tasks.append(
                asyncio.create_task(self._guard_max_messages(event_log))
            )
        if self.max_cost_usd > 0:
            guard_tasks.append(
                asyncio.create_task(self._guard_cost_limit(event_log))
            )

        #
        #

        async def _guard_chairman_exit():
            try:
                await chairman_task
            except (asyncio.CancelledError, Exception):
                pass
            await asyncio.sleep(self._CHAIRMAN_EXIT_GRACE_SECONDS)
            if not self._done.is_set():
                if event_log:
                    event_log.log("runner.chairman_exited_without_terminate", data={
                        "elapsed": round(time.time() - start, 2),
                        "phase": self._phase,
                    })
                bridge_result = await self.submission_bridge.finalize_available_answer(
                    reason="chairman_exited_without_commit",
                    event_log=event_log,
                )
                if not bridge_result.used:
                    self.message_store.terminate(reason="chairman_exited")
                    self._done.set()

        chairman_guard = asyncio.create_task(_guard_chairman_exit())
        guard_tasks.append(chairman_guard)

        timeout_kind = await self._wait_for_completion_with_budgets(start)
        if timeout_kind:
            wall_elapsed = round(time.time() - start, 2)
            effective_elapsed = round(self._effective_elapsed(start), 2)
            recovery_seconds = round(self.api_tracker.recovery_seconds(), 2)
            total_cost = round(self._get_total_cost_usd(), 4)
            limit = (
                self.max_seconds
                if timeout_kind == "effective_task_timeout"
                else self.wall_clock_seconds
            )
            if event_log:
                event_log.log("runner.timeout", data={
                    "kind": timeout_kind,
                    "effective_task_budget_seconds": self.max_seconds,
                    "wall_clock_seconds": self.wall_clock_seconds,
                    "wall_elapsed_seconds": wall_elapsed,
                    "effective_elapsed_seconds": effective_elapsed,
                    "api_recovery_seconds": recovery_seconds,
                    "total_cost_usd": total_cost,
                    "api_reliability": self.api_tracker.summary(),
                })
            self._forced_termination = (
                f"{timeout_kind} (wall {wall_elapsed}s, effective {effective_elapsed}s, "
                f"limit {limit}s, API recovery {recovery_seconds}s, cost ${total_cost})"
            )

        try:
            if (
                self._forced_termination
                and self._phase == "task_execution"
            ):
                for t in guard_tasks:
                    t.cancel()
                if guard_tasks:
                    await asyncio.gather(*guard_tasks, return_exceptions=True)
                guard_tasks.clear()

                # Stop normal agent loops before making the no-tool AnswerAgent
                # emergency call; otherwise an agent could issue a concurrent
                # normal request while its final output is being synthesized.
                execution_tasks = list(self._agent_tasks.values())
                for t in execution_tasks:
                    if not t.done():
                        t.cancel()
                if execution_tasks:
                    await asyncio.gather(*execution_tasks, return_exceptions=True)

                await self._emergency_finalize_after_forced_termination(event_log)

                if self.enable_reflection:
                    await self._enter_forced_reflection(event_log)

                if self.enable_reflection and not self._done.is_set():
                    reflection_total_timeout = self.reflection_phase_timeout * 3 + 60
                    try:
                        await asyncio.wait_for(
                            self._done.wait(), timeout=reflection_total_timeout,
                        )
                    except asyncio.TimeoutError:
                        if event_log:
                            event_log.log("runner.reflection_total_timeout", data={
                                "timeout": reflection_total_timeout,
                                "elapsed": round(time.time() - start, 2),
                                "phase": self._phase,
                            })
                        self.message_store.terminate(reason="forced_reflection_total_timeout")
        finally:
            cleanup_handlers(self)

        if self._agent_tasks:
            done, pending = await asyncio.wait(
                self._agent_tasks.values(),
                timeout=10.0,
                return_when=asyncio.ALL_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        for t in guard_tasks:
            t.cancel()
        if self._reflection_phase_guard and not self._reflection_phase_guard.done():
            self._reflection_phase_guard.cancel()
            guard_tasks.append(self._reflection_phase_guard)
        if guard_tasks:
            await asyncio.gather(*guard_tasks, return_exceptions=True)

        cost = time.time() - start
        total_cost_usd = self._get_total_cost_usd()

        # Reflection messages are never a task answer.  If the Chairman did
        # not submit an answer before the task deadline, preserve an explicit
        # no-output marker instead of leaking L1/L2 prose into evaluation.
        if not self._result and not self.enable_reflection:
            chairman = self.agents.get(self.chairman_name)
            fallback_output = ""
            fallback_reason = ""
            if chairman is not None and hasattr(chairman, "messages"):
                for msg in reversed(chairman.messages or []):
                    if msg.get("role") != "assistant":
                        continue
                    if msg.get("tool_calls"):
                        continue
                    content = msg.get("content") or ""
                    if isinstance(content, str) and len(content.strip()) >= 200:
                        fallback_output = content.strip()
                        fallback_reason = "last_pure_assistant_message"
                        break

                if not fallback_output:
                    for msg in reversed(chairman.messages or []):
                        if msg.get("role") != "assistant":
                            continue
                        content = msg.get("content") or ""
                        if isinstance(content, str) and len(content.strip()) >= 500:
                            fallback_output = content.strip()
                            fallback_reason = "last_assistant_content_with_tool_calls"
                            break

            if fallback_output:
                if event_log:
                    event_log.log("runner.fallback_output_from_assistant", data={
                        "reason": fallback_reason,
                        "output_len": len(fallback_output),
                    })
                logger.warning(
                    "Runner[%s]: Chairman did not call set_final_output/finalize_task; "
                    "falling back to last assistant content (%d chars, reason=%s)",
                    self.runner_id[:8], len(fallback_output), fallback_reason,
                )
                self._result = fallback_output

        output = self._result or "[no output — Chairman did not call set_final_output/finalize_task]"
        agent_usage = self._agent_usage_snapshot()

        logger.info(
            "Runner[%s] finished: %.1fs elapsed, $%.4f cost, agents=%s, "
            "phase=%s%s",
            self.runner_id[:8], cost, total_cost_usd,
            agent_usage["agents_used"], self._phase,
            f", forced_termination={self._forced_termination}" if self._forced_termination else "",
        )

        if event_log:
            event_log.log("runner.end", data={
                "output": output[:1000],
                "cost_seconds": round(cost, 2),
                "cost_usd": round(total_cost_usd, 4),
                "effective_task_seconds": round(self._effective_elapsed(start), 2),
                "api_recovery_seconds": round(self.api_tracker.recovery_seconds(), 2),
                **agent_usage,
            })

        return RunnerResult(
            output=output,
            agents=self.agents,
            cost_seconds=cost,
            metadata={
                **agent_usage,
                "enable_reflection": self.enable_reflection,
                "output_contract": self.output_contract.name,
                "execution_policy": (
                    self.execution_policy.to_dict()
                    if self.execution_policy is not None else None
                ),
                "global_service_agent_ids": sorted(self._global_service_agent_ids),
                "team_selection": self.team_selection_metadata,
                "allowed_agent_ids": (
                    sorted(self.allowed_agent_ids)
                    if self.allowed_agent_ids is not None
                    else None
                ),
                "phase": self._phase,
                "forced_termination": self._forced_termination,
                "finalization_mode": self._finalization_mode,
                "task_budget": self.task_budget.snapshot() if self.task_budget else None,
                "submission_bridge_used": self._submission_bridge_used,
                "effective_task_seconds": round(self._effective_elapsed(start), 3),
                "effective_task_budget_seconds": self.max_seconds,
                "wall_clock_seconds": self.wall_clock_seconds,
                "api_reliability": self.api_tracker.summary(),
                "global_service_agent_ids": sorted(self._global_service_agent_ids),
                "handoff_trace": list(getattr(self, "_handoff_trace", [])),
                "handoff_rule_usage": list(
                    getattr(self, "_handoff_rule_usage", [])
                ),
            },
        )

    # ------------------------------------------------------------------

    def request_forced_termination(self, reason: str) -> None:
        """Wake the execution wait and enter the existing emergency path."""
        if self._phase == "task_execution" and not self._result:
            self._forced_termination = self._forced_termination or reason
            if self._event_log:
                self._event_log.log("runner.budget_forced_termination", data={
                    "reason": self._forced_termination,
                    "task_budget": self.task_budget.snapshot() if self.task_budget else None,
                })
            self._done.set()

    def _effective_elapsed(self, started_at: float) -> float:
        """Elapsed task time after discounting failed retryable API requests."""
        return max(
            0.0,
            time.time() - started_at - self.api_tracker.recovery_seconds(),
        )

    async def _wait_for_completion_with_budgets(
        self, started_at: float,
    ) -> str | None:
        """Enforce the effective budget and independent recovery wall-clock cap.

        The dual budget applies to task execution only. Once a submitted task
        enters reflection, the existing reflection phase guards own its
        lifecycle and the task clock no longer cuts that phase short.
        """
        while not self._done.is_set():
            if self._phase != "task_execution":
                await self._done.wait()
                return None

            if self.task_budget is not None:
                reason = self.task_budget.submission_reason()
                if reason:
                    return f"macbench_budget_exceeded:{reason}"
            wall_elapsed = time.time() - started_at
            effective_elapsed = self._effective_elapsed(started_at)
            active_remaining = [
                remaining for remaining in (
                    self.max_seconds - effective_elapsed if self.max_seconds > 0 else None,
                    self.wall_clock_seconds - wall_elapsed if self.wall_clock_seconds > 0 else None,
                )
                if remaining is not None
            ]
            if (
                active_remaining
                and self.chairman_forced_submission_seconds is not None
                and min(active_remaining) <= self.chairman_forced_submission_seconds
            ):
                if self._event_log:
                    self._event_log.log("runner.chairman_forced_submission", data={
                        "remaining_seconds": round(min(active_remaining), 2),
                        "trigger_seconds": self.chairman_forced_submission_seconds,
                    })
                return "chairman_deadline_submission"

            if (
                active_remaining
                and not self._chairman_submission_warning_sent
                and self.chairman_forced_submission_seconds is None
            ):
                budget = min(
                    value for value in (self.max_seconds, self.wall_clock_seconds)
                    if value > 0
                )
                warning_lead = min(
                    self._CHAIRMAN_SUBMISSION_WARNING_MAX_SECONDS,
                    max(1.0, budget * self._CHAIRMAN_SUBMISSION_WARNING_FRACTION),
                )
                remaining = min(active_remaining)
                if remaining <= warning_lead:
                    self._chairman_submission_warning_sent = True
                    self._deadline_submission_requested = True
                    self.message_store.send(
                        sender="[SYSTEM]",
                        receiver=self.chairman_name,
                        content=(
                            "[DEADLINE WARNING] The task time limit is nearly exhausted. "
                            "Immediately submit the best supported result from work already "
                            "completed. Clearly mark uncertainty or unfinished parts; do not "
                            "claim unverified work is complete. Use the submission tool now."
                        ),
                    )
                    if self._event_log:
                        self._event_log.log("runner.chairman_submission_warning", data={
                            "remaining_seconds": round(remaining, 2),
                            "warning_lead_seconds": round(warning_lead, 2),
                        })
            if (
                self.wall_clock_seconds > 0
                and wall_elapsed >= self.wall_clock_seconds
            ):
                return "wall_clock_recovery_timeout"
            if self.max_seconds > 0 and effective_elapsed >= self.max_seconds:
                return "effective_task_timeout"

            waits = [1.0]
            if self.wall_clock_seconds > 0:
                waits.append(max(0.01, self.wall_clock_seconds - wall_elapsed))
            if self.max_seconds > 0:
                waits.append(max(0.01, self.max_seconds - effective_elapsed))
            if (
                active_remaining
                and not self._chairman_submission_warning_sent
                and self.chairman_forced_submission_seconds is None
            ):
                budget = min(
                    value for value in (self.max_seconds, self.wall_clock_seconds)
                    if value > 0
                )
                warning_lead = min(
                    self._CHAIRMAN_SUBMISSION_WARNING_MAX_SECONDS,
                    max(1.0, budget * self._CHAIRMAN_SUBMISSION_WARNING_FRACTION),
                )
                waits.append(max(0.01, min(active_remaining) - warning_lead))
            try:
                await asyncio.wait_for(self._done.wait(), timeout=min(waits))
            except asyncio.TimeoutError:
                continue
        return None

    # ------------------------------------------------------------------

    def _build_chairman_context(self) -> str:
        parts = []

        if self.constitution:
            parts.append(f"## Constitution\n\n{self.constitution}")

        # The Chairman is a real sender in the execution trace.  Give it only
        # its own outgoing family contracts, so plan_agent -> specialist rules
        # can be selected on the corresponding formal handoff without exposing
        # unrelated worker-to-worker rules.
        handoff_rules = self.team_selection_metadata.get("handoff_rules", [])
        outgoing_rules = [
            rule for rule in handoff_rules
            if rule.get("from_agent") == self.chairman_name
            and self.chairman_name not in self._global_service_agent_ids
        ]
        if outgoing_rules:
            parts.append(
                "## Family-Scoped Handoff Rules\n\n"
                "These are the active rules you may use when sending a formal handoff. "
                "Inspect each rule's trigger_description and rule_id; "
                "pass the selected rule_id in send_message, or NONE for ordinary "
                "coordination. A rule may only be used when you are its from_agent and "
                "you send to its to_agent.\n\n"
                f"{json.dumps(outgoing_rules, ensure_ascii=False, indent=2)}"
            )

        if self.execution_policy is not None:
            parts.append(self.execution_policy.chairman_context())

        if self.output_contract.answer_agent_enabled:
            parts.append(
                "## Current Native Output Contract\n\n"
                f"Submission type: `{self.output_contract.submission_type}`\n"
                "Required AnswerAgent prefix: "
                f"`{self.output_contract.answer_agent_prefix}`\n\n"
                f"{self.output_contract.chairman_instruction}\n"
                "Wait for a message from the AnswerAgent beginning exactly with "
                f"`{self.output_contract.answer_agent_prefix}` before submitting. "
                "Submit only the content following that prefix."
            )

        if self.enable_reflection:
            parts.append(
                "## Runtime Submission Protocol (Authoritative)\n\n"
                "This section overrides any submission-tool names or sequences in "
                "pool prompts, constitutions, skills, or teammate messages.\n\n"
                f"{self.output_contract.chairman_submission_hint} Call exactly "
                "`finalize_task(output=<final output>)`. This records and locks "
                "the benchmark answer, then starts reflection. Do not call "
                "`set_final_output` or `terminate` during task execution. After "
                "reflection completes, call `terminate`.\n\n"
                "## Your Role\n\n"
                "You are the Chairman of this Meta-Team. Your responsibilities:\n"
                "1. **Analyze the task** and decide which agents to recruit from the Pool\n"
                "2. **Use `list_pool`** to see available agents and their capabilities\n"
                "3. **Use `start_agent`** to recruit agents into the team\n"
                "4. **Use `send_message`** to assign tasks and coordinate work\n"
                "\n"
                "5. **Use `wait_for_replies`** to wait for agents to complete their work\n"
                "6. **Use the Runtime Submission Protocol above** to submit the final result\n"
                "7. After reflection is complete, **use `terminate`** to end the task\n\n"
                "All agents can communicate freely with each other via send_message.\n\n"
                "The Runtime Submission Protocol is the only source of truth for "
                "how this run ends."
            )
        else:
            parts.append(
                "## Runtime Submission Protocol (Authoritative)\n\n"
                "This section overrides any submission-tool names or sequences in "
                "pool prompts, constitutions, skills, or teammate messages.\n\n"
                f"{self.output_contract.chairman_submission_hint} Call exactly "
                "`set_final_output(output=<final output>)`, confirm success, and "
                "then call `terminate`. Do not call `finalize_task` in this run.\n\n"
                "## Your Role\n\n"
                "You are the Chairman of this Meta-Team. Your responsibilities:\n"
                "1. **Analyze the task** and decide which agents to recruit from the Pool\n"
                "2. **Use `list_pool`** to see available agents and their capabilities\n"
                "3. **Use `start_agent`** to recruit agents into the team\n"
                "4. **Use `send_message`** to assign tasks and coordinate work\n"
                "\n"
                "5. **Use `wait_for_replies`** to wait for agents to complete their work\n"
                "6. **Use the Runtime Submission Protocol above** to submit the final result\n"
                "7. **Use `terminate`** to end the task\n\n"
                "All agents can communicate freely with each other via send_message.\n\n"
                "The Runtime Submission Protocol is the only source of truth for "
                "how this run ends."
            )

        return "\n\n---\n\n".join(parts)

    def _build_agent_context(self, agent: Agent) -> str:
        parts = []

        if self.constitution:
            parts.append(f"## Constitution\n\n{self.constitution}")

        handoff_rules = self.team_selection_metadata.get("handoff_rules", [])
        if handoff_rules and agent.config.name not in self._global_service_agent_ids:
            outgoing_rules = [
                rule for rule in handoff_rules
                if rule.get("from_agent") == agent.config.name
            ]
        else:
            outgoing_rules = []
        if outgoing_rules:
            parts.append(
                "## Family-Scoped Handoff Rules\n\n"
                "These are the active rules you may use when sending a formal handoff. "
                "Inspect each rule's trigger_description and rule_id; "
                "pass the selected rule_id in send_message, or NONE for ordinary "
                "coordination. A rule may only be used when you are its from_agent and "
                "you send to its to_agent.\n\n"
                f"{json.dumps(outgoing_rules, ensure_ascii=False, indent=2)}"
            )

        if (
            self.execution_policy is not None
            and agent.config.name not in self._global_service_agent_ids
        ):
            native_role_context = self.execution_policy.agent_context(
                agent.config.name,
                agent.config.role,
                agent.config.tools,
            )
            if native_role_context:
                parts.append(native_role_context)

        if (
            self.output_contract.answer_agent_enabled
            and agent.config.name == self.output_contract.answer_agent_name
        ):
            parts.append(
                "## Current Native Output Contract\n\n"
                f"Submission type: `{self.output_contract.submission_type}`\n"
                f"Required response prefix: `{self.output_contract.answer_agent_prefix}`\n\n"
                f"{self.output_contract.answer_agent_instruction}\n"
                "Send the formatted response to the Chairman. The prefix must be "
                "the first non-whitespace text and must match exactly."
            )

        parts.append(
            "## Your Role\n\n"
            "You are a member of a Meta-Team, recruited by the Chairman for this task.\n"
            "- Wait for the Chairman to assign you work via message\n"
            "- Complete your assigned work using your tools\n"
            "- Report results back to the Chairman via send_message\n"
            "- You can also communicate with other team members directly"
        )

        return "\n\n---\n\n".join(parts)

    def _build_coordinator_context(self) -> str:
        """Build system context for the Coordinator (peer mode)."""
        parts = []

        if self.constitution:
            parts.append(f"## Constitution\n\n{self.constitution}")

        parts.append(
            "## Your Role\n\n"
            "You are the **Coordinator** of this peer team. You are NOT a Chairman or supervisor - "
            "you are an **equal peer** who helps organize the work.\n\n"
            "**Your teammates** (all already active and equal to you):\n"
        )

        team_list = []
        for agent_info in self.list_pool_agents():
            if agent_info['name'] == self.chairman_name:
                continue
            team_list.append(f"- **{agent_info['name']}**: {agent_info['description']}")
        parts.append("\n".join(team_list))

        parts.append(
            "\n\n**Your workflow**:\n"
            "1. **Analyze the task** and understand requirements\n"
            "2. **Discuss with peers** via send_message to decompose work\n"
            "3. **Coordinate (not command)** - suggest who might work on what\n"
            "4. **Collect results** using wait_for_replies\n"
            "5. **Synthesize** and submit the final answer\n\n"
            "**Key principles**:\n"
            "- You are a peer, not a boss\n"
            "- Make suggestions, don't give orders\n"
            "- Any agent can communicate with any other agent\n"
            "- Collaborative decision-making\n\n"
            "**Available tools**:\n"
            "- send_message: Talk to teammates\n"
            "- wait_for_replies: Collect responses\n"
            "- (Your own domain tools from config.yaml)\n\n"
        )

        # Output contract
        if self.output_contract.answer_agent_enabled:
            if self.enable_reflection:
                parts.append(
                    "## Output Protocol\n\n"
                    f"{self.output_contract.chairman_submission_hint}\n"
                    "After collecting all evidence, send it to answer_agent for formatting, "
                    "then call `finalize_task(output=<final output>)` to record the answer "
                    "and start reflection. After reflection completes, call `terminate`."
                )
            else:
                parts.append(
                    "## Output Protocol\n\n"
                    f"{self.output_contract.chairman_submission_hint}\n"
                    "After collecting all evidence, send it to answer_agent for formatting, "
                    "then call `set_final_output(output=<final output>)` and `terminate`."
                )
        else:
            if self.enable_reflection:
                parts.append(
                    "## Output Protocol\n\n"
                    "After completing the task, call `finalize_task(output=<final output>)` "
                    "to record the answer and start reflection. After reflection completes, "
                    "call `terminate`."
                )
            else:
                parts.append(
                    "## Output Protocol\n\n"
                    "After completing the task, call `set_final_output(output=<final output>)` "
                    "and then `terminate`."
                )

        return "\n\n---\n\n".join(parts)

    def _build_peer_agent_context(self, agent: Agent) -> str:
        """Build system context for peer agents (no chairman mode)."""
        parts = []

        if self.constitution:
            parts.append(f"## Constitution\n\n{self.constitution}")

        parts.append(
            "## Your Role\n\n"
            f"You are **{agent.config.name}**, a member of an equal peer team. "
            "There is no Chairman - all agents are peers.\n\n"
            f"**Your coordinator**: {self.chairman_name} helps organize work but is NOT your boss.\n\n"
            "**Your teammates** (all equal peers):\n"
        )

        team_list = []
        for agent_info in self.list_pool_agents():
            if agent_info['name'] == agent.config.name:
                continue
            team_list.append(f"- **{agent_info['name']}**: {agent_info['description']}")
        parts.append("\n".join(team_list))

        parts.append(
            "\n\n**How to work**:\n"
            "1. Wait for messages from teammates (especially the coordinator)\n"
            "2. Complete your assigned work using your tools\n"
            "3. Share results back via send_message\n"
            "4. You can also proactively communicate with any teammate\n"
            "5. Collaborate as equals - no hierarchy\n\n"
            "**Communication style**:\n"
            "- Collaborative, not subordinate\n"
            "- You can suggest ideas and ask questions\n"
            "- Any agent can talk to any agent\n"
        )

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------

    @staticmethod
    def _plain_text_emergency_history(messages: list[dict]) -> list[dict]:
        """Convert an agent/tool transcript into a valid no-tool chat history.

        Anthropic rejects a request without ``tools`` when its history still
        contains OpenAI tool-call fields or ``role="tool"`` messages.  The
        emergency path intentionally makes a no-tool completion, so retain
        the useful text while removing that protocol-specific structure.
        """
        clean_messages: list[dict] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if content is None:
                continue
            if not isinstance(content, str):
                content = str(content)
            if not content.strip():
                continue

            if role == "system":
                clean_role = "system"
            elif role == "assistant":
                clean_role = "assistant"
            else:
                # Tool results have no valid representation in a no-tool
                # request.  Present their text as already-observed evidence.
                clean_role = "user"
                if role == "tool":
                    content = f"[Previously gathered tool result]\n{content}"

            clean_messages.append({"role": clean_role, "content": content})

        return clean_messages

    async def _force_finalize_agent(
        self, finalizer_agent, chairman, event_log=None,
        *, chairman_deadline: bool = False,
    ) -> str:
        from core import llm

        if self.output_contract.emergency_mode == "strict_tagged_answer":
            format_instruction = (
                "Your entire response must be exactly one line in this form: "
                f"{self.output_contract.emergency_prefix} <answer>. "
                "Do not add any other text."
            )
        else:
            format_instruction = (
                "Start your response with the exact prefix "
                f"`{self.output_contract.emergency_prefix}` and then provide only "
                "the required final output. Do not add meta-commentary."
            )
        FORCE_FINISH_MSG = (
            "You have reached the resource budget limit "
            f"({self._forced_termination}). "
            "You can no longer call any tools. "
            "Based only on the emergency evidence packet below, "
            f"{self.output_contract.emergency_instruction} "
            f"{format_instruction}"
        )
        history = self._plain_text_emergency_history(chairman.messages)
        evidence = "\n\n".join(
            f"[{message['role'].upper()}]\n{message['content']}"
            for message in history
        )
        if self.message_store:
            pending_reports = self.message_store.peek(
                self.chairman_name, unread_only=False,
            )
            if pending_reports:
                evidence += "\n\n[Messages delivered to Chairman]\n" + "\n\n".join(
                    f"[{message.sender}]\n{message.content}"
                    for message in pending_reports
                )
        system_prompt = ""
        builder = getattr(finalizer_agent, "build_system_prompt", None)
        if callable(builder):
            system_prompt = builder()
        if chairman_deadline:
            system_prompt += (
                "\n\n## HARD DEADLINE OVERRIDE\n"
                "The task has reached the forced submission point with 90 seconds "
                "remaining. You are the Chairman and must now produce the best "
                "complete answer supported by the evidence already in your transcript. "
                "Do not wait for AnswerAgent or teammates, request more work, or call "
                "tools. Follow the exact answer prefix and native format below."
            )
        force_msgs = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "## Emergency Evidence Packet\n\n"
                    f"Original task:\n{self._task}\n\n"
                    f"Chairman transcript and gathered evidence:\n{evidence}\n\n"
                    f"## Required action\n{FORCE_FINISH_MSG}"
                ),
            },
        ]

        completion_limit = max(4096, finalizer_agent.config.max_tokens)
        if self.task_budget is not None:
            # The final request must fit inside the reserved budget. Keep the
            # instruction intact and retain the newest evidence, bounded by bytes.
            def clipped(text, limit, tail=False):
                raw = text.encode("utf-8")
                return (raw[-limit:] if tail else raw[:limit]).decode("utf-8", errors="ignore")

            completion_limit = min(2048, self.task_budget.reserve_tokens // 4)
            force_msgs = [
                {"role": "system", "content": (
                    "You are the Chairman. Submit the currently completed work now. "
                    "No tools, delegation, or further work is allowed. "
                    "Distinguish verified completion from incomplete work. "
                    + format_instruction
                )},
                {"role": "user", "content": (
                    "Original task (possibly truncated):\n" + clipped(self._task, 2000)
                    + "\nLatest evidence (possibly truncated):\n" + clipped(evidence, 7000, tail=True)
                    + "\n" + FORCE_FINISH_MSG
                )},
            ]
            from core.task_budget import prompt_token_reservation
            # Small synthetic budgets also get a bounded final prompt.
            while (prompt_token_reservation(force_msgs) + completion_limit
                   > self.task_budget.reserve_tokens and len(force_msgs[1]["content"]) > 100):
                force_msgs[1]["content"] = force_msgs[1]["content"][len(force_msgs[1]["content"]) // 4:]

        if event_log:
            event_log.log("runner.force_finalize_start",
                agent=finalizer_agent.config.name, data={
                    "reason": self._forced_termination,
                    "history_len": len(chairman.messages),
                    "finalizer": finalizer_agent.config.name,
                    "chairman_deadline": chairman_deadline,
                    "plain_history_len": len(history),
                })

        request_timeout = (
            min(45.0, max(5.0, self.chairman_forced_submission_seconds / 2))
            if chairman_deadline and self.chairman_forced_submission_seconds
            else 240.0
        )
        if self.task_budget is not None:
            request_timeout = min(request_timeout, self.task_budget.remaining_seconds())
        emergency_kwargs = {"budget_emergency": True} if self.task_budget else {}
        response = await asyncio.wait_for(
            llm.complete(
                model=finalizer_agent.config.model,
                messages=force_msgs,
                temperature=finalizer_agent.config.temperature,
                max_tokens=completion_limit,
                api_tracker=self.api_tracker,
                **emergency_kwargs,
            ),
            timeout=request_timeout,
        )

        if not response.choices:
            return ""
        msg = response.choices[0].message
        content = (msg.content or "").strip()

        try:
            if hasattr(finalizer_agent, "_track_cost_usd"):
                finalizer_agent._track_cost_usd(response)
        except Exception as e:
            logger.debug("Cost tracking failed for force-finalize: %s", e)

        return content

    async def _emergency_finalize_after_forced_termination(self, event_log=None) -> None:
        """Lock a best-effort answer before reflection after a forced stop."""
        if (
            self._result
            or not self._forced_termination
            or not self.output_contract.has_emergency_finalize
        ):
            return

        chairman = self.agents.get(self.chairman_name)
        answer_agent = self.agents.get(self.output_contract.answer_agent_name)
        chairman_deadline = "chairman_deadline_submission" in self._forced_termination
        chairman_budget = self.task_budget is not None
        finalizer = chairman if (chairman_deadline or chairman_budget) else answer_agent
        if chairman is None or finalizer is None or len(getattr(chairman, "messages", []) or []) < 2:
            return

        try:
            forced_output = await self._force_finalize_agent(
                finalizer, chairman, event_log,
                chairman_deadline=chairman_deadline,
            )
            if not forced_output.strip():
                if event_log:
                    event_log.log("runner.force_finalize_empty", data={
                        "reason": self._forced_termination,
                    })
                return
            lines = [line.strip() for line in forced_output.splitlines() if line.strip()]
            prefix_pattern = re.escape(self.output_contract.emergency_prefix)
            strict_one_line = self.output_contract.emergency_mode == "strict_tagged_answer"
            if strict_one_line:
                valid = len(lines) == 1 and bool(re.fullmatch(
                    rf"{prefix_pattern}\s*.+", lines[0], flags=re.IGNORECASE,
                ))
            else:
                valid = bool(re.match(
                    rf"^\s*{prefix_pattern}\s*.+", forced_output,
                    flags=re.IGNORECASE | re.DOTALL,
                ))
            if not valid:
                if event_log:
                    event_log.log("runner.force_finalize_rejected", data={
                        "reason": self._forced_termination,
                        "output_len": len(forced_output),
                        "rejection": (
                            "expected_single_final_answer_line"
                            if strict_one_line else "expected_tagged_final_output"
                        ),
                    })
                logger.info(
                    "Runner[%s]: rejected non-strict emergency finalization output "
                    "(trigger=%s)", self.runner_id[:8], self._forced_termination,
                )
                return
            answer = re.sub(
                rf"^\s*{prefix_pattern}\s*", "", forced_output,
                flags=re.IGNORECASE,
            ).strip()
            if not answer:
                if event_log:
                    event_log.log("runner.force_finalize_rejected", data={
                        "reason": self._forced_termination,
                        "output_len": len(forced_output),
                        "rejection": "empty_final_answer",
                    })
                return
            if chairman_deadline or chairman_budget:
                self._finalization_mode = (
                    "chairman_budget_emergency" if chairman_budget else "chairman_forced_deadline"
                )
                await self.set_final_output(answer)
            else:
                # Timeout fallback remains synthesized by the AnswerAgent.
                self._result = answer
                self._final_output_locked = True
                self._finalization_mode = "emergency"
            if event_log:
                event_log.log("runner.force_finalize_success", data={
                    "reason": self._forced_termination,
                    "output_len": len(self._result or ""),
                    "finalization_mode": self._finalization_mode,
                })
            logger.info(
                "Runner[%s]: emergency finalization recorded %d-char output "
                "(trigger=%s)",
                self.runner_id[:8], len(self._result or ""),
                self._forced_termination,
            )
        except Exception as e:
            if event_log:
                event_log.log("runner.force_finalize_error", data={
                    "reason": self._forced_termination,
                    "error": str(e)[:500],
                })
            logger.warning("Runner[%s]: emergency finalization failed: %s", self.runner_id[:8], e)

    # ------------------------------------------------------------------

    async def _guard_max_messages(self, event_log=None) -> None:
        try:
            while not self._done.is_set():
                await asyncio.sleep(5.0)
                if self._phase != "task_execution":
                    continue
                if self.message_store.total_count >= self.max_messages:
                    if event_log:
                        event_log.log("runner.guard", data={
                            "reason": "max_messages",
                            "count": self.message_store.total_count,
                            "limit": self.max_messages,
                        })
                    self._forced_termination = (
                        f"max_messages ({self.message_store.total_count} >= {self.max_messages})"
                    )
                    self._done.set()
                    break
        except asyncio.CancelledError:
            pass

    def _get_total_cost_usd(self) -> float:
        total = self._stopped_agents_cost
        for agent in self.agents.values():
            total += getattr(agent, "_total_cost_usd", 0.0)
        return total

    REFLECTION_MAX_COST_USD = 30.0

    async def _guard_cost_limit(self, event_log=None) -> None:
        try:
            reflection_cost_baseline: float | None = None

            while not self._done.is_set():
                await asyncio.sleep(10.0)
                total_cost = self._get_total_cost_usd()

                if self._phase == "task_execution":
                    if total_cost >= self.max_cost_usd:
                        if event_log:
                            event_log.log("runner.guard", data={
                                "reason": "cost_limit",
                                "total_cost_usd": round(total_cost, 4),
                                "limit_usd": self.max_cost_usd,
                            })
                        self._forced_termination = (
                            f"cost_limit (${total_cost:.2f} >= ${self.max_cost_usd:.2f})"
                        )
                        self._done.set()
                        break
                else:
                    if self.REFLECTION_MAX_COST_USD <= 0:
                        continue
                    if reflection_cost_baseline is None:
                        reflection_cost_baseline = total_cost
                    reflection_cost = total_cost - reflection_cost_baseline
                    if reflection_cost >= self.REFLECTION_MAX_COST_USD:
                        if event_log:
                            event_log.log("runner.guard", data={
                                "reason": "reflection_cost_limit",
                                "reflection_cost_usd": round(reflection_cost, 4),
                                "limit_usd": self.REFLECTION_MAX_COST_USD,
                                "total_cost_usd": round(total_cost, 4),
                            })
                        self.terminate(
                            reason=f"reflection_cost_limit_${reflection_cost:.2f}"
                                   f">=${self.REFLECTION_MAX_COST_USD:.2f}"
                        )
                        break
        except asyncio.CancelledError:
            pass
