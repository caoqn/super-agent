import math
import shutil

import numpy as np
from collections import deque
import random
import time
from colorama import init, Fore, Back, Style
from framework.env import Environment
from masbench.task import *
from masbench.physics import Mesh, Node, CapacityMesh
from masbench.enhanced_terrain import EnhancedCapacityMesh, Item
from masbench.terrain_adapter import TerrainAdapter, integrate_terrain_system
from masbench.agent import Actions  # Action constants
from masbench.coords import to_math_xy  # Convert top-left to bottom-left coordinates.
from masbench.output_logger import OutputManager  # Output redirection utilities
from masbench.mechanisms import MessageCandidate, OutgoingMessage, ParsedIncomingMessage, RoundContext
from enum import Enum
from typing import Dict, Tuple, Any, Optional

dirc_map = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

class ActionResult(Enum):
    """Action execution result enumeration."""
    SUCCESS = "success"
    FAILED_CAPACITY = "failed_capacity"
    FAILED_NO_ITEMS = "failed_no_items"
    FAILED_NO_SPACE = "failed_no_space"
    FAILED_INVALID = "failed_invalid"
    FAILED_SINGLE_ITEM_CONSTRAINT = "failed_single_item_constraint"


class MASEnvironment(Environment):
    def __init__(self, env_name, agents, logger, task=1,
                 max_round=100, width=12, height=12, seed=42, view_size=9,
                 communication_size=15, food_capacities=None, nest_capacities=None,
                 custom_map: dict | None = None,
                 message_phase_enabled: bool = True,
                 messages_enabled: bool = True,
                 message_phase_timeout_s: int = 60,
                 message_phase_concurrency: int | None = None,
                 explicit_interaction_enabled: bool = True,
                 auto_clamp: bool = False,
                 # Message memory and one-message-per-round mode.
                 enable_message_memory: bool = False,
                 message_memory_k_sent: int = 3,
                 message_memory_k_recv: int = 3,
                 single_message_per_round: bool = False,
                 single_message_prompt_mode: str = 'basic',
                 ring_mode: str = 'angle_halves',
                 # Protocol V2.
                 protocol_v2_enabled: bool = False,
                 protocol_v2_auto: bool = True,
                 ask_cooldown_rounds: int = 2,
                 max_items_in_message: int = 5,
                 max_tables_in_message: int = 3,
                 # Protocol V2: free-text portion and header length limits.
                 proto_v2_free_enable: bool = False,
                 proto_v2_free_slot: str = 'C',
                 proto_v2_free_max: int = 150,
                 proto_v2_header_max_len: int = 0,
                 proto_v2_total_max_len: int = 220,
                 mechanism_preset: str = 'base',
                 mechanism_raw_config: dict | None = None,
                 mechanism_effective_config: dict | None = None):
        print("DEBUG: Entered MASEnvironment.__init__")
        super().__init__(env_name, agents, logger)

        # Initialize the output redirection manager.
        self.output_manager = OutputManager(log_dir="txt_logs", experiment_name=env_name)
        self.log_file_path = None
        print(f"DEBUG: [OUTPUT_LOGGER] Output manager initialized; experiment: {env_name}")
        self.task = task
        self.width = width
        self.height = height
        self.grid = np.full((self.height, self.width), '.', dtype=str)
        self.round = 0
        self.done = False
        self.messages = []
        # self.agents_pos = {}
        self.tasks = {
            'Transport': Transport(seed),
            'Flocking': Flocking(seed),
            'Pursuit': Pursuit(seed),
            'Synchronization': Synchronization(seed),
            'Foraging': Foraging(seed, food_capacities, nest_capacities)
        }
        self.push_i = {}
        self.push_j = {}
        self.actions = {}
        self.agent_meshes = {}
        self.mesh_map = np.full_like(self.grid, None, dtype=object)
        # Retain meshes for backward compatibility; use mesh_map for primary management.
        self.meshes = []
        self.max_round = max_round
        self.__grid_dtype = None
        self.seed = seed
        self.view_size = view_size
        self.communication_size = communication_size
        # Optional custom map: reset builds the environment from it when provided.
        self.custom_map = custom_map
        # Message phase switches and parameters.
        self.message_phase_enabled = message_phase_enabled
        self.messages_enabled = bool(messages_enabled)
        self.mechanism_preset = str(mechanism_preset or 'base')
        self.mechanism_raw_config = dict(mechanism_raw_config or {})
        self.mechanism_effective_config = dict(mechanism_effective_config or {})
        self._compat_stable_fair = False
        self.message_phase_log_immediately = False
        self.message_phase_include_action_intent = False
        try:
            runtime_cfgs = []
            compat_cfgs = []
            for agent in (agents or {}).values():
                bundle = getattr(agent, 'mechanisms', None)
                runtime_cfg = dict((getattr(bundle, 'config', {}) or {}).get('runtime') or {})
                if runtime_cfg:
                    runtime_cfgs.append(runtime_cfg)
                compat_cfg = dict((getattr(bundle, 'config', {}) or {}).get('compat') or {})
                if str(compat_cfg.get('profile') or '') == 'stable_fair':
                    compat_cfgs.append(compat_cfg)
            if runtime_cfgs and len(runtime_cfgs) == len(agents or {}):
                runtime_cfg = runtime_cfgs[0]
                message_phase_order = str(runtime_cfg.get('message_phase_order') or '').strip()
                if message_phase_order in ('before_update', 'after_update'):
                    self.message_phase_order = message_phase_order
                if 'message_phase_use_cached_obs' in runtime_cfg:
                    self.message_phase_use_cached_obs = bool(runtime_cfg.get('message_phase_use_cached_obs'))
                if 'message_phase_log_immediately' in runtime_cfg:
                    self.message_phase_log_immediately = bool(runtime_cfg.get('message_phase_log_immediately'))
                if 'message_phase_include_action_intent' in runtime_cfg:
                    self.message_phase_include_action_intent = bool(runtime_cfg.get('message_phase_include_action_intent'))
                runtime_messages_enabled = runtime_cfg.get('messages_enabled')
                if runtime_messages_enabled is False:
                    self.messages_enabled = False
            if compat_cfgs and len(compat_cfgs) == len(agents or {}):
                self._compat_stable_fair = True
                compat_messages_enabled = compat_cfgs[0].get('messages_enabled')
                if compat_messages_enabled is False:
                    self.messages_enabled = False
        except Exception:
            pass
        # Individual contribution statistics (accumulated only for PUT_DOWN onto a table).
        self.agent_contributions = {an: {} for an in agents}
        self.message_phase_timeout_s = message_phase_timeout_s
        self.message_phase_concurrency = message_phase_concurrency
        # Explicit interaction mode: PICK_UP/PUT_DOWN require QTY/TYPE.
        self.explicit_interaction_enabled = explicit_interaction_enabled
        self.auto_clamp = auto_clamp
        # Message memory and one-message-per-round mode.
        self.enable_message_memory = bool(enable_message_memory)
        self.message_memory_k_sent = int(message_memory_k_sent)
        self.message_memory_k_recv = int(message_memory_k_recv)
        self.single_message_per_round = bool(single_message_per_round)
        self.single_message_prompt_mode = str(single_message_prompt_mode or 'basic')
        self.ring_mode = str(ring_mode or 'angle_halves')
        # Protocol V2.
        self.protocol_v2_enabled = bool(protocol_v2_enabled)
        self.protocol_v2_auto = bool(protocol_v2_auto)
        self.ask_cooldown_rounds = int(ask_cooldown_rounds)
        self.max_items_in_message = int(max_items_in_message)
        self.max_tables_in_message = int(max_tables_in_message)
        # Protocol V2 free-text portion and length.
        self.proto_v2_free_enable = bool(proto_v2_free_enable)
        self.proto_v2_free_slot = str(proto_v2_free_slot or 'C')
        self.proto_v2_free_max = int(proto_v2_free_max)
        self.proto_v2_header_max_len = int(proto_v2_header_max_len)
        try:
            # Safety limit for final concatenation if truncation is needed.
            self.proto_v2_total_max_len = int(proto_v2_total_max_len)
        except Exception:
            self.proto_v2_total_max_len = 220
        # Initialize message memory structures.
        self.recent_msgs = {}
        for an in agents:
            try:
                self.recent_msgs[an] = {
                    'sent': deque(maxlen=self.message_memory_k_sent),
                    'recv': deque(maxlen=self.message_memory_k_recv)
                }
            except Exception:
                self.recent_msgs[an] = {'sent': deque(maxlen=3), 'recv': deque(maxlen=3)}
        # Routing helpers: track recent targets and questioners already answered.
        self.routing_history_len = 2
        self.routing_recent_targets = {an: deque(maxlen=self.routing_history_len) for an in agents}
        self.routing_reply_cooldown_window = max(1, self.ask_cooldown_rounds)
        print(f"DEBUG: [MSG_MEMORY] enable={self.enable_message_memory}, sentK={self.message_memory_k_sent}, recvK={self.message_memory_k_recv}")
        print(f"DEBUG: [MSG_LIMIT] single_message_per_round={self.single_message_per_round}")
        print(f"DEBUG: [MSG_LIMIT] single_message_prompt_mode={self.single_message_prompt_mode}")
        print(f"DEBUG: [PROTO_V2] enabled={self.protocol_v2_enabled}, auto={self.protocol_v2_auto}, cooldown={self.ask_cooldown_rounds}, max_items={self.max_items_in_message}, max_tables={self.max_tables_in_message}")
        print(f"DEBUG: [PROTO_V2_FREE] enable={self.proto_v2_free_enable}, slot={self.proto_v2_free_slot}, free_max={self.proto_v2_free_max}, header_max_len={self.proto_v2_header_max_len}, total_max={self.proto_v2_total_max_len}")

        # Validate parameters.
        if communication_size < view_size:
            print(f"WARNING: communication_size ({communication_size}) < view_size ({view_size}); this may conflict with the intended logic")

        print(f"DEBUG: MASEnvironment initialized - observation range: {view_size}, communication range: {communication_size}")
        print(f"DEBUG: [MESSAGING] enabled={self.messages_enabled}, message_phase_enabled={self.message_phase_enabled}")

        # Initialize the terrain adapter.
        print("DEBUG: Initializing terrain adapter")
        self.terrain_adapter = TerrainAdapter(self)

        # Direct position-to-items mapping.
        self.position_items = {}  # {(y, x): [Item, Item, ...]} maps positions to item lists.

        # init colorama
        init()
        # init task
        print("DEBUG: In __init__, about to call self.reset()")
        self.reset()

    @property
    def grid_dtype(self):
        if self.__grid_dtype is None:
            self.__grid_dtype = f'<U{max([len(s) for s in self.agents]) - 5}'
        return self.__grid_dtype

    def reset(self):
        print("DEBUG: Entered MASEnvironment.reset")

        # Start output redirection when resetting the environment.
        if self.output_manager and not self.output_manager.is_redirected:
            self.log_file_path = self.output_manager.start_logging()
            print(f"DEBUG: [OUTPUT_LOGGER] Terminal output redirection started; log file: {self.log_file_path}")

        # reset grids
        self.__grid_dtype = None
        self.grid = np.full((self.height, self.width), '.', dtype=str)
        self.mesh_map = np.full_like(self.grid, None, dtype=object)
        self.messages = []
        self.round = 0
        self.agent_meshes = {}
        self.meshes = []  # Reset the meshes list.
        self.position_items = {}  # Reset the item position mapping.
        # Reset communication memory structures.
        for an in self.agents:
            try:
                self.recent_msgs[an]['sent'].clear()
                self.recent_msgs[an]['recv'].clear()
            except Exception:
                self.recent_msgs[an] = {
                    'sent': deque(maxlen=self.message_memory_k_sent),
                    'recv': deque(maxlen=self.message_memory_k_recv)
                }
            self.routing_recent_targets.setdefault(an, deque(maxlen=self.routing_history_len)).clear()
            try:
                bundle = getattr(self.agents[an], 'mechanisms', None)
                if bundle is not None:
                    bundle.memory.reset()
                    bundle.routing.reset()
                    bundle.protocol.reset()
            except Exception as e:
                print(f"WARNING: Failed to clear mechanism state during reset: agent={an}, err={e}")

        # Build from the custom map when available; otherwise use default task generation.
        if self.custom_map is not None:
            print("DEBUG: Custom map detected; building environment from JSON")
            self._reset_with_custom_map(self.custom_map)
        else:
            print("DEBUG: In reset, about to call task.reset()")
            self.tasks[self.task].reset(self)

            # Integrate terrain after the task has created it.
            print("DEBUG: Integrating enhanced terrain system")
            integrate_terrain_system(self)

        # log init state
        self.logger.log_game_state(self, 0, time.time())
        print(self.render(use_color=True))  # Use colors in terminal output.
        return self.obs(None)  # return initial state

    def _ensure_message_phase_slot(self, name: str):
        self.delayed_actions = getattr(self, 'delayed_actions', {})
        if name not in self.delayed_actions:
            agent = self.agents[name]
            obs = self.get_round_obs(agent)
            visible_msgs = obs.get('messages') or []
            msgs_summary = [{'from': m.get('agent_name'), 'text': m.get('message')} for m in visible_msgs]
            self.delayed_actions[name] = {
                'prompt': '',
                'response': '',
                'action': 'STAY',
                'message': '',
                'api_call_time': 0,
                'action_success': True,
                'view': obs.get('view', self.get_view(name)),
                'usage': self._collect_message_tokens(agent),
                'messages_received': msgs_summary,
                'subactions': {'message_phase': {'decisions': [], 'sent': []}}
            }
        else:
            self.delayed_actions[name].setdefault('subactions', {})
            self.delayed_actions[name]['subactions'].setdefault('message_phase', {'decisions': [], 'sent': []})
        return self.delayed_actions[name]

    def _obs_messages_to_dict(self, obs: dict) -> dict[str, str]:
        msg_map = {}
        for msg in (obs.get('messages') or []):
            sender = msg.get('agent_name')
            text = msg.get('message')
            if sender is not None and text is not None:
                msg_map[str(sender)] = str(text)
        return msg_map

    def _make_round_context(self, agent, obs: dict, phase: str = 'message') -> RoundContext:
        bundle = getattr(agent, 'mechanisms', None)
        ctx_obs = obs
        if phase == 'message':
            use_raw_message_obs = False
            try:
                routing = getattr(bundle, 'routing', None) if bundle is not None else None
                if routing is not None and hasattr(routing, 'requires_raw_message_obs'):
                    use_raw_message_obs = bool(routing.requires_raw_message_obs())
            except Exception:
                use_raw_message_obs = False
            try:
                runtime_cfg = dict((getattr(bundle, 'config', {}) or {}).get('runtime') or {})
                if not use_raw_message_obs:
                    use_raw_message_obs = bool(runtime_cfg.get('raw_message_obs', False))
            except Exception:
                pass
            try:
                compat_cfg = {}
                compat_cfg = dict((getattr(bundle, 'config', {}) or {}).get('compat') or {})
                if not use_raw_message_obs:
                    use_raw_message_obs = bool(compat_cfg.get('raw_message_obs', False))
            except Exception:
                pass
            if not use_raw_message_obs:
                ctx_obs = self._sanitize_message_obs(agent.name, obs)
        return RoundContext(
            round_num=int(ctx_obs.get('round', self.round) or self.round),
            agent_name=agent.name,
            obs=ctx_obs,
            received_messages=self._obs_messages_to_dict(ctx_obs),
            phase=phase,
            config=(getattr(bundle, 'config', {}) if bundle is not None else {}),
        )

    def _message_looks_like_question(self, text: str) -> bool:
        low = str(text or '').strip().lower()
        if not low:
            return False
        if '?' in low:
            return True
        markers = (
            'where', 'what', 'which', 'who', 'when', 'how',
            'need', 'can you', 'could you', 'do you know', 'tell me',
            'is there',
        )
        return any(marker in low for marker in markers)

    def _sanitize_message_obs(self, agent_name: str, obs: dict) -> dict:
        safe_obs = dict(obs or {})
        safe_agents = {}
        recent_targets = set()
        try:
            agent = self.agents.get(agent_name)
            bundle = getattr(agent, 'mechanisms', None) if agent is not None else None
            routing = getattr(bundle, 'routing', None) if bundle is not None else None
            recent_targets = {str(name) for name in list(getattr(routing, 'recent_targets', []) or []) if name}
        except Exception:
            recent_targets = set()

        recent_askers = set()
        try:
            for msg in (obs.get('messages') or []):
                sender = str(msg.get('agent_name') or '').strip()
                text = str(msg.get('message') or '')
                if sender and self._message_looks_like_question(text):
                    recent_askers.add(sender)
        except Exception:
            recent_askers = set()

        for target_name, info in sorted((obs.get('agents_in_communication_range') or {}).items()):
            if target_name == agent_name:
                continue
            info = info or {}
            safe_agents[target_name] = {
                'in_view_range': bool(info.get('in_view_range', False)),
                'asked_last_round': target_name in recent_askers,
                'recently_contacted': target_name in recent_targets,
            }
        safe_obs['agents_in_communication_range'] = safe_agents
        return safe_obs

    def _build_message_candidates(self, name: str, obs: dict) -> list[MessageCandidate]:
        candidates = []
        for target_name, info in sorted((obs.get('agents_in_communication_range') or {}).items()):
            if target_name == name:
                continue
            info = info or {}
            candidates.append(
                MessageCandidate(
                    agent_name=target_name,
                    position={},
                    in_view=bool(info.get('in_view_range', False)),
                    raw_info={
                        'in_view_range': bool(info.get('in_view_range', False)),
                        'asked_last_round': bool(info.get('asked_last_round', False)),
                        'recently_contacted': bool(info.get('recently_contacted', False)),
                    },
                )
            )
        return candidates

    def _get_memory_state(self, name: str) -> dict:
        agent = self.agents.get(name)
        bundle = getattr(agent, 'mechanisms', None) if agent is not None else None
        if bundle is not None:
            try:
                return dict(bundle.memory.export_state() or {})
            except Exception as e:
                print(f"WARNING: [MEMORY] export_state failed for {name}: {e}")
        rm = self.recent_msgs.get(name, {})
        return {
            'tables': [],
            'items': [],
            'messages': {
                'sent': list(rm.get('sent', [])) if rm else [],
                'recv': list(rm.get('recv', [])) if rm else [],
            },
        }

    def _get_message_memory_buffers(self, name: str) -> tuple[list[dict], list[dict]]:
        state = self._get_memory_state(name)
        msg_state = state.get('messages', {}) if isinstance(state, dict) else {}
        sent = list(msg_state.get('sent', [])) if isinstance(msg_state, dict) else []
        recv = list(msg_state.get('recv', [])) if isinstance(msg_state, dict) else []
        return sent, recv

    def _sync_recent_msgs_from_memory(self, name: str) -> None:
        try:
            sent, recv = self._get_message_memory_buffers(name)
            agent_store = self.recent_msgs.setdefault(name, {})
            sent_deque = agent_store.setdefault('sent', deque(maxlen=self.message_memory_k_sent))
            recv_deque = agent_store.setdefault('recv', deque(maxlen=self.message_memory_k_recv))
            sent_deque.clear()
            recv_deque.clear()
            for item in sent[-self.message_memory_k_sent:]:
                sent_deque.append(item)
            for item in recv[-self.message_memory_k_recv:]:
                recv_deque.append(item)
        except Exception as e:
            print(f"WARNING: [MSG_MEMORY] sync failed for {name}: {e}")

    def _record_declared_goal(self, agent_name: str, msg_text: str) -> None:
        try:
            import re
            match = re.search(r"GOAL\s*:\s*\(?(\d+)\s*,\s*(\d+)\)?", msg_text, re.IGNORECASE)
            alt = re.search(r"My\s+goal\s+is\s*\(?(\d+)\s*,\s*(\d+)\)?", msg_text, re.IGNORECASE) if not match else None
            goal_match = match or alt
            if goal_match:
                self.agents[agent_name].last_goal = (int(goal_match.group(1)), int(goal_match.group(2)))
        except Exception as e:
            print(f"WARNING: [MSG_PHASE] Failed to parse GOAL: {e}")

    def _collect_message_tokens(self, agent) -> dict:
        try:
            raw = getattr(agent.brain, 'usage', {})
            if isinstance(raw, dict):
                return dict(raw)
            prompt_tokens = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
            completion_tokens = int(getattr(agent.brain, 'usage_completion', raw or 0) or 0)
            completion_tokens_raw = int(getattr(agent.brain, 'usage_completion_raw', completion_tokens) or completion_tokens)
            reasoning_tokens = int(getattr(agent.brain, 'usage_reasoning', 0) or 0)
            total_tokens = int(getattr(agent.brain, 'usage_total', prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
            message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0)
            message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0)
            return {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'completion_tokens_raw': completion_tokens_raw,
                'reasoning_tokens': reasoning_tokens,
                'total_tokens': total_tokens,
                'message_prompt_tokens': message_prompt_tokens,
                'message_completion_tokens': message_completion_tokens,
            }
        except Exception:
            return {}

    def _empty_message_tokens(self) -> dict:
        return {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'completion_tokens_raw': 0,
            'reasoning_tokens': 0,
            'total_tokens': 0,
            'message_prompt_tokens': 0,
            'message_completion_tokens': 0,
        }

    def _append_message_decision(
        self,
        name: str,
        recipient: str | None,
        prompt: str,
        response: str,
        send: bool,
        message: str,
        api_call_time: float,
        tokens: dict,
        meta: dict | None = None,
    ) -> None:
        self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
            'recipient': recipient,
            'prompt': prompt,
            'response': response,
            'send': bool(send),
            'message': str(message or ''),
            'api_call_time': api_call_time,
            'tokens': tokens,
            'meta': meta or {},
        })

    def _queue_outgoing_message(
        self,
        name: str,
        recipient: str,
        message_text: str,
        api_call_time: float,
        tokens: dict,
        prompt: str = '',
        meta: dict | None = None,
    ) -> None:
        msg_text = str(message_text or '').strip()
        if not msg_text:
            return
        msg_obj = {
            'round': self.round,
            'agent_name': name,
            'message': msg_text,
            'recipients': [recipient],
        }
        if isinstance(meta, dict):
            llm_generated_text = str(meta.get('llm_generated_text') or '').strip()
            deterministic_text = str(meta.get('deterministic_text') or '').strip()
            if llm_generated_text or deterministic_text:
                msg_obj['message_parts'] = {
                    'llm_generated_text': llm_generated_text,
                    'deterministic_text': deterministic_text,
                }
        self.messages.append(msg_obj)
        self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
            'recipient': recipient,
            'message': msg_text,
            'api_call_time': api_call_time,
            'tokens': tokens,
            'prompt': prompt,
            'meta': dict(meta or {}),
        })
        self._record_declared_goal(name, msg_text)
        self._routing_record_target(name, recipient)

    def _flush_message_phase_logs(self) -> None:
        if not getattr(self, 'message_phase_log_immediately', False):
            return
        if not hasattr(self, 'delayed_actions') or not self.delayed_actions:
            return
        print(f"DEBUG: [MSG_PHASE][LOG] Saving message-phase decisions and send summaries")
        for name, action_data in self.delayed_actions.items():
            actual_success = bool(
                (action_data.get('subactions', {}) or {}).get('message_phase', {}).get('sent')
            )
            try:
                view_now = self.get_view(name)
            except Exception:
                view_now = action_data.get('view')
            self.logger.log_agent_action(
                self,
                name,
                action_data.get('prompt', ''),
                action_data.get('response', ''),
                action_data.get('action', 'STAY'),
                action_data.get('message', ''),
                action_data.get('api_call_time', 0),
                actual_success,
                view_now,
                action_data.get('usage', {}),
                extra={
                    'messages_received': action_data.get('messages_received', []),
                    'subactions': action_data.get('subactions', {}),
                    'interaction_request': action_data.get('interaction_request'),
                    'interaction_result': action_data.get('interaction_result'),
                    'capacity_snapshot': action_data.get('capacity_snapshot'),
                    'action_trace': action_data.get('action_trace', []),
                }
            )
        self.delayed_actions = {}

    def _format_message_phase_response_for_history(
        self,
        recipient: str,
        route_response: str,
        outgoing,
        decision_meta: dict | None = None,
        protocol=None,
    ) -> str:
        formatter = getattr(protocol, 'format_history_response', None)
        if callable(formatter):
            try:
                return formatter(recipient, route_response, outgoing, dict(decision_meta or {}))
            except Exception as e:
                print(f"WARNING: [MSG_PHASE] protocol.format_history_response failed: {e}")
        route_response = str(route_response or '').strip()
        content_response = str(getattr(outgoing, 'raw_response', '') or '').strip()
        outgoing_message = str(getattr(outgoing, 'message', '') or '').strip()
        if route_response and content_response:
            return f"[ROUTING]\n{route_response}\n[CONTENT]\n{content_response}"
        if route_response:
            return route_response
        return content_response or outgoing_message

    async def _run_compat_message_phase_agent(self, name: str, agent, obs: dict) -> None:
        target_names = [candidate.agent_name for candidate in self._build_message_candidates(name, obs)]
        for target in target_names:
            prompt = ''
            response = ''
            api_call_time = 0.0
            send = False
            msg_text = ''
            try:
                prompt = agent.build_message_prompt(obs, target)
                t0 = time.time()
                response = await agent.brain.generate(prompt)
                api_call_time = time.time() - t0
                send, msg_text = self._parse_msg_decision(response)
                if hasattr(agent.brain, 'jump_back'):
                    try:
                        agent.brain.jump_back()
                    except Exception:
                        pass
            except Exception as e:
                print(f"WARNING: [MSG_PHASE][COMPAT] {name}->{target} generation/parsing error: {e}")
            tokens = self._collect_message_tokens(agent)
            self._append_message_decision(name, target, prompt, response or msg_text, send, msg_text, api_call_time, tokens, {'mode': 'compat'})
            if hasattr(agent, 'record_response') and response:
                try:
                    agent.record_response(response, 'message', obs)
                except Exception:
                    pass
            if send and msg_text:
                self._queue_outgoing_message(
                    name,
                    target,
                    msg_text,
                    api_call_time,
                    tokens,
                    prompt,
                    {'llm_generated_text': msg_text, 'deterministic_text': ''},
                )
                try:
                    if self.enable_message_memory:
                        self.recent_msgs[name]['sent'].append({'to': target, 'message': msg_text, 'round': self.round})
                except Exception:
                    pass

    async def message_phase(self):
        """Run the LLM-driven P2P message phase after each action round; environment hooks control timing."""
        try:
            if not getattr(self, 'messages_enabled', True):
                return None
            import asyncio
            print(f"DEBUG: [MSG_PHASE] === Starting message phase (Round {self.round}) ===")
            if getattr(self, '_use_mechanism_message_phase_v2', True):
                if not getattr(self, 'message_phase_enabled', False):
                    await self._message_phase_fallback()
                    print(f"DEBUG: [MSG_PHASE] === Ending message phase (Round {self.round}) ===")
                    return None

                async def process_agent_v2(name: str):
                    agent = self.agents[name]
                    obs = self.get_round_obs(agent)
                    self._ensure_message_phase_slot(name)
                    bundle = getattr(agent, 'mechanisms', None)
                    if bundle is None:
                        await self._run_compat_message_phase_agent(name, agent, obs)
                        return

                    ctx = self._make_round_context(agent, obs, phase='message')
                    try:
                        bundle.memory.observe(ctx)
                    except Exception as e:
                        print(f"WARNING: [MSG_PHASE] memory.observe failed: {name}, err={e}")
                    try:
                        await bundle.memory.before_message_routing(ctx)
                    except Exception as e:
                        print(f"WARNING: [MSG_PHASE] memory.before_message_routing failed: {name}, err={e}")
                    candidates = self._build_message_candidates(name, ctx.obs)
                    memory_ctx = bundle.memory.build_message_context(ctx)
                    routing_decision = await bundle.routing.select_targets(ctx, candidates, memory_ctx)
                    if not routing_decision.send or not routing_decision.recipients:
                        routing_meta = dict(routing_decision.meta or {})
                        if not routing_meta.get('error_reason'):
                            routing_meta['error_reason'] = 'model_send_no' if str(routing_meta.get('routing_response') or '').strip() else 'no_candidates'
                        self._append_message_decision(
                            name=name,
                            recipient=None,
                            prompt=str(routing_meta.get('routing_prompt') or ''),
                            response=str(routing_meta.get('routing_response') or ''),
                            send=False,
                            message='',
                            api_call_time=float(routing_meta.get('routing_api_call_time', 0) or 0),
                            tokens=(dict(routing_meta.get('routing_tokens')) if isinstance(routing_meta.get('routing_tokens'), dict) else self._empty_message_tokens()),
                            meta=routing_meta,
                        )
                        return

                    legal_names = {candidate.agent_name for candidate in candidates}
                    recipients = [recipient for recipient in routing_decision.recipients if recipient in legal_names]
                    if not recipients:
                        routing_meta = dict(routing_decision.meta or {})
                        routing_meta.setdefault('error_reason', 'no_legal_recipients')
                        self._append_message_decision(
                            name=name,
                            recipient=None,
                            prompt=str(routing_meta.get('routing_prompt') or ''),
                            response=str(routing_meta.get('routing_response') or ''),
                            send=False,
                            message='',
                            api_call_time=float(routing_meta.get('routing_api_call_time', 0) or 0),
                            tokens=(dict(routing_meta.get('routing_tokens')) if isinstance(routing_meta.get('routing_tokens'), dict) else self._empty_message_tokens()),
                            meta={'error': 'no_legal_recipients', **routing_meta},
                        )
                        return

                    async def compose_for_recipient(recipient: str):
                        routing_meta = dict(routing_decision.meta or {})
                        outgoing = await bundle.protocol.compose(ctx, recipient, memory_ctx, routing_meta)
                        outgoing.recipient = recipient
                        decision_meta = dict(routing_meta)
                        if isinstance(outgoing.meta, dict):
                            decision_meta.update(outgoing.meta)
                        route_response = str(routing_meta.get('routing_response') or '').strip()
                        recorded_response = self._format_message_phase_response_for_history(
                            recipient=recipient,
                            route_response=route_response,
                            outgoing=outgoing,
                            decision_meta=decision_meta,
                            protocol=bundle.protocol,
                        )
                        total_api_call_time = float(routing_meta.get('routing_api_call_time', 0) or 0) + float(outgoing.api_call_time or 0)
                        if isinstance(outgoing.tokens, dict):
                            tokens = dict(outgoing.tokens)
                        elif isinstance(routing_meta.get('routing_tokens'), dict):
                            tokens = dict(routing_meta.get('routing_tokens'))
                        else:
                            tokens = self._collect_message_tokens(agent)
                        self._append_message_decision(
                            name=name,
                            recipient=recipient,
                            prompt=outgoing.prompt,
                            response=recorded_response,
                            send=outgoing.send,
                            message=outgoing.message,
                            api_call_time=total_api_call_time,
                            tokens=tokens,
                            meta=decision_meta,
                        )
                        should_record_history_response = True
                        try:
                            should_record_history_response = bool(
                                bundle.protocol.should_record_history_response(
                                    ctx=ctx,
                                    recipient=recipient,
                                    route_response=route_response,
                                    outgoing=outgoing,
                                    decision_meta=decision_meta,
                                )
                            )
                        except Exception:
                            should_record_history_response = True
                        if hasattr(agent, 'record_response') and recorded_response and should_record_history_response:
                            try:
                                agent.record_response(recorded_response, 'message', obs)
                            except Exception:
                                pass
                        if outgoing.send and outgoing.message:
                            self._queue_outgoing_message(
                                name=name,
                                recipient=recipient,
                                message_text=outgoing.message,
                                api_call_time=total_api_call_time,
                                tokens=tokens,
                                prompt=outgoing.prompt,
                                meta=decision_meta,
                            )
                            try:
                                bundle.memory.record_outgoing(ctx, outgoing)
                                self._sync_recent_msgs_from_memory(name)
                            except Exception as e:
                                print(f"WARNING: [MSG_PHASE] record_outgoing failed: {name}->{recipient}, err={e}")

                    if self.message_phase_concurrency is None:
                        await asyncio.gather(*[compose_for_recipient(recipient) for recipient in recipients])
                    else:
                        sem = asyncio.Semaphore(self.message_phase_concurrency)

                        async def sem_task(recipient: str):
                            async with sem:
                                await compose_for_recipient(recipient)

                        await asyncio.gather(*[sem_task(recipient) for recipient in recipients])

                await asyncio.gather(*[process_agent_v2(name) for name in sorted(self.agents.keys())])
                print(f"DEBUG: [MSG_PHASE] === Ending message phase (Round {self.round}) ===")
                return None
            if not getattr(self, 'message_phase_enabled', False):
                await self._message_phase_fallback()
                print(f"DEBUG: [MSG_PHASE] === Ending message phase (Round {self.round}) ===")
                return None
            if getattr(self, 'single_message_per_round', False):
                print(f"DEBUG: [MSG_PHASE] One-message-per-round mode enabled")
                async def process_agent_single(name: str):
                    agent = self.agents[name]
                    obs = self.get_round_obs(agent)
                    # Before routing, parse last round's messages with the LLM to update unified memory for table/resource content.
                    try:
                        if hasattr(agent, 'kb_update_from_messages_via_llm'):
                            await agent.kb_update_from_messages_via_llm(obs)
                    except Exception as e:
                        print(f"WARNING: [KB][LLM] Single-message pre-parsing error: {e}")
                    targets_info = obs.get('agents_in_communication_range', {}) or {}
                    candidate_names = [t for t in sorted(targets_info.keys()) if t != name]
                    print(f"DEBUG: [MSG_PHASE][CANDIDATES][SINGLE] Agent {name} candidate targets: {candidate_names}; mode={self.single_message_prompt_mode}")
                    self._ensure_message_phase_slot(name)

                    # === Star mode: pick a global center (deterministic per round), others -> center; center -> one random leaf ===
                    if str(self.single_message_prompt_mode).lower() == 'star':
                        try:
                            import random
                            # Deterministic center by seed+round on sorted agent names
                            all_names = sorted(list(self.agents.keys()))
                            rnd = random.Random(int(getattr(self, 'seed', 0)) * 1000003 + int(getattr(self, 'round', 0)))
                            center = rnd.choice(all_names) if all_names else None
                            if center is None or name not in all_names:
                                return
                            if name == center:
                                # center selects one random other agent to send (deterministic)
                                leaves = [n for n in all_names if n != center]
                                if not leaves:
                                    return
                                parsed_to = rnd.choice(leaves)
                            else:
                                parsed_to = center

                            target_info = (obs.get('agents_in_communication_range') or {}).get(parsed_to, {})
                            prompt = agent.build_random_target_message_prompt(obs, parsed_to, target_info)
                            api_call_time = 0
                            response = ''
                            try:
                                t0 = time.time()
                                p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                response = await agent.brain.generate(prompt)
                                api_call_time = time.time() - t0
                                p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                                try:
                                    agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                                    agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                                except Exception:
                                    pass
                            except Exception as e:
                                print(f"WARNING: [STAR_MODE] Generation error: {e}")
                                response = ''
                            _, msg_only = self._parse_msg_decision(response)
                            # Response memory: record raw message-phase response (no parsed action)
                            try:
                                if hasattr(agent, 'record_response') and response:
                                    agent.record_response(response, 'message', obs)
                            except Exception:
                                pass
                            final_msg = msg_only.strip() if msg_only.strip() else str(response).strip()
                            if not final_msg:
                                final_msg = 'MSG: Hello.'
                            self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                'recipient': parsed_to,
                                'prompt': prompt,
                                'response': response,
                                'send': True,
                                'message': final_msg,
                                'api_call_time': api_call_time,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'meta': {'mode': 'star'}
                            })
                            msg_text = final_msg.strip()
                            self.messages.append({'round': self.round, 'agent_name': name, 'message': msg_text, 'recipients': [parsed_to]})
                            self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                'recipient': parsed_to,
                                'message': msg_text,
                                'api_call_time': api_call_time,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'prompt': prompt
                            })
                            self._routing_record_target(name, parsed_to)
                            return
                        except Exception as e:
                            print(f"WARNING: [STAR_MODE] Error: {e}")

                    # === Ring mode: deterministic ring successor, LLM decides message content ===
                    if str(self.single_message_prompt_mode).lower() == 'ring':
                        try:
                            ring_mode = str(getattr(self, 'ring_mode', 'angle_halves')).lower()
                            if ring_mode == 'angle_halves':
                                order = self._ring_order_angle_halves(name, obs)
                                # successor is next element
                                if name in order and len(order) > 1:
                                    idx = order.index(name)
                                    parsed_to = order[(idx + 1) % len(order)]
                                else:
                                    parsed_to = None
                            elif ring_mode == 'angle_skip':
                                order, kskip = self._ring_order_angle_skip(name, obs)
                                if name in order and len(order) > 1:
                                    idx = order.index(name)
                                    parsed_to = order[(idx + kskip) % len(order)]
                                else:
                                    parsed_to = None
                            elif ring_mode == 'carry_interleave':
                                order = self._ring_order_carry_interleave(name, obs)
                                if name in order and len(order) > 1:
                                    idx = order.index(name)
                                    parsed_to = order[(idx + 1) % len(order)]
                                else:
                                    parsed_to = None
                            elif ring_mode == 'id':
                                order = self._ring_order_id(name, obs)
                                if name in order and len(order) > 1:
                                    idx = order.index(name)
                                    parsed_to = order[(idx + 1) % len(order)]
                                else:
                                    parsed_to = None
                            elif ring_mode == 'random':
                                order = self._ring_order_random(name, obs)
                                if name in order and len(order) > 1:
                                    idx = order.index(name)
                                    parsed_to = order[(idx + 1) % len(order)]
                                else:
                                    parsed_to = None
                            else:
                                # fallback
                                order = self._ring_order_angle_halves(name, obs)
                                if name in order and len(order) > 1:
                                    idx = order.index(name)
                                    parsed_to = order[(idx + 1) % len(order)]
                                else:
                                    parsed_to = None

                            if not parsed_to:
                                # fallback: choose any candidate
                                parsed_to = (candidate_names[0] if candidate_names else None)
                            if not parsed_to:
                                # Record an empty decision when no candidates exist.
                                self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                    'recipient': None,
                                    'prompt': '',
                                    'response': '',
                                    'send': False,
                                    'message': '',
                                    'api_call_time': 0,
                                    'tokens': getattr(agent.brain, 'usage', {}),
                                    'meta': {'mode': 'ring', 'ring_mode': ring_mode}
                                })
                                return
                            target_info = (obs.get('agents_in_communication_range') or {}).get(parsed_to, {})
                            prompt = agent.build_random_target_message_prompt(obs, parsed_to, target_info)
                            api_call_time = 0
                            response = ''
                            try:
                                t0 = time.time()
                                p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                response = await agent.brain.generate(prompt)
                                api_call_time = time.time() - t0
                                p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                                try:
                                    agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                                    agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                                except Exception:
                                    pass
                            except Exception as e:
                                print(f"WARNING: [RING_MODE] Generation error: {e}")
                                response = ''
                            _, msg_only = self._parse_msg_decision(response)
                            final_msg = msg_only.strip() if msg_only.strip() else str(response).strip()
                            if not final_msg:
                                final_msg = 'MSG: Hello.'
                            self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                'recipient': parsed_to,
                                'prompt': prompt,
                                'response': response,
                                'send': True,
                                'message': final_msg,
                                'api_call_time': api_call_time,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'meta': {'mode': 'ring', 'ring_mode': ring_mode}
                            })
                            msg_text = final_msg.strip()
                            self.messages.append({'round': self.round, 'agent_name': name, 'message': msg_text, 'recipients': [parsed_to]})
                            self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                'recipient': parsed_to,
                                'message': msg_text,
                                'api_call_time': api_call_time,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'prompt': prompt
                            })
                            self._routing_record_target(name, parsed_to)
                            return
                        except Exception as e:
                            print(f"WARNING: [RING_MODE] Error: {e}")

                    # === Random mode: random recipient; Protocol V2 (if enabled) generates header; fallback to LLM ===
                    if str(self.single_message_prompt_mode).lower() == 'random':
                        try:
                            import random
                            if not candidate_names:
                                # Record an empty decision.
                                self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                    'recipient': None,
                                    'prompt': '',
                                    'response': '',
                                    'send': False,
                                    'message': '',
                                    'api_call_time': 0,
                                    'tokens': getattr(agent.brain, 'usage', {}),
                                    'meta': {'mode': 'random'}
                                })
                                return

                            parsed_to = random.choice(candidate_names)
                            api_call_time = 0
                            response = ''
                            prompt = ''
                            final_msg = ''
                            decision_meta = {'mode': 'random'}

                            # Prefer enabled Protocol V2 headers over the LLM path; optionally append free text.
                            used_proto_v2 = False
                            try:
                                if getattr(self, 'protocol_v2_enabled', False) and getattr(self, 'protocol_v2_auto', True):
                                    decision = agent.protocol_v2_message_for_target(
                                        obs, parsed_to,
                                        ask_cooldown=self.ask_cooldown_rounds,
                                        max_items=self.max_items_in_message,
                                        max_tables=self.max_tables_in_message,
                                        header_max_len=int(self.proto_v2_header_max_len)
                                    )
                                    header_msg = str(decision.get('message') or '')
                                    if header_msg:
                                        final_msg = header_msg
                                        decision_meta = decision | {'mode': 'random'} if isinstance(decision, dict) else {'mode': 'random'}
                                        used_proto_v2 = True
                                        # Optionally append free text.
                                        if self.proto_v2_free_enable:
                                            try:
                                                free_prompt = self._build_proto_v2_free_prompt(name, header_msg, obs, self.proto_v2_free_max)
                                                t1 = time.time()
                                                p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                                c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                                free_resp = await agent.brain.generate(free_prompt)
                                                api_call_time += (time.time() - t1)
                                                p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                                c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                                dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                                                try:
                                                    agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                                                    agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                                                except Exception:
                                                    pass
                                                free_line = self._sanitize_free_line(free_resp, self.proto_v2_free_max)
                                                if free_line:
                                                    if self.proto_v2_free_slot.upper() == 'REQ' and 'REQ:none' in header_msg:
                                                        final_msg = header_msg.replace('REQ:none', free_line, 1)
                                                    else:
                                                        final_msg = (header_msg + ' ' + free_line)
                                                try:
                                                    if hasattr(agent.brain, 'jump_back'):
                                                        agent.brain.jump_back()
                                                except Exception:
                                                    pass
                                            except Exception as e:
                                                print(f"WARNING: [RANDOM_MODE][V2] Free-text generation failed: {e}")
                            except Exception as e:
                                # Fall back to the LLM path if the protocol fails.
                                print(f"WARNING: [RANDOM_MODE][V2] Generation error: {e}")

                            if not used_proto_v2:
                                # Legacy path: generate a message for a random target with the LLM.
                                try:
                                    target_info = (obs.get('agents_in_communication_range') or {}).get(parsed_to, {})
                                    prompt = agent.build_random_target_message_prompt(obs, parsed_to, target_info)
                                    t0 = time.time()
                                    p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                    c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                    response = await agent.brain.generate(prompt)
                                    api_call_time = time.time() - t0
                                    p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                    c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                    dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                                    try:
                                        agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                                        agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print(f"WARNING: [RANDOM_MODE] Generation error: {e}")
                                    response = ''
                                # Parse only the MSG line; use the full text if absent.
                                _, msg_only = self._parse_msg_decision(response)
                                final_msg = msg_only.strip() if msg_only.strip() else str(response).strip()
                                if not final_msg:
                                    final_msg = 'MSG: Hello.'

                            # Record the decision and whether the protocol was used.
                            recorded_response = response
                            if used_proto_v2:
                                marker = f"[PROTO_V2_APPLIED] reason={str(decision_meta.get('reason') if isinstance(decision_meta, dict) else '')}"
                                recorded_response = (response + "\n" + marker) if response else marker

                            self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                'recipient': parsed_to,
                                'prompt': prompt,
                                'response': recorded_response,
                                'send': True,
                                'message': final_msg,
                                'api_call_time': api_call_time,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'meta': decision_meta
                            })

                            # Queue the message and record sending.
                            msg_text = (final_msg or '').strip()
                            if msg_text:
                                self.messages.append({'round': self.round, 'agent_name': name, 'message': msg_text, 'recipients': [parsed_to]})
                                self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                    'recipient': parsed_to,
                                    'message': msg_text,
                                    'api_call_time': api_call_time,
                                    'tokens': getattr(agent.brain, 'usage', {}),
                                    'prompt': prompt
                                })
                                self._routing_record_target(name, parsed_to)
                            return
                        except Exception:
                            # Fall back to the subsequent legacy path.
                            pass

                    # Build the single-message prompt and make one request.
                    prompt = agent.build_single_message_prompt(obs, candidate_names, mode=self.single_message_prompt_mode)
                    api_call_time = 0
                    response = ""
                    parsed_send = False
                    parsed_msg = ""
                    parsed_to = None
                    try:
                        t0 = time.time()
                        # Track message-phase token deltas for routing prompt
                        p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                        c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                        response = await agent.brain.generate(prompt)
                        api_call_time = time.time() - t0
                        p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                        c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                        dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                        try:
                            agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                            agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                        except Exception:
                            pass
                        print(f"DEBUG: [MSG_PHASE][RESP][SINGLE] {name}: {response[:200]}...")
                        send, m, to = self._parse_msg_decision_with_to(response)
                        parsed_send, parsed_msg, parsed_to = send, m, to
                        try:
                            if hasattr(agent.brain, 'jump_back'):
                                agent.brain.jump_back()
                                print(f"DEBUG: [MSG_PHASE][SINGLE] jump_back applied for {name}")
                        except Exception as e:
                            print(f"WARNING: [MSG_PHASE][SINGLE] jump_back error: {e}")
                    except Exception as e:
                        print(f"WARNING: [MSG_PHASE][SINGLE] {name} generation/parsing error: {e}")

                    # Protocol V2 integration: in auto mode, replace only message content, preserving SEND/TO routing decisions.
                    used_proto_v2 = False
                    v2_reason = ''
                    proto_decision = None
                    recorded_response = response

                    if getattr(self, 'protocol_v2_enabled', False) and getattr(self, 'protocol_v2_auto', True):
                        if parsed_send and parsed_to in candidate_names:
                            try:
                                proto_decision = agent.protocol_v2_message_for_target(
                                    obs, parsed_to,
                                    ask_cooldown=self.ask_cooldown_rounds,
                                    max_items=self.max_items_in_message,
                                    max_tables=self.max_tables_in_message,
                                    header_max_len=int(self.proto_v2_header_max_len)
                                )
                                header_msg = proto_decision.get('message') or ''
                                if header_msg:
                                    final_msg = header_msg
                                    if self.proto_v2_free_enable and isinstance(header_msg, str) and header_msg:
                                        try:
                                            free_prompt = self._build_proto_v2_free_prompt(name, header_msg, obs, self.proto_v2_free_max)
                                            t1 = time.time()
                                            p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                            c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                            free_resp = await agent.brain.generate(free_prompt)
                                            api_call_time += (time.time() - t1)
                                            p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                            c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                            dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                                            try:
                                                agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                                                agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                                            except Exception:
                                                pass
                                            free_line = self._sanitize_free_line(free_resp, self.proto_v2_free_max)
                                            if free_line:
                                                if self.proto_v2_free_slot.upper() == 'REQ' and 'REQ:none' in header_msg:
                                                    final_msg = header_msg.replace('REQ:none', free_line, 1)
                                                else:
                                                    final_msg = (header_msg + ' ' + free_line)
                                            try:
                                                if hasattr(agent.brain, 'jump_back'):
                                                    agent.brain.jump_back()
                                            except Exception:
                                                pass
                                        except Exception as e:
                                            print(f"WARNING: [MSG_PHASE][V2][SINGLE] Free-text generation failed: {e}")
                                    parsed_msg = final_msg
                                    used_proto_v2 = True
                                    v2_reason = proto_decision.get('reason') or ''
                            except Exception as e:
                                print(f"WARNING: [MSG_PHASE][V2][SINGLE] Generation error: {e}")

                    if used_proto_v2:
                        marker = f"[PROTO_V2_APPLIED] reason={v2_reason}"
                        msg_block = parsed_msg.strip() if isinstance(parsed_msg, str) else ''
                        recorded_response = f"SEND: yes\nTO: {parsed_to}\nMSG:\n{msg_block}\n{marker}"

                    # Record the decision.
                    try:
                        self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                            'recipient': parsed_to,
                            'prompt': prompt,
                            'response': recorded_response,
                            'send': parsed_send,
                            'message': parsed_msg,
                            'api_call_time': api_call_time,
                            'tokens': getattr(agent.brain, 'usage', {}),
                            'meta': proto_decision if used_proto_v2 else None
                        })
                        print(f"DEBUG: [MSG_PHASE][COLLECT][SINGLE] Decision: send={parsed_send}, to={parsed_to}")
                    except Exception as e:
                        print(f"WARNING: [MSG_PHASE][COLLECT][SINGLE] Cannot collect decision: {e}")

                    # Response memory: record effective message-phase response
                    try:
                        resp_to_record = recorded_response if used_proto_v2 else response
                        if hasattr(agent, 'record_response') and resp_to_record:
                            agent.record_response(resp_to_record, 'message', obs)
                    except Exception:
                        pass
                    # Validate the target and send.
                    if parsed_send and isinstance(parsed_msg, str) and parsed_msg.strip():
                        if parsed_to in candidate_names:
                            msg_text = parsed_msg.strip()
                            msg_obj = {
                                'round': self.round,
                                'agent_name': name,
                                'message': msg_text,
                                'recipients': [parsed_to]
                            }
                            self.messages.append(msg_obj)
                            print(f"DEBUG: [MSG_PHASE][QUEUE][SINGLE] Appending message: from={name} to={parsed_to} round={self.round}")
                            # Parse and record the agent's GOAL, if present, supporting both formats.
                            try:
                                import re
                                m = re.search(r"GOAL\s*:\s*\(?(\d+)\s*,\s*(\d+)\)?", msg_text, re.IGNORECASE)
                                mg = re.search(r"My\s+goal\s+is\s*\(?(\d+)\s*,\s*(\d+)\)?", msg_text, re.IGNORECASE) if not m else None
                                mm = m or mg
                                if mm:
                                    gx = int(mm.group(1)); gy = int(mm.group(2))
                                    self.agents[name].last_goal = (gx, gy)
                                    print(f"DEBUG: [MSG_PHASE][SINGLE] {name} announced GOAL=({gx},{gy})")
                            except Exception as e:
                                print(f"WARNING: [MSG_PHASE][SINGLE] Failed to parse GOAL: {e}")
                            # Update sent-message memory.
                            try:
                                if self.enable_message_memory:
                                    self.recent_msgs[name]['sent'].append({'to': parsed_to, 'message': msg_text, 'round': self.round})
                                    print(f"DEBUG: [MSG_MEMORY][SENT] {name} -> {parsed_to} recorded sent message")
                            except Exception as e:
                                print(f"WARNING: [MSG_MEMORY][SENT] Recording failed: {e}")
                            # Collect sent messages.
                            try:
                                self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                    'recipient': parsed_to,
                                    'message': msg_text,
                                    'api_call_time': api_call_time,
                                    'tokens': getattr(agent.brain, 'usage', {}),
                                    'prompt': prompt
                                })
                            except Exception:
                                pass
                            # Record routing.
                            self._routing_record_target(name, parsed_to)
                        else:
                            print(f"DEBUG: [MSG_PHASE][SINGLE] TO target invalid or outside communication range: to={parsed_to}, candidates={candidate_names}")
                    else:
                        print(f"DEBUG: [MSG_PHASE][SINGLE] {name} chose not to send or the message is empty")

                await asyncio.gather(*[process_agent_single(n) for n in sorted(self.agents.keys())])
            else:
                async def process_agent(name: str):
                    agent = self.agents[name]
                    obs = self.get_round_obs(agent)
                    # Before per-target routing, parse last round's messages with the LLM to update unified memory for table/resource content.
                    try:
                        if hasattr(agent, 'kb_update_from_messages_via_llm'):
                            await agent.kb_update_from_messages_via_llm(obs)
                    except Exception as e:
                        print(f"WARNING: [KB][LLM] Multi-target pre-parsing error: {e}")
                    targets_info = obs.get('agents_in_communication_range', {}) or {}
                    target_names = [t for t in sorted(targets_info.keys()) if t != name]
                    print(f"DEBUG: [MSG_PHASE][CANDIDATES] Agent {name} candidate targets: {target_names}")
                    self._ensure_message_phase_slot(name)

                    # === Star / Ring / Random modes for multi-target path ===
                    if str(self.single_message_prompt_mode).lower() == 'star':
                        import random
                        try:
                            all_names = sorted(list(self.agents.keys()))
                            rnd = random.Random(int(getattr(self, 'seed', 0)) * 1000003 + int(getattr(self, 'round', 0)))
                            center = rnd.choice(all_names) if all_names else None
                            if center is None or name not in all_names:
                                return
                            if name == center:
                                leaves = [n for n in all_names if n != center]
                                if not leaves:
                                    return
                                tgt = rnd.choice(leaves)
                            else:
                                tgt = center
                            final_msg = ''
                            if getattr(self, 'protocol_v2_enabled', False) and getattr(self, 'protocol_v2_auto', True):
                                try:
                                    decision = agent.protocol_v2_message_for_target(
                                        obs, tgt,
                                        ask_cooldown=self.ask_cooldown_rounds,
                                        max_items=self.max_items_in_message,
                                        max_tables=self.max_tables_in_message,
                                        header_max_len=int(self.proto_v2_header_max_len)
                                    )
                                    final_msg = str(decision.get('message') or '')
                                except Exception:
                                    final_msg = ''
                            if not final_msg:
                                pos = obs.get('position', {}) or {}
                                x, y = int(pos.get('x', 0)), int(pos.get('y', 0))
                                final_msg = f"I am {name}. My location is ({x},{y}). What’s your current status?"
                            self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                'recipient': tgt,
                                'prompt': '',
                                'response': '',
                                'send': True,
                                'message': final_msg,
                                'api_call_time': 0,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'meta': {'mode': 'star'}
                            })
                            msg_text = final_msg.strip()
                            if msg_text:
                                self.messages.append({'round': self.round, 'agent_name': name, 'message': msg_text, 'recipients': [tgt]})
                                self._routing_record_target(name, tgt)
                                self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                    'recipient': tgt,
                                    'message': msg_text,
                                    'api_call_time': 0,
                                    'tokens': getattr(agent.brain, 'usage', {})
                                })
                            return
                        except Exception as e:
                            print(f"WARNING: [STAR_MODE] Multi-target error: {e}")

                    if str(self.single_message_prompt_mode).lower() == 'random':
                        import random
                        try:
                            if not target_names:
                                self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                    'recipient': None,
                                    'prompt': '',
                                    'response': '',
                                    'send': False,
                                    'message': '',
                                    'api_call_time': 0,
                                    'tokens': getattr(agent.brain, 'usage', {}),
                                    'meta': {'mode': 'random'}
                                })
                                return
                            tgt = random.choice(target_names)
                            final_msg = ''
                            if getattr(self, 'protocol_v2_enabled', False) and getattr(self, 'protocol_v2_auto', True):
                                try:
                                    decision = agent.protocol_v2_message_for_target(
                                        obs, tgt,
                                        ask_cooldown=self.ask_cooldown_rounds,
                                        max_items=self.max_items_in_message,
                                        max_tables=self.max_tables_in_message,
                                        header_max_len=int(self.proto_v2_header_max_len)
                                    )
                                    final_msg = str(decision.get('message') or '')
                                except Exception:
                                    final_msg = ''
                            if not final_msg:
                                pos = obs.get('position', {}) or {}
                                x, y = int(pos.get('x', 0)), int(pos.get('y', 0))
                                final_msg = f"I am {name}. My location is ({x},{y}). What’s your current status?"
                            # Decide and enqueue.
                            self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                'recipient': tgt,
                                'prompt': '',
                                'response': '',
                                'send': True,
                                'message': final_msg,
                                'api_call_time': 0,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'meta': {'mode': 'random'}
                            })
                            msg_text = final_msg.strip()
                            if msg_text:
                                self.messages.append({'round': self.round, 'agent_name': name, 'message': msg_text, 'recipients': [tgt]})
                                self._routing_record_target(name, tgt)
                                self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                    'recipient': tgt,
                                    'message': msg_text,
                                    'api_call_time': 0,
                                    'tokens': getattr(agent.brain, 'usage', {})
                                })
                            return
                        except Exception as e:
                            print(f"WARNING: [RANDOM_MODE] Random routing error: {e}")
                            # Fall back to the legacy branch below.

                    # Generate Protocol V2 or legacy fallback messages for each target.
                    async def decide_for_target(tgt: str):
                        api_call_time = 0
                        parsed_send = False
                        parsed_msg = ""
                        prompt = ''
                        response = ''
                        decision_meta = None
                        if getattr(self, 'protocol_v2_enabled', False) and getattr(self, 'protocol_v2_auto', True):
                            # New protocol: generate with a function, without calling the LLM.
                            try:
                                decision = agent.protocol_v2_message_for_target(
                                    obs, tgt,
                                    ask_cooldown=self.ask_cooldown_rounds,
                                    max_items=self.max_items_in_message,
                                    max_tables=self.max_tables_in_message,
                                    header_max_len=int(self.proto_v2_header_max_len)
                                )
                                parsed_send = bool(decision.get('send'))
                                header_msg = (decision.get('message') or '')
                                # Generate and append free text.
                                final_msg = header_msg
                                if parsed_send and self.proto_v2_free_enable and isinstance(header_msg, str) and header_msg:
                                    try:
                                        free_prompt = self._build_proto_v2_free_prompt(name, header_msg, obs, self.proto_v2_free_max)
                                        t1 = time.time()
                                        # Token deltas for free sentence generation
                                        p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                        c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                        free_resp = await agent.brain.generate(free_prompt)
                                        api_call_time += (time.time() - t1)
                                        p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                        c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                        dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                                        try:
                                            agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                                            agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                                        except Exception:
                                            pass
                                        free_line = self._sanitize_free_line(free_resp, self.proto_v2_free_max)
                                        if free_line:
                                            if self.proto_v2_free_slot.upper() == 'REQ' and 'REQ:none' in header_msg:
                                                final_msg = header_msg.replace('REQ:none', free_line, 1)
                                            else:
                                                final_msg = (header_msg + ' ' + free_line)
                                            # Keep the complete free-text sentence without truncation.
                                            # The full message may exceed the configured total length limit.
                                        try:
                                            if hasattr(agent.brain, 'jump_back'):
                                                agent.brain.jump_back()
                                        except Exception:
                                            pass
                                    except Exception as e:
                                        print(f"WARNING: [MSG_PHASE][V2] Free-text generation failed: {e}")
                                parsed_msg = final_msg
                                reason = decision.get('reason')
                                decision_meta = decision
                                print(f"DEBUG: [MSG_PHASE][V2] {name}->{tgt} send={parsed_send} reason={reason}")
                            except Exception as e:
                                print(f"WARNING: [MSG_PHASE][V2] Generation error: {e}")
                        else:
                            # Retain the legacy LLM path as fallback.
                            try:
                                prompt = agent.build_message_prompt(obs, tgt)
                                t0 = time.time()
                                p0 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                c0 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                response = await agent.brain.generate(prompt)
                                api_call_time = time.time() - t0
                                p1 = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                                c1 = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                                dp, dc = max(0, p1 - p0), max(0, c1 - c0)
                                try:
                                    agent.brain.message_prompt_tokens = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0) + dp
                                    agent.brain.message_completion_tokens = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0) + dc
                                except Exception:
                                    pass
                                print(f"DEBUG: [MSG_PHASE][RESP] {name} -> {tgt}: {response[:200]}...")
                                # Parse.
                                send, m = self._parse_msg_decision(response)
                                parsed_send = send
                                parsed_msg = m
                                # Do not persist message-phase history; roll back user and assistant messages.
                                try:
                                    if hasattr(agent.brain, 'jump_back'):
                                        agent.brain.jump_back()
                                        print(f"DEBUG: [MSG_PHASE] jump_back applied for {name}->{tgt}")
                                except Exception as e:
                                    print(f"WARNING: [MSG_PHASE] jump_back error: {e}")
                            except Exception as e:
                                print(f"WARNING: [MSG_PHASE] {name}->{tgt} generation/parsing error: {e}")
                        # Collect decisions into sub-actions.
                        try:
                            self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                                'recipient': tgt,
                                'prompt': prompt,
                                'response': response or parsed_msg,
                                'send': parsed_send,
                                'message': parsed_msg,
                                'api_call_time': api_call_time,
                                'tokens': getattr(agent.brain, 'usage', {}),
                                'meta': decision_meta
                            })
                            print(f"DEBUG: [MSG_PHASE][COLLECT] Collected decision: {name}->{tgt} send={parsed_send}")
                        except Exception as e:
                            print(f"WARNING: [MSG_PHASE][COLLECT] Cannot collect decision: {e}")

                        # Send queue and records.
                        if parsed_send and isinstance(parsed_msg, str) and parsed_msg.strip():
                            msg_text = parsed_msg.strip()
                            msg_obj = {
                                'round': self.round,  # Sent this round, received next round.
                                'agent_name': name,
                                'message': msg_text,
                                'recipients': [tgt]
                            }
                            self.messages.append(msg_obj)
                            print(f"DEBUG: [MSG_PHASE][QUEUE] Appending message: from={name} to={tgt} len={len(msg_text)} round={self.round}")
                            self._routing_record_target(name, tgt)
                            # Parse and record the agent's GOAL, if present, supporting both formats.
                            try:
                                import re
                                m = re.search(r"GOAL\s*:\s*\(?(\d+)\s*,\s*(\d+)\)?", msg_text, re.IGNORECASE)
                                mg = re.search(r"My\s+goal\s+is\s*\(?(\d+)\s*,\s*(\d+)\)?", msg_text, re.IGNORECASE) if not m else None
                                mm = m or mg
                                if mm:
                                    gx = int(mm.group(1)); gy = int(mm.group(2))
                                    self.agents[name].last_goal = (gx, gy)
                                    print(f"DEBUG: [MSG_PHASE] {name} announced GOAL=({gx},{gy})")
                            except Exception as e:
                                print(f"WARNING: [MSG_PHASE] Failed to parse GOAL: {e}")
                            # Update sent-message memory.
                            try:
                                if self.enable_message_memory:
                                    self.recent_msgs[name]['sent'].append({'to': tgt, 'message': msg_text, 'round': self.round})
                                    print(f"DEBUG: [MSG_MEMORY][SENT] {name} -> {tgt} recorded sent message")
                            except Exception as e:
                                print(f"WARNING: [MSG_MEMORY][SENT] Recording failed: {e}")
                            # Collect successful sends into sub-actions.
                            try:
                                self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                    'recipient': tgt,
                                    'message': msg_text,
                                    'api_call_time': api_call_time,
                                    'tokens': getattr(agent.brain, 'usage', {})
                                })
                                print(f"DEBUG: [MSG_PHASE][COLLECT] Collected send: {name}->{tgt}")
                            except Exception as e:
                                print(f"WARNING: [MSG_PHASE][COLLECT] Cannot collect send: {e}")
                            self._routing_record_target(name, tgt)

                    # Concurrency control.
                    if self.message_phase_concurrency is None:
                        await asyncio.gather(*[decide_for_target(t) for t in target_names])
                    else:
                        sem = asyncio.Semaphore(self.message_phase_concurrency)
                        async def sem_task(t):
                            async with sem:
                                await decide_for_target(t)
                        await asyncio.gather(*[sem_task(t) for t in target_names])

                # Process agents sequentially; agent-level concurrency is also possible.
                await asyncio.gather(*[process_agent(n) for n in sorted(self.agents.keys())])
            print(f"DEBUG: [MSG_PHASE] === Ending message phase (Round {self.round}) ===")
        except Exception as e:
            print(f"ERROR: [MSG_PHASE] Error: {e}")
        finally:
            try:
                self._flush_message_phase_logs()
            except Exception as e:
                print(f"WARNING: [MSG_PHASE][LOG] Failed to write log: {e}")
        return None

    async def _message_phase_fallback(self):
        """Legacy decide_p2p_for_target path: preserve the interface with pre-update observations and unified logging semantics."""
        try:
            if getattr(self, '_use_mechanism_message_phase_v2', True):
                print(f"DEBUG: [P2P_MSG] === Starting pre-update message phase (legacy path) (Round {self.round}) ===")
                for name in sorted(self.agents.keys()):
                    agent = self.agents[name]
                    obs = self.get_round_obs(agent)
                    target_names = [candidate.agent_name for candidate in self._build_message_candidates(name, obs)]
                    print(f"DEBUG: [P2P_MSG] Agent {name} candidate targets: {target_names}")
                    self._ensure_message_phase_slot(name)
                    bundle = getattr(agent, 'mechanisms', None)
                    ctx = self._make_round_context(agent, obs, phase='message_fallback')
                    for target in target_names:
                        send = False
                        msg_text = ""
                        if hasattr(agent, 'decide_p2p_for_target'):
                            try:
                                send, msg_text = agent.decide_p2p_for_target(obs, target)
                            except Exception as e:
                                print(f"DEBUG: [P2P_MSG] Agent {name} decide_p2p_for_target error: {e}")
                                send, msg_text = False, ""
                        msg_clean = msg_text.strip() if isinstance(msg_text, str) else ""
                        self._append_message_decision(
                            name=name,
                            recipient=target,
                            prompt='',
                            response=msg_clean,
                            send=bool(send and msg_clean),
                            message=msg_clean,
                            api_call_time=0,
                            tokens=self._collect_message_tokens(agent),
                            meta={'mode': 'fallback'},
                        )
                        if send and msg_clean:
                            self._queue_outgoing_message(
                                name,
                                target,
                                msg_clean,
                                0,
                                self._collect_message_tokens(agent),
                                meta={'llm_generated_text': msg_clean, 'deterministic_text': ''},
                            )
                            try:
                                if bundle is not None:
                                    bundle.memory.record_outgoing(
                                        ctx,
                                        OutgoingMessage(
                                            recipient=target,
                                            send=True,
                                            message=msg_clean,
                                            api_call_time=0,
                                            tokens=self._collect_message_tokens(agent),
                                        ),
                                    )
                                    self._sync_recent_msgs_from_memory(name)
                                elif self.enable_message_memory:
                                    self.recent_msgs[name]['sent'].append({'to': target, 'message': msg_clean, 'round': self.round})
                            except Exception as e:
                                print(f"WARNING: [MSG_MEMORY][SENT] Recording failed: {e}")
                        else:
                            print(f"DEBUG: [P2P_MSG] Agent {name} -> {target} not sending")
                print(f"DEBUG: [P2P_MSG] === Ending pre-update message phase (legacy path) (Round {self.round}) ===")
                return None
            print(f"DEBUG: [P2P_MSG] === Starting pre-update message phase (legacy path) (Round {self.round}) ===")
            for name in sorted(self.agents.keys()):
                agent = self.agents[name]
                obs = self.get_round_obs(agent)
                targets_info = obs.get('agents_in_communication_range', {}) or {}
                target_names = [t for t in sorted(targets_info.keys()) if t != name]
                print(f"DEBUG: [P2P_MSG] Agent {name} candidate targets: {target_names}")
                self._ensure_message_phase_slot(name)
                for target in target_names:
                    send = False
                    msg_text = ""
                    if hasattr(agent, 'decide_p2p_for_target'):
                        try:
                            send, msg_text = agent.decide_p2p_for_target(obs, target)
                        except Exception as e:
                            print(f"DEBUG: [P2P_MSG] Agent {name} decide_p2p_for_target error: {e}")
                            send, msg_text = False, ""
                    else:
                        print(f"DEBUG: [P2P_MSG] Agent {name} has no decide_p2p_for_target; skipping")

                    msg_clean = msg_text.strip() if isinstance(msg_text, str) else ""
                    try:
                        self.delayed_actions[name]['subactions']['message_phase']['decisions'].append({
                            'recipient': target,
                            'prompt': '',
                            'response': msg_clean,
                            'send': bool(send and msg_clean),
                            'message': msg_clean,
                            'api_call_time': 0,
                            'tokens': getattr(agent.brain, 'usage', {}),
                            'meta': {'mode': 'fallback'}
                        })
                    except Exception as e:
                        print(f"WARNING: [P2P_MSG][COLLECT] Cannot collect decision: {e}")

                    if send and msg_clean:
                        msg_obj = {
                            'round': self.round,
                            'agent_name': name,
                            'message': msg_clean,
                            'recipients': [target]
                        }
                        self.messages.append(msg_obj)
                        print(f"DEBUG: [P2P_MSG] Agent {name} -> {target} sent: '{msg_clean}' (round={self.round})")
                        try:
                            if self.enable_message_memory:
                                self.recent_msgs[name]['sent'].append({'to': target, 'message': msg_clean, 'round': self.round})
                        except Exception as e:
                            print(f"WARNING: [MSG_MEMORY][SENT] Recording failed: {e}")
                        try:
                            self.delayed_actions[name]['subactions']['message_phase']['sent'].append({
                                'recipient': target,
                                'message': msg_clean,
                                'api_call_time': 0,
                                'tokens': getattr(agent.brain, 'usage', {})
                            })
                        except Exception as e:
                            print(f"WARNING: [P2P_MSG][COLLECT] Cannot collect send: {e}")
                        self._routing_record_target(name, target)
                    else:
                        print(f"DEBUG: [P2P_MSG] Agent {name} -> {target} not sending")
            print(f"DEBUG: [P2P_MSG] === Ending pre-update message phase (legacy path) (Round {self.round}) ===")
        except Exception as e:
            print(f"DEBUG: [P2P_MSG] Message phase error: {e}")

    def _parse_msg_decision(self, response: str):
        """Parse SEND: yes|no and MSG: text from message-phase model output."""
        try:
            send = False
            msg = ""
            lines = response.splitlines()
            for line in lines:
                low = line.strip().lower()
                if low.startswith('send:'):
                    val = low.split(':', 1)[1].strip()
                    send = val in ('yes', 'y', 'true', '1')
                elif low.startswith('msg:'):
                    msg = line.split(':', 1)[1].strip()
            return send, msg
        except Exception:
            return False, ""

    def _parse_msg_decision_with_to(self, response: str):
        """Parse SEND, TO, and MSG from single-message output."""
        try:
            send = False
            msg = ""
            to = None
            lines = response.splitlines()
            for line in lines:
                raw = line.strip()
                low = raw.lower()
                if low.startswith('send:'):
                    val = low.split(':', 1)[1].strip()
                    send = val in ('yes', 'y', 'true', '1')
                elif low.startswith('to:'):
                    to = raw.split(':', 1)[1].strip()
                elif low.startswith('msg:'):
                    msg = raw.split(':', 1)[1].strip()
            return send, msg, to
        except Exception:
            return False, "", None

    def _build_proto_v2_free_prompt(self, agent_name: str, header_msg: str, obs: dict, max_chars: int) -> str:
        """Build a free-text prompt asking the LLM for one natural-language sentence, without truncation.
        Requirements:
          - Output one short natural-language sentence on one line, without PLAN/GOAL/REQ/C markers.
          - Express a plan, goal, or brief request/comment.
          - Do not repeat R:/T: coordinates or quantities from the header verbatim; references to the target table are allowed.
          - Output only the sentence, without SEND/TO/MSG prefixes.
        """
        try:
            pos = obs.get('position', {}) or {}
            x, y = int(pos.get('x', 0)), int(pos.get('y', 0))
        except Exception:
            x, y = 0, 0
        lines = [
            f"You are {agent_name}. Write ONE short sentence (single line) in natural language.",
            "Express your plan, goal, or a brief request/comment in plain words.",
            "Do NOT repeat coordinates/quantities from header R:/T: verbatim; you may reference the goal table succinctly.",
            "Do not add prefixes like SEND/TO/MSG/PLAN/GOAL/REQ/C.",
            f"My position: {x},{y}",
            f"Header: {header_msg}",
        ]
        return "\n".join(lines)

    def _sanitize_free_line(self, text: str, max_chars: int) -> str:
        """Clean free-text model output: take the first line, strip prefixes, and retain its full length."""
        if not isinstance(text, str):
            return ''
        line = text.splitlines()[0] if '\n' in text else text
        # Remove prefixes.
        for pfx in ('SEND:', 'send:', 'TO:', 'to:', 'MSG:', 'msg:'):
            if line.strip().startswith(pfx):
                try:
                    line = line.split(':', 1)[1].strip()
                except Exception:
                    pass
        line = ' '.join(line.strip().split())  # Collapse repeated whitespace.
        return line

    def _reset_with_custom_map(self, custom_map: dict):
        """Build the environment from custom map JSON, overriding random task generation."""
        try:
            from masbench.enhanced_terrain import create_terrain
            print(f"DEBUG: [_reset_with_custom_map] Input map keys: {list(custom_map.keys())}")
            map_cfg = custom_map.get('map', {})
            task_cfg = custom_map.get('task', {})

            # 1) Read dimensions and agent configuration.
            w = int(map_cfg.get('width', self.width))
            h = int(map_cfg.get('height', self.height))
            agent_cfg = map_cfg.get('agent_config', {})
            print(f"DEBUG: [_reset_with_custom_map] Target dimensions: width={w}, height={h}")

            # Allow observation/communication range overrides.
            if isinstance(agent_cfg, dict):
                vs = agent_cfg.get('view_size')
                cs = agent_cfg.get('communication_size')
                if isinstance(vs, int):
                    self.view_size = vs
                    print(f"DEBUG: [_reset_with_custom_map] Override view_size={self.view_size}")
                if isinstance(cs, int):
                    self.communication_size = cs
                    print(f"DEBUG: [_reset_with_custom_map] Override communication_size={self.communication_size}")

            # Update dimensions and reset the grid.
            self.width, self.height = w, h
            self.grid = np.full((self.height, self.width), '.', dtype=str)
            self.mesh_map = np.full_like(self.grid, None, dtype=object)
            self.meshes = []
            self.agent_meshes = {}

            # 2) Boundary walls.
            print("DEBUG: [_reset_with_custom_map] Building boundary walls")
            wall_shape = np.full((self.height, self.width), 'W', dtype=str)
            wall_shape[1:-1, 1:-1] = '.'
            wall_mesh = create_terrain('wall', (0, 0), shape=wall_shape)
            wall_mesh.place(self.grid, self.mesh_map)
            self.meshes.append(wall_mesh)

            # 3) Place elements.
            elements = map_cfg.get('elements', []) or []
            print(f"DEBUG: [_reset_with_custom_map] Placing {len(elements)} elements")
            for idx, el in enumerate(elements):
                etype = el.get('type')
                pos = el.get('pos')
                cap = el.get('capacity')
                if not (isinstance(pos, list) and len(pos) == 2):
                    print(f"WARNING: [_reset_with_custom_map] Invalid pos for element {idx}: {pos}")
                    continue
                # JSON pos uses 1-based mathematical (x,y); convert to internal grid indices (y_img, x_img).
                x_math, y_math = int(pos[0]), int(pos[1])
                y, x = (h - y_math), (x_math - 1)
                print(f"DEBUG: [_reset_with_custom_map] Placing element idx={idx} type={etype} at (x={x_math},y={y_math}) -> grid(y={y},x={x})")

                # Treat nest as table.
                if etype == 'nest':
                    etype = 'table'

                if etype in ('mountain', 'river', 'grass', 'forest', 'table', 'wall'):
                    kwargs = {}
                    if cap is not None:
                        kwargs['capacity'] = int(cap)
                    # Parse per-type capacities (tables only).
                    if etype == 'table':
                        try:
                            type_caps = el.get('type_caps') or None
                            if isinstance(type_caps, dict):
                                kwargs['type_caps'] = {str(k): int(v) for k, v in type_caps.items()}
                            ptd = el.get('per_type_default', None)
                            if ptd is not None:
                                kwargs['per_type_default'] = int(ptd)
                        except Exception as e:
                            print(f"WARNING: [_reset_with_custom_map] Failed to parse table capacities by type: {e}")
                    terrain = create_terrain(etype, (y, x), **kwargs)
                    terrain.place(self.grid, self.mesh_map)
                    self.meshes.append(terrain)
                elif etype in ('wood', 'sand', 'stone'):
                    qty = int(cap) if cap is not None else 1
                    # Ensure storage: create resource terrain if absent (symbol r in mesh_map), expanding capacity for initial items.
                    if self.mesh_map[y, x] is None:
                        base = create_terrain('resource', (y, x))
                        # Increase insufficient default capacity to accommodate the initial quantity.
                        try:
                            if hasattr(base, 'capacity') and int(base.capacity) < qty:
                                base.capacity = int(qty)
                        except Exception:
                            pass
                        base.place(self.grid, self.mesh_map)
                        self.meshes.append(base)
                    else:
                        # Existing terrain: increase capacity if it supports storage but lacks space.
                        try:
                            terr = self.mesh_map[y, x]
                            if hasattr(terr, 'can_store_items') and terr.can_store_items and hasattr(terr, 'capacity'):
                                if int(terr.capacity) < qty:
                                    terr.capacity = int(qty)
                        except Exception:
                            pass
                    # Place initial items once sufficient capacity is ensured.
                    self.add_item_at_position((y, x), etype, qty)
                elif etype == 'block':
                    # Represent obstacle B with Mesh.
                    block_shape = np.array([['B']], dtype=str)
                    block_mesh = Mesh(pos=(y, x), shape=block_shape, static=True, name='B')
                    block_mesh.place(self.grid, self.mesh_map)
                    self.meshes.append(block_mesh)
                else:
                    print(f"WARNING: [_reset_with_custom_map] Unknown element type: {etype} at (x={x_math},y={y_math}); skipped")

            # 4) Place agents.
            agents_in_map = map_cfg.get('agents', []) or []
            print(f"DEBUG: [_reset_with_custom_map] Placing {len(agents_in_map)} agents")
            # Sort available environment agent names by key for deterministic assignment.
            remaining_env_agents = [n for n in sorted(self.agents.keys())]
            assigned_env_agents = set()
            for a in agents_in_map:
                pos = a.get('pos')
                json_name = a.get('name')
                if not (isinstance(pos, list) and len(pos) == 2):
                    print(f"WARNING: [_reset_with_custom_map] Invalid agent position: {pos} (json_name={json_name})")
                    continue
                # Prefer a matching environment agent name; otherwise use the next available name.
                if json_name in self.agents and json_name not in assigned_env_agents:
                    env_name = json_name
                else:
                    env_name = None
                    while remaining_env_agents:
                        cand = remaining_env_agents.pop(0)
                        if cand not in assigned_env_agents:
                            env_name = cand
                            break
                    if env_name is None:
                        print(f"WARNING: [_reset_with_custom_map] Not enough environment agents to place another agent (json_name={json_name})")
                        continue
                assigned_env_agents.add(env_name)

                # JSON pos uses 1-based mathematical (x,y); convert to internal grid indices (y_img, x_img).
                x_math, y_math = int(pos[0]), int(pos[1])
                y, x = (h - y_math), (x_math - 1)
                # Place lowercase 'a' for agents carrying nothing.
                agent_mesh = Mesh((y, x), np.array([['a']], dtype=str), static=False, name=env_name)
                agent_mesh.place(self.grid, self.mesh_map)
                self.agent_meshes[env_name] = agent_mesh
                self.meshes.append(agent_mesh)
                print(f"DEBUG: [_reset_with_custom_map] Agent {env_name} (from json '{json_name}') placed at (x={x_math},y={y_math}) -> grid(y={y},x={x})")

            # 5) Task-specific settings (Foraging).
            if self.task == 'Foraging':
                task_obj = self.tasks[self.task]
                # Target items (scoring thresholds) come from task.scoring.
                scoring = task_cfg.get('scoring', {}) or {}
                if isinstance(scoring, dict) and scoring:
                    # Convert to int.
                    converted = {}
                    for k, v in scoring.items():
                        try:
                            converted[k] = int(v)
                        except Exception:
                            print(f"WARNING: [_reset_with_custom_map] Invalid scoring value for {k}: {v}")
                    if converted:
                        task_obj.target_items = converted
                        print(f"DEBUG: [_reset_with_custom_map] Override target items: {task_obj.target_items}")
                # Initialize resource state.
                task_obj.resource_state = {name: None for name in self.agents}
                print(f"DEBUG: [_reset_with_custom_map] Initialized Foraging resource state for {len(task_obj.resource_state)} agents")

            # 6) Integrate terrain, keeping the two-layer architecture consistent with terrain_adapter.
            print("DEBUG: [_reset_with_custom_map] Integrating enhanced terrain system")
            integrate_terrain_system(self)

            # 7) Synchronize legacy position_items to enhanced terrain so rendering displays w/s/S.
            try:
                sync_count = 0
                for pos, items in list(self.position_items.items()):
                    for it in list(items):
                        if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                            terrain = self.terrain_adapter.get_terrain_at(pos)
                            if terrain and hasattr(terrain, 'add_item'):
                                ok = terrain.add_item(it.item_type, it.quantity)
                                print(f"DEBUG: [_reset_with_custom_map] Synchronizing items to terrain: pos={pos}, {it.item_type}x{it.quantity}, ok={ok}")
                                sync_count += 1
                print(f"DEBUG: [_reset_with_custom_map] Item synchronization complete: {sync_count} entries")
            except Exception as e:
                print(f"WARNING: [_reset_with_custom_map] Item synchronization error: {e}")

        except Exception as e:
            print(f"ERROR: [_reset_with_custom_map] Custom map construction error: {e}")
            raise

    def is_done(self):
        if not self.done:
            task_done = self.tasks[self.task].is_done(self)
            if task_done or self.round >= self.max_round:
                self.done = True
                # Stop output redirection when the experiment ends.
                self.cleanup_output_logging()
                # Write key information to meta_log.json at completion.
                try:
                    if hasattr(self.logger, 'finalize_meta') and callable(self.logger.finalize_meta):
                        self.logger.finalize_meta(self)
                except Exception as e:
                    print(f"WARNING: [META] Failed to write completion summary: {e}")
                return True
            return False
        return True

    def cleanup_output_logging(self):
        """Clean up output redirection."""
        if self.output_manager and self.output_manager.is_redirected:
            print(f"DEBUG: [OUTPUT_LOGGER] Experiment ended; stopping output redirection")
            self.output_manager.stop_logging()
            if self.log_file_path:
                print(f"DEBUG: [OUTPUT_LOGGER] Terminal output saved to: {self.log_file_path}")

    def __del__(self):
        """Ensure output redirection is cleaned up during destruction."""
        try:
            self.cleanup_output_logging()
        except Exception:
            pass  # Suppress exceptions in the destructor.

    def obs(self, agent):
        if agent is None:
            # return global state
            return {
                'task': self.tasks[self.task],
                'round': self.round,
                'grid': self.grid.copy(),
                **self.tasks[self.task].task_obs(None)
            }

        name = agent.name
        view = self.get_view(name)
        x_img, y_img = self.agent_meshes[name].pos[1], self.agent_meshes[name].pos[0]
        agent_pos = (y_img, x_img)
        # Use mathematical coordinates consistently for external distances and display.
        x, y = to_math_xy(x_img, y_img, self.height)
        # get all visible messages
        visible_messages = self.get_agent_visible_messages(name)

        # get capacity information for visible CapacityMesh objects
        capacity_info = self.get_agent_capacity_info(name)
        # simplified tables info for agents: only position and remaining_capacity
        tables_info = self.get_visible_tables(name)

        # Get agent positions within communication range.
        agents_in_communication_range = self.get_agent_positions_in_communication_range(name)
        print(f"DEBUG: obs - {name} agents within communication range: {len(agents_in_communication_range)}")

        # Explicit interaction: current-cell items and remaining placement capacity, using only the new terrain system.
        tile_items = self._get_items_at_position(agent_pos)
        tile_free_capacity = 0
        tile_remaining_by_type = None
        try:
            terrain = self.terrain_adapter.get_terrain_at(agent_pos) if hasattr(self, 'terrain_adapter') and self.terrain_adapter else None
            if terrain and getattr(terrain, 'can_store_items', False):
                cur_total = int(terrain.get_items_total_quantity()) if hasattr(terrain, 'get_items_total_quantity') else 0
                cap_here = int(getattr(terrain, 'capacity', 0) or 0)
                tile_free_capacity = max(0, cap_here - cur_total)
            # Tables expose only remaining capacity, not their items.
            if terrain and getattr(terrain, 'terrain_type', '') == 'table':
                tile_items = {}
                # Include the current cell's per-type remaining capacity when supported.
                try:
                    if hasattr(terrain, 'get_remaining_by_type') and (getattr(terrain, 'capacity_by_type', None) or getattr(terrain, 'per_type_default', None) is not None):
                        tile_remaining_by_type = terrain.get_remaining_by_type(['wood', 'sand', 'stone'])
                except Exception:
                    pass
        except Exception as e:
            print(f"WARNING: obs failed to calculate tile_free_capacity: {e}")

        # Previous-round action feedback, if available.
        last_feedback = None
        try:
            if hasattr(self, 'last_action_feedback'):
                last_feedback = self.last_action_feedback.get(name)
        except Exception:
            last_feedback = None

        # Build intuitive inventory state when enhanced inventory is enabled.
        inventory_info = None
        try:
            agent_obj = self.agents.get(name)
            if agent_obj is not None and hasattr(agent_obj, 'enable_inventory') and agent_obj.enable_inventory:
                inv_type = agent_obj.get_carrying_item_type()
                inv_qty = agent_obj.get_total_carrying()
                inv_cap = getattr(agent_obj, 'carrying_capacity', None)
                inv_disp = agent_obj.get_inventory_display()
                inventory_info = {
                    'type': inv_type,
                    'quantity': inv_qty,
                    'capacity': inv_cap,
                    'display': inv_disp
                }
        except Exception as e:
            print(f"WARNING: obs failed to build enhanced inventory information: {e}")

        # Optionally provide nearest tables/items from the agent's unified KB memory.
        kb_nearest_tables = []
        kb_nearest_items = []
        try:
            agent_obj = self.agents.get(name)
            if agent_obj is not None and hasattr(agent_obj, 'known_map') and isinstance(agent_obj.known_map, dict):
                def _ed(p):
                    dx = int(p[0]) - x
                    dy = int(p[1]) - y
                    return (dx*dx + dy*dy) ** 0.5
                try:
                    kb_ttl = max(0, int(getattr(agent_obj, 'kb_ttl_rounds', 0) or 0))
                except Exception:
                    kb_ttl = 0
                try:
                    now_round = int(getattr(self, 'round', 0) or 0)
                except Exception:
                    now_round = 0

                def _kb_last_round(rec):
                    last_round = None
                    for field in ('last_view_round', 'last_msg_round', 'last_env_round', 'last_updated_round', 'first_seen_round'):
                        value = rec.get(field) if isinstance(rec, dict) else None
                        if value is None:
                            continue
                        try:
                            value_i = int(value)
                        except Exception:
                            continue
                        if last_round is None or value_i > last_round:
                            last_round = value_i
                    return last_round

                tables = []
                items_list = []
                for (kx, ky), rec in agent_obj.known_map.items():
                    if kb_ttl > 0:
                        last_round = _kb_last_round(rec)
                        if last_round is not None and now_round - int(last_round) > kb_ttl:
                            continue
                    ter = str(rec.get('terrain_type') or '')
                    if ter == 'table':
                        rem = rec.get('remaining_capacity')
                        entry = {'position': {'x': int(kx), 'y': int(ky)}, 'remaining_capacity': rem, 'distance': _ed((kx, ky))}
                        try:
                            rb = rec.get('remaining_by_type')
                            if isinstance(rb, dict) and rb:
                                entry['remaining_by_type'] = rb
                        except Exception:
                            pass
                        tables.append(entry)
                        # Exclude items on tables from nearest-KB-item hints.
                        continue
                    its = rec.get('items') or {}
                    if its:
                        items_list.append({'position': {'x': int(kx), 'y': int(ky)}, 'items': its, 'distance': _ed((kx, ky))})
                tables.sort(key=lambda e: e['distance'])
                items_list.sort(key=lambda e: e['distance'])
                try:
                    tables_k = max(0, int(getattr(agent_obj, 'kb_nearest_tables_k', 2) or 0))
                except Exception:
                    tables_k = 2
                try:
                    items_k = max(0, int(getattr(agent_obj, 'kb_nearest_items_k', 3) or 0))
                except Exception:
                    items_k = 3
                kb_nearest_tables = tables[:tables_k] if tables_k else []
                kb_nearest_items = items_list[:items_k] if items_k else []
        except Exception:
            kb_nearest_tables = []
            kb_nearest_items = []

        # Individual contribution only: report this agent's cumulative deliveries to tables.
        my_contrib = {}
        try:
            my_contrib = dict(getattr(self, 'agent_contributions', {}).get(name, {}))
        except Exception:
            my_contrib = {}
        recent_sent, recent_recv = self._get_message_memory_buffers(name)

        return {
            'task': self.tasks[self.task],
            'round': self.round,
            'view': view,
            'num_agents': len(self.agents),
            'map_width': self.width,
            'map_height': self.height,
            'position': {'x': x, 'y': y},
            'messages': visible_messages,
            'name': name,
            'capacity_info': capacity_info,
            'agents_in_communication_range': agents_in_communication_range,  # Additional field
            'explicit_interaction': bool(getattr(self, 'explicit_interaction_enabled', False)),
            'tile_items_counts': tile_items,
            'tile_placeable_free_capacity': tile_free_capacity,
            'tile_remaining_by_type': tile_remaining_by_type,
            'tile_terrain_type': (self.terrain_adapter.get_terrain_at((y_img, x_img)).terrain_type if hasattr(self, 'terrain_adapter') and self.terrain_adapter and self.terrain_adapter.get_terrain_at((y_img, x_img)) else 'unknown'),
            'last_action_feedback': last_feedback,
            'inventory': inventory_info,
            'tables_info': tables_info,
            'kb_nearest_tables': kb_nearest_tables,
            'kb_nearest_items': kb_nearest_items,
            # Expose only this agent's contribution, without omniscient information.
            'my_contribution': my_contrib,
            # Display message memory as needed.
            'recent_messages_sent': recent_sent,
            'recent_messages_received': recent_recv,
            **self.tasks[self.task].task_obs(agent)
        }

    def _compute_remaining_needed(self) -> dict:
        try:
            task_obj = self.tasks[self.task]
            targets = getattr(task_obj, 'target_items', None)
            if not isinstance(targets, dict):
                return {}
            current = {}
            if hasattr(task_obj, 'count_items_on_tables'):
                current = task_obj.count_items_on_tables(self) or {}
            remaining = {}
            for k, v in targets.items():
                try:
                    remaining[k] = max(0, int(v) - int(current.get(k, 0) or 0))
                except Exception:
                    remaining[k] = int(v) if isinstance(v, int) else 0
            return remaining
        except Exception:
            return {}

    def notify(self, agent):
        # Get the visible messages of other agents
        name = agent.name
        if name not in self.agent_meshes:
            return

        for msg in self.messages:
            if msg['round'] == self.round - 1:  # Messages from the previous round
                speaker_name = msg['agent_name']
                # Use communication range rather than observation range.
                if self.is_in_communication_range(name, speaker_name):
                    recipients = msg.get('recipients', None)
                    deliver = False
                    # Strict P2P: deliver only to explicit recipients; ignore None (broadcast) and empty lists.
                    if isinstance(recipients, list) and len(recipients) > 0:
                        deliver = name in recipients
                    else:
                        deliver = False
                    print(f"DEBUG: notify - [P2P] Checking delivery: from={speaker_name} -> to={name}, recipients={recipients}, deliver={deliver}")
                    if deliver:
                        agent.notify(speaker_name, msg['message'])
                        print(f"DEBUG: notify - {name} received a message from {speaker_name}: {msg['message'][:50]}...")
                        # Update received-message memory.
                        try:
                            bundle = getattr(agent, 'mechanisms', None)
                            if bundle is not None:
                                compat_cfg = dict((getattr(bundle, 'config', {}) or {}).get('compat') or {})
                                incoming_round = int(self.round)
                                if str(compat_cfg.get('profile') or '') == 'stable_fair':
                                    incoming_round = max(0, int(self.round) - 1)
                                notify_ctx = RoundContext(
                                    round_num=incoming_round,
                                    agent_name=name,
                                    obs={'round': incoming_round, 'name': name, 'messages': []},
                                    received_messages={speaker_name: msg['message']},
                                    phase='notify',
                                    config=getattr(bundle, 'config', {}),
                                )
                                bundle.memory.receive_incoming(notify_ctx, bundle.protocol, speaker_name, msg['message'])
                                self._sync_recent_msgs_from_memory(name)
                            elif self.enable_message_memory:
                                self.recent_msgs[name]['recv'].append({'from': speaker_name, 'message': msg['message'], 'round': self.round - 1})
                            print(f"DEBUG: [MSG_MEMORY][RECV] {name} <- {speaker_name} recorded received message")
                        except Exception as e:
                            print(f"WARNING: [MSG_MEMORY][RECV] Recording failed: {e}")

    def is_visible(self, observer_name, target_name):
        """Check whether the target is within the observer's detailed observation range."""
        if observer_name not in self.agent_meshes or target_name not in self.agent_meshes:
            return False

        observer_x, observer_y = self.agent_meshes[observer_name].pos[1], self.agent_meshes[observer_name].pos[0]
        target_x, target_y = self.agent_meshes[target_name].pos[1], self.agent_meshes[target_name].pos[0]

        view_size = self.view_size
        half_view = view_size // 2

        is_in_view = abs(observer_x - target_x) <= half_view and abs(observer_y - target_y) <= half_view
        print(f"DEBUG: is_visible - {observer_name} observing {target_name}: distance(dx={abs(observer_x - target_x)}, dy={abs(observer_y - target_y)}), view range={half_view}, result={is_in_view}")
        return is_in_view

    def is_in_communication_range(self, observer_name, target_name):
        """Check whether the target is within the observer's communication range."""
        if observer_name not in self.agent_meshes or target_name not in self.agent_meshes:
            print(f"DEBUG: is_in_communication_range - agent not found: {observer_name} or {target_name}")
            return False

        observer_x, observer_y = self.agent_meshes[observer_name].pos[1], self.agent_meshes[observer_name].pos[0]
        target_x, target_y = self.agent_meshes[target_name].pos[1], self.agent_meshes[target_name].pos[0]

        communication_size = self.communication_size
        half_comm = communication_size // 2

        is_in_comm = abs(observer_x - target_x) <= half_comm and abs(observer_y - target_y) <= half_comm
        print(f"DEBUG: is_in_communication_range - {observer_name} communicating with {target_name}: distance(dx={abs(observer_x - target_x)}, dy={abs(observer_y - target_y)}), communication range={half_comm}, result={is_in_comm}")
        return is_in_comm

    # ================= Routing helpers =================
    def _routing_filter_candidates(self, agent_name: str, candidates: list[str], exclude: str | None = None) -> list[str]:
        base = [c for c in candidates if c != exclude]
        if not base:
            base = list(candidates)
        recent = list(self.routing_recent_targets.get(agent_name, []))
        filtered = [c for c in base if c not in recent]
        return filtered if filtered else base

    def _routing_recently_answered(self, sender: str, recipient: str) -> bool:
        # with header-only protocol, we no longer gate responses by recent answers
        return False

    def _routing_record_target(self, sender: str, recipient: str) -> None:
        if sender not in self.routing_recent_targets:
            self.routing_recent_targets[sender] = deque(maxlen=self.routing_history_len)
        self.routing_recent_targets[sender].append(recipient)

    # ================= Ring helpers =================
    def _ring_order_angle_halves(self, name: str, obs: dict) -> list[str]:
        agents_info = obs.get('agents_in_communication_range', {}) or {}
        me_pos = obs.get('position', {}) or {}
        coords = {}
        coords[name] = (int(me_pos.get('x', 0)), int(me_pos.get('y', 0)))
        for k, v in agents_info.items():
            p = (v.get('position') or {})
            try:
                coords[k] = (int(p.get('x')), int(p.get('y')))
            except Exception:
                continue
        if len(coords) <= 1:
            return [name]
        # centroid
        cx = sum(x for x, _ in coords.values()) / len(coords)
        cy = sum(y for _, y in coords.values()) / len(coords)
        import math
        def angle_of(k):
            x, y = coords[k]
            return math.atan2(y - cy, x - cx)
        # stable sort by (angle, x, y, id)
        order = sorted(coords.keys(), key=lambda k: (angle_of(k), coords[k][0], coords[k][1], str(k)))
        n = len(order)
        mid = n // 2
        front = order[:mid]
        back = order[mid:]
        interleaved = []
        for i in range(max(len(front), len(back))):
            if i < len(front):
                interleaved.append(front[i])
            if i < len(back):
                interleaved.append(back[i])
        return interleaved

    def _ring_order_angle_skip(self, name: str, obs: dict) -> tuple[list[str], int]:
        # return full order and skip value k
        agents_info = obs.get('agents_in_communication_range', {}) or {}
        me_pos = obs.get('position', {}) or {}
        coords = {name: (int(me_pos.get('x', 0)), int(me_pos.get('y', 0)))}
        for k, v in agents_info.items():
            p = (v.get('position') or {})
            try:
                coords[k] = (int(p.get('x')), int(p.get('y')))
            except Exception:
                continue
        if len(coords) <= 1:
            return [name], 1
        # centroid angle sort
        cx = sum(x for x, _ in coords.values()) / len(coords)
        cy = sum(y for _, y in coords.values()) / len(coords)
        import math
        def angle_of(k):
            x, y = coords[k]
            return math.atan2(y - cy, x - cx)
        order = sorted(coords.keys(), key=lambda k: (angle_of(k), coords[k][0], coords[k][1], str(k)))
        n = len(order)
        kskip = (n + 1) // 2  # ceil(n/2)
        return order, kskip

    def _ring_order_carry_interleave(self, name: str, obs: dict) -> list[str]:
        # Start from angle_halves order; try to alternate by observed carrying info where possible.
        base = self._ring_order_angle_halves(name, obs)
        if len(base) <= 2:
            return base
        info = obs.get('agents_in_communication_range', {}) or {}
        # carrying known if in_view_range true and carrying_resource not None
        carrying = set()
        for k in base:
            if k == name:
                # own carrying via inventory in obs
                try:
                    inv = obs.get('inventory') or {}
                    if int(inv.get('quantity', 0) or 0) > 0:
                        carrying.add(k)
                except Exception:
                    pass
            else:
                v = info.get(k, {})
                if v.get('in_view_range', False) and v.get('carrying_resource'):
                    carrying.add(k)
        # greedy swap to maximize alternating pattern
        arr = base[:]
        n = len(arr)
        def is_carry(k): return k in carrying
        for i in range(n):
            a = arr[i]
            b = arr[(i+1) % n]
            if (is_carry(a) == is_carry(b)) and n > 3:
                # find j to swap with b that alternates
                for j in range(i+2, i+n-1):
                    jj = j % n
                    c = arr[jj]
                    if is_carry(a) != is_carry(c):
                        arr[(i+1) % n], arr[jj] = arr[jj], arr[(i+1) % n]
                        break
        return arr

    def _ring_order_id(self, name: str, obs: dict) -> list[str]:
        """Form an ID-ordered ring by natural numeric name suffix, e.g. Agent_0, Agent_1."""
        agents_info = obs.get('agents_in_communication_range', {}) or {}
        names = [name] + [k for k in agents_info.keys() if k != name]
        import re
        def sort_key(k):
            m = re.search(r'\d+$', str(k))
            return (str(k) if m is None else int(m.group()))
        # For stability, sort first by numeric-suffix presence, then by number/string.
        def full_key(k):
            import re
            m = re.search(r'\d+$', str(k))
            has_num = 0 if m else 1
            val = int(m.group()) if m else 0
            return (has_num, val, str(k))
        order = sorted(names, key=full_key)
        return order

    def _ring_order_random(self, name: str, obs: dict) -> list[str]:
        """Form a reproducible random ring each round using the environment seed and current round."""
        agents_info = obs.get('agents_in_communication_range', {}) or {}
        names = [name] + [k for k in agents_info.keys() if k != name]
        import random
        rnd = random.Random(int(getattr(self, 'seed', 0)) * 1000003 + int(getattr(self, 'round', 0)))
        order = names[:]
        rnd.shuffle(order)
        return order

    def get_agent_positions_in_communication_range(self, agent_name):
        """Return positions of other agents within communication range."""
        if agent_name not in self.agent_meshes:
            print(f"DEBUG: get_agent_positions_in_communication_range - agent {agent_name} does not exist")
            return {}

        agent_positions = {}
        print(f"DEBUG: Getting agent positions within communication range of {agent_name}")

        for other_name, other_mesh in self.agent_meshes.items():
            if other_name != agent_name:  # Exclude self.
                in_comm_range = self.is_in_communication_range(agent_name, other_name)
                in_view_range = self.is_visible(agent_name, other_name)

                if in_comm_range:
                    pos_x_img, pos_y_img = other_mesh.pos[1], other_mesh.pos[0]
                    pos_x, pos_y = to_math_xy(pos_x_img, pos_y_img, self.height)
                    info = {'position': {'x': pos_x, 'y': pos_y}}

                    # Provide resource information only within observation range.
                    if in_view_range:
                        carrying_resource = None
                        # Prefer enhanced inventory when determining carrying state.
                        agent_obj = self.agents.get(other_name)
                        if agent_obj is not None and hasattr(agent_obj, 'enable_inventory') and agent_obj.enable_inventory:
                            try:
                                ctype = agent_obj.get_carrying_item_type()
                                qty = agent_obj.get_total_carrying()
                                carrying_resource = f"{ctype}:{qty}" if qty and ctype else None
                            except Exception:
                                carrying_resource = None
                        elif hasattr(self.tasks[self.task], 'resource_state'):
                            carrying_resource = self.tasks[self.task].resource_state.get(other_name)

                        info.update({
                            'carrying_resource': carrying_resource,
                            'symbol': 'A' if carrying_resource else 'a',
                            'in_view_range': True  # Within observation range.
                        })
                        print(f"DEBUG: Agent {other_name} visible at mathematical coordinates ({pos_x}, {pos_y}), carrying resource: {carrying_resource}")
                    else:
                        info.update({
                            'carrying_resource': None,  # Explicitly unknown.
                            'symbol': '?',  # Unknown-state symbol.
                            'in_view_range': False  # Outside observation range.
                        })
                        print(f"DEBUG: Agent {other_name} within communication range at mathematical coordinates ({pos_x}, {pos_y}), resource state: unknown")

                    agent_positions[other_name] = info

        print(f"DEBUG: {agent_name} has {len(agent_positions)} agents within communication range")
        return agent_positions

    def get_agent_visible_messages(self, agent_name):
        if agent_name not in self.agent_meshes:
            return []

        visible_messages = []
        for msg in self.messages:
            if msg['round'] == self.round - 1:  # the past frame
                speaker_name = msg['agent_name']
                # Use communication range rather than observation range.
                if self.is_in_communication_range(agent_name, speaker_name):
                    recipients = msg.get('recipients', None)
                    deliver = False
                    # Strict P2P: deliver only to explicit recipients; ignore None (broadcast) and empty lists.
                    if isinstance(recipients, list) and len(recipients) > 0:
                        deliver = agent_name in recipients
                    else:
                        deliver = False
                    print(f"DEBUG: get_agent_visible_messages - [P2P] Checking message visibility: from={speaker_name} -> to={agent_name}, recipients={recipients}, deliver={deliver}")
                    if deliver:
                        visible_messages.append(msg)
                        print(f"DEBUG: get_agent_visible_messages - {agent_name} can receive messages from {speaker_name}")

        print(f"DEBUG: {agent_name} total receivable messages: {len(visible_messages)}")
        return visible_messages

    def _is_in_rendered_view_window(self, center_x, center_y, grid_x, grid_y):
        """Match the exact rectangular window returned by get_view()."""
        view_size = int(self.view_size)
        half_view = view_size // 2
        dx = int(grid_x) - int(center_x)
        dy = int(grid_y) - int(center_y)
        return (-half_view <= dx < view_size - half_view and
                -half_view <= dy < view_size - half_view)

    def get_agent_capacity_info(self, agent_name):
        """
        Get capacity information for all visible CapacityMesh objects
        """
        if agent_name not in self.agent_meshes:
            return {}

        capacity_info = {}
        resource_detail_info = {}  # Detailed resource types, including remaining capacity.

        agent_mesh = self.agent_meshes[agent_name]
        agent_x, agent_y = agent_mesh.pos[1], agent_mesh.pos[0]
        view_size = self.view_size

        print(f"DEBUG: Getting capacity information for agent {agent_name}, position=({agent_x}, {agent_y}), view_size={view_size}")

        # Two-layer architecture: terrain type mapping.
        terrain_type_mapping = {
            'T': 'table',       # Table terrain.
            'r': 'resource',    # Resource terrain (stores wood/sand/stone).
            'F': 'forest',      # Forest terrain.
            'G': 'grass',       # Grass terrain.
            '.': 'empty',       # Empty ground, also supports storage.
        }

        # Check all meshes in the environment
        for mesh in self.meshes:
            if isinstance(mesh, CapacityMesh):
                mesh_x, mesh_y = mesh.pos[1], mesh.pos[0]
                # Check if this mesh is within the agent's view
                if self._is_in_rendered_view_window(agent_x, agent_y, mesh_x, mesh_y):
                    # Check terrain types supporting item storage, including empty ground '.'.
                    if mesh.name in ['T', 'r', 'F', 'G', '.']:
                        # Calculate total occupancy and remaining capacity.
                        total_qty = 0
                        items_list = []
                        if hasattr(mesh, 'items') and mesh.items:
                            for it in mesh.items:
                                total_qty += int(getattr(it, 'quantity', 0) or 0)
                                # For tables, expose remaining capacity only, without item lists in view details.
                                if getattr(mesh, 'name', None) == 'T':
                                    continue
                                items_list.append({'type': it.item_type, 'quantity': it.quantity})
                        free_cap = max(0, int(getattr(mesh, 'capacity', 0)) - int(total_qty))

                        # Add remaining capacity information with position
                        capacity_key = f"{mesh.name}({mesh_x},{mesh_y})"
                        capacity_info[capacity_key] = free_cap

                        # Add detailed terrain type information for the two-layer architecture.
                        terrain_name = terrain_type_mapping.get(mesh.name, mesh.name)
                        # Expose positions using mathematical coordinates.
                        out_x, out_y = to_math_xy(mesh_x, mesh_y, self.height)
                        rec = {
                            'terrain_type': terrain_name,      # Terrain layer information.
                            'display_symbol': mesh.name,
                            'position': (out_x, out_y),
                            'capacity': int(getattr(mesh, 'capacity', 0)),  # Total capacity.
                            'item_count': total_qty,                           # Total quantity, summed across items.
                            'remaining_capacity': free_cap,                   # Remaining capacity.
                            'items': items_list
                        }
                        # Tables: attach remaining capacity by type when supported.
                        try:
                            if terrain_name == 'table' and hasattr(mesh, 'get_remaining_by_type') and (getattr(mesh, 'capacity_by_type', None) or getattr(mesh, 'per_type_default', None) is not None):
                                rec['remaining_by_type'] = mesh.get_remaining_by_type(['wood', 'sand', 'stone'])
                        except Exception:
                            pass
                        resource_detail_info[capacity_key] = rec

                        # print(f"DEBUG: Agent {agent_name} sees {terrain_name} at ({mesh_x}, {mesh_y}), capacity={mesh.capacity}, item entries={len(getattr(mesh, 'items', []))}")

        print(f"DEBUG: Agent {agent_name} found {len(capacity_info)} resources")

        # Merge details into the returned result.
        result = capacity_info.copy()
        result['_resource_details'] = resource_detail_info  # Attach details.
        return result

    def get_visible_tables(self, agent_name):
        """Return simplified table info in view: list of {'position': (x,y), 'remaining_capacity': int}."""
        from masbench.physics import CapacityMesh
        if agent_name not in self.agent_meshes:
            return []
        agent_mesh = self.agent_meshes[agent_name]
        agent_x, agent_y = agent_mesh.pos[1], agent_mesh.pos[0]
        tables = []
        for mesh in self.meshes:
            if not isinstance(mesh, CapacityMesh):
                continue
            # only table meshes (symbol 'T')
            if getattr(mesh, 'name', None) != 'T':
                continue
            mesh_x, mesh_y = mesh.pos[1], mesh.pos[0]
            if self._is_in_rendered_view_window(agent_x, agent_y, mesh_x, mesh_y):
                total_qty = 0
                try:
                    if hasattr(mesh, 'items') and mesh.items:
                        for it in mesh.items:
                            total_qty += int(getattr(it, 'quantity', 0) or 0)
                except Exception:
                    total_qty = 0
                cap = int(getattr(mesh, 'capacity', 0) or 0)
                free_cap = max(0, cap - total_qty)
                # Expose positions using mathematical coordinates.
                out_x, out_y = to_math_xy(mesh_x, mesh_y, self.height)
                info = {'position': (out_x, out_y), 'remaining_capacity': free_cap}
                # Attach remaining wood/sand/stone capacities when the table supports per-type limits.
                try:
                    if hasattr(mesh, 'get_remaining_by_type') and (getattr(mesh, 'capacity_by_type', None) or getattr(mesh, 'per_type_default', None) is not None):
                        info['remaining_by_type'] = mesh.get_remaining_by_type(['wood', 'sand', 'stone'])
                except Exception:
                    pass
                tables.append(info)
        return tables

    async def act(self, agent, action):
        name = agent.name
        self.actions[name] = action
        if name not in self.agent_meshes:
            return {'success': False, 'error': 'Agent not found'}

        result = {'success': False}
        original_move_action = action.get('move', 'STAY')

        print(f"DEBUG: [ACT] {name} submitted action: {original_move_action}")

        # Record movement intent; resolve conflicts together during update.
        if 'move' in action and action['move'] in dirc_map:
            move_action = action['move']
            if move_action == 'STAY':
                # STAY always succeeds.
                result['success'] = True
                print(f"DEBUG: [ACT] {name} STAY succeeded immediately")
            else:
                # Record movement intent for joint processing during update.
                result['success'] = True  # Provisional success; actual movement occurs during update.
                print(f"DEBUG: [ACT] {name} movement {move_action} recorded for update")

        # In act, record PICK_UP/PUT_DOWN intent only; execute and log both during update.
        elif 'move' in action and action['move'] == Actions.PICK_UP:
            result['success'] = True  # Action accepted; execution success is determined during update.
            print(f"DEBUG: [ACT] {name} PICK_UP intent recorded for update")

        elif 'move' in action and action['move'] == Actions.PUT_DOWN:
            result['success'] = True  # Action accepted; execution success is determined during update.
            print(f"DEBUG: [ACT] {name} PUT_DOWN intent recorded for update")

        # Disable direct action-phase messaging; use pre-update messaging sharing the action's observation.
        if 'speak' in action and action['speak']:
            print(f"DEBUG: [ACT][MSG][IGNORED] {name} action-phase message ignored (use the pre-update P2P phase); attempted content: '{action.get('speak','')[:60]}'")
            print(f"DEBUG: [ACT][MSG][IGNORED] Let the environment decide per target during the message phase sharing the action observation.")

        # Log all actions, including STAY, together during update.
        print(f"DEBUG: [ACT] {name} action {original_move_action} recorded for end-of-round logging")
        # Save action information and log after execution during update.
        self.delayed_actions = getattr(self, 'delayed_actions', {})
        # Capture visible messages delivered from the previous round for aligned logging.
        visible_msgs = self.get_agent_visible_messages(name)
        msgs_summary = [{'from': m.get('agent_name'), 'text': m.get('message')}
                        for m in visible_msgs]
        print(f"DEBUG: [ACT] {name} visible messages this round: {len(msgs_summary)}")
        # Record enhanced inventory before interaction.
        agent_before = self.agents.get(name)
        inv_before_total = None
        inv_before_type = None
        inv_before_display = None
        if agent_before is not None and hasattr(agent_before, 'enable_inventory') and agent_before.enable_inventory:
            try:
                inv_before_total = agent_before.get_total_carrying()
                inv_before_type = agent_before.get_carrying_item_type()
                inv_before_display = agent_before.get_inventory_display()
                print(f"DEBUG: [ACT] {name} inventory before interaction: total={inv_before_total}, type={inv_before_type}, display='{inv_before_display}'")
            except Exception as e:
                print(f"DEBUG: [ACT] {name} failed to read inventory before interaction: {e}")
        # Snapshot the current cell for logging and analysis.
        capacity_snapshot = {}
        try:
            agent_pos = tuple(self.agent_meshes[name].pos)
            tile_items = self._get_items_at_position(agent_pos)
            tile_free = 0
            tile_type = 'unknown'
            if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                terrain = self.terrain_adapter.get_terrain_at(agent_pos)
                if terrain:
                    tile_type = getattr(terrain, 'terrain_type', tile_type)
                    if getattr(terrain, 'can_store_items', False):
                        cur_total = 0
                        if hasattr(terrain, 'get_items_total_quantity'):
                            cur_total = terrain.get_items_total_quantity()
                        tile_free = max(0, int(getattr(terrain, 'capacity', 0)) - int(cur_total))
                    # Tables expose remaining capacity only; also clear items from capacity snapshots.
                    if getattr(terrain, 'terrain_type', '') == 'table':
                        tile_items = {}
            capacity_snapshot = {
                'tile_terrain_type': tile_type,
                'tile_items_counts': tile_items,
                'tile_placeable_free_capacity': tile_free,
                'remaining_capacity': (agent_before.get_remaining_capacity() if agent_before and hasattr(agent_before, 'get_remaining_capacity') else None)
            }
            print(f"DEBUG: [ACT] {name} capacity_snapshot: {capacity_snapshot}")
        except Exception as e:
            print(f"WARNING: [ACT] Capacity snapshot error: {e}")
        self.delayed_actions[name] = {
            'prompt': action.get('prompt', ''),
            'response': action.get('response', ''),
            'action': original_move_action,
            'message': action.get('speak', ''),
            'api_call_time': action.get('api_call_time', 0),
            'action_success': result['success'],
            'view': self.get_view(name),
            'usage': self.agents[name].brain.usage,
            'messages_received': msgs_summary,
            # Sub-action placeholders, populated during the message phase.
            'subactions': {
                'message_phase': {
                    'decisions': [],
                    'sent': []
                }
            },
            'capacity_snapshot': capacity_snapshot,
            # Enhanced inventory: record state before interaction.
            'inv_before_total': inv_before_total,
            'inv_before_type': inv_before_type,
            'inv_before_display': inv_before_display
        }

        return result

    # Enhanced interaction methods.

    def _validate_pick_up_preconditions(self, agent_name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate PICK_UP preconditions."""
        try:
            print(f"DEBUG: [ENHANCED_PICK_UP] Validating PICK_UP preconditions for {agent_name}")

            # 1. Check that the agent exists.
            if agent_name not in self.agents:
                return False, f"Agent {agent_name} not found", {}

            agent = self.agents[agent_name]

            # 2. Check whether enhanced inventory is enabled.
            if not getattr(agent, 'enable_inventory', False):
                print(f"DEBUG: [ENHANCED_PICK_UP] {agent_name} uses the legacy system; skipping enhanced validation")
                return True, "Traditional system", {}

            # 3. Get agent position.
            agent_mesh = self.agent_meshes[agent_name]
            agent_pos = tuple(agent_mesh.pos)
            print(f"DEBUG: [ENHANCED_PICK_UP] {agent_name} position: {agent_pos}")

            # Tables allow deposits only, not pickups.
            try:
                if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                    terrain = self.terrain_adapter.get_terrain_at(agent_pos)
                    if terrain and getattr(terrain, 'terrain_type', None) == 'table':
                        return False, "Cannot pick up from table (deposit-only)", {}
            except Exception as e:
                print(f"WARNING: [ENHANCED_PICK_UP] Table pickup restriction check failed: {e}")

            # 4. Check for items available at the current position.
            available_items = self._get_items_at_position(agent_pos)
            if not available_items:
                return False, "No items available at current position", {}

            print(f"DEBUG: [ENHANCED_PICK_UP] Available items at current position: {available_items}")

            # 5. Check agent capacity.
            remaining_capacity = agent.get_remaining_capacity()
            if remaining_capacity <= 0:
                return False, f"Agent {agent_name} inventory is full", {}

            print(f"DEBUG: [ENHANCED_PICK_UP] {agent_name} remaining capacity: {remaining_capacity}")

            # 6. Check the single-item-type constraint.
            current_type = agent.get_carrying_item_type()
            compatible_items = {}

            if current_type is not None:
                # If already carrying items, pick up only the same type.
                if current_type in available_items:
                    compatible_items[current_type] = available_items[current_type]
                    print(f"DEBUG: [ENHANCED_PICK_UP] Single-item-type constraint: can pick up only {current_type}")
                else:
                    return False, f"Single-item constraint: carrying {current_type}, no compatible items available", {}
            else:
                # If carrying nothing, any item type is allowed.
                compatible_items = available_items.copy()
                print(f"DEBUG: [ENHANCED_PICK_UP] No item-type restriction; all items are eligible")

            context = {
                'agent_pos': agent_pos,
                'available_items': available_items,
                'compatible_items': compatible_items,
                'remaining_capacity': remaining_capacity,
                'current_type': current_type
            }

            print(f"DEBUG: [ENHANCED_PICK_UP] {agent_name} preconditions passed")
            return True, "Preconditions satisfied", context

        except Exception as e:
            print(f"ERROR: [ENHANCED_PICK_UP] Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}", {}

    def _validate_put_down_preconditions(self, agent_name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate PUT_DOWN preconditions."""
        try:
            print(f"DEBUG: [ENHANCED_PUT_DOWN] Validating PUT_DOWN preconditions for {agent_name}")

            # 1. Check that the agent exists.
            if agent_name not in self.agents:
                return False, f"Agent {agent_name} not found", {}

            agent = self.agents[agent_name]

            # 2. Check whether enhanced inventory is enabled.
            if not getattr(agent, 'enable_inventory', False):
                print(f"DEBUG: [ENHANCED_PUT_DOWN] {agent_name} uses the legacy system; skipping enhanced validation")
                return True, "Traditional system", {}

            # 3. Check whether the agent carries items.
            if agent.get_total_carrying() == 0:
                return False, f"Agent {agent_name} is not carrying any items", {}

            print(f"DEBUG: [ENHANCED_PUT_DOWN] {agent_name} carried items: {agent.get_inventory_display()}")

            # 4. Get agent position.
            agent_mesh = self.agent_meshes[agent_name]
            agent_pos = tuple(agent_mesh.pos)
            print(f"DEBUG: [ENHANCED_PUT_DOWN] {agent_name} position: {agent_pos}")

            # 5. Check whether items can be placed at the current position.
            can_place, place_reason = self._can_place_items_at_position(agent_pos)
            if not can_place:
                return False, f"Cannot place items at {agent_pos}: {place_reason}", {}

            context = {
                'agent_pos': agent_pos,
                'carrying_items': {
                    agent.inventory_item.item_type: agent.inventory_item.quantity
                } if agent.inventory_item else {},
                'place_reason': place_reason
            }

            print(f"DEBUG: [ENHANCED_PUT_DOWN] {agent_name} preconditions passed")
            return True, "Preconditions satisfied", context

        except Exception as e:
            print(f"ERROR: [ENHANCED_PUT_DOWN] Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}", {}

    def _get_items_at_position(self, position: Tuple[int, int]) -> Dict[str, int]:
        """Return items available for pickup at a position, using only the new terrain system."""
        try:
            print(f"DEBUG: [GET_ITEMS] Querying items at {position}")
            y, x = position
            if hasattr(self, 'mesh_map') and 0 <= y < self.mesh_map.shape[0] and 0 <= x < self.mesh_map.shape[1]:
                mesh_obj = self.mesh_map[y, x]
                if mesh_obj is not None and hasattr(mesh_obj, 'items') and mesh_obj.items:
                    items = {}
                    for item in mesh_obj.items:
                        items[item.item_type] = items.get(item.item_type, 0) + item.quantity
                    print(f"DEBUG: [GET_ITEMS] Terrain system found items: {items}")
                    return items
            print(f"DEBUG: [GET_ITEMS] No items at {position}")
            return {}
        except Exception as e:
            print(f"ERROR: [GET_ITEMS] Query error: {str(e)}")
            return {}

    def _can_place_items_at_position(self, position: Tuple[int, int]) -> Tuple[bool, str]:
        """Check whether items can be placed at the specified position."""
        try:
            print(f"DEBUG: [CAN_PLACE] Checking item placement at {position}")

            # Check the mesh to identify table positions.
            if hasattr(self, 'mesh_map'):
                y, x = position
                h, w = self.mesh_map.shape
                print(f"DEBUG: [CAN_PLACE] mesh_map boundary check: position={position}, shape=({h},{w}))")
                if 0 <= y < h and 0 <= x < w:
                    mesh_obj = self.mesh_map[y, x]
                else:
                    mesh_obj = None
                if hasattr(mesh_obj, 'terrain_type') and mesh_obj.terrain_type == 'table':
                    if hasattr(mesh_obj, 'is_depleted') and not mesh_obj.is_depleted():
                        print(f"DEBUG: [CAN_PLACE] Available table found")
                        return True, "Table available"
                    else:
                        print(f"DEBUG: [CAN_PLACE] Table is full")
                        return False, "Table is full"
                elif hasattr(mesh_obj, 'can_store_items') and mesh_obj.can_store_items:
                    print(f"DEBUG: [CAN_PLACE] Terrain supports storage")
                    return True, "Storage terrain available"

            # Check grid boundaries.
            if (0 <= position[0] < self.height and 0 <= position[1] < self.width):
                terrain_char = self.grid[position[0], position[1]]
                if terrain_char in ['.', 'G', 'F', 'T']:  # Passable terrain
                    print(f"DEBUG: [CAN_PLACE] Passable terrain: {terrain_char}")
                    return True, f"Passable terrain {terrain_char}"

            print(f"DEBUG: [CAN_PLACE] Position {position} is unsuitable for item placement")
            return False, "Position not suitable for placing items"

        except Exception as e:
            print(f"ERROR: [CAN_PLACE] Check failed: {str(e)}")
            return False, f"Check error: {str(e)}"

    def _execute_enhanced_pick_up(self, agent_name: str, requested_qty: int | None = None, requested_type: str | None = None) -> Dict[str, Any]:
        """Execute an enhanced PICK_UP action."""
        print(f"DEBUG: [ENHANCED_PICK_UP] Executing PICK_UP for {agent_name}")

        # 1. Validate preconditions.
        valid, reason, context = self._validate_pick_up_preconditions(agent_name)
        if not valid:
            print(f"DEBUG: [ENHANCED_PICK_UP] Preconditions failed: {reason}")
            return {
                'success': False,
                'result': ActionResult.FAILED_INVALID.value,
                'reason': reason,
                'items_picked': {},
                'agent_name': agent_name
            }

        # 2. Skip enhanced processing for the legacy system.
        if reason == "Traditional system":
            return {
                'success': True,
                'result': ActionResult.SUCCESS.value,
                'reason': "Traditional system handled",
                'items_picked': {},
                'agent_name': agent_name
            }

        # 3. Get validation context.
        agent = self.agents[agent_name]
        agent_pos = context['agent_pos']
        compatible_items = context['compatible_items']
        remaining_capacity = context['remaining_capacity']

        # 4. Select pickup type and quantity, preferring explicit values.
        if not compatible_items:
            return {
                'success': False,
                'result': ActionResult.FAILED_NO_ITEMS.value,
                'reason': "No compatible items available",
                'items_picked': {},
                'agent_name': agent_name
            }
        # Explicit mode.
        if getattr(self, 'explicit_interaction_enabled', False):
            if requested_qty is None or requested_qty <= 0:
                return {
                    'success': False,
                    'result': ActionResult.FAILED_INVALID.value,
                    'reason': "QTY required and must be > 0",
                    'items_picked': {},
                    'agent_name': agent_name
                }
            if len(compatible_items) > 1:
                if not requested_type or requested_type not in compatible_items:
                    return {
                        'success': False,
                        'result': ActionResult.FAILED_INVALID.value,
                        'reason': "TYPE required when multiple item types available",
                        'items_picked': {},
                        'agent_name': agent_name
                    }
                selected_type = requested_type
            else:
                selected_type = list(compatible_items.keys())[0]
            max_available = compatible_items[selected_type]
            max_pickable = min(max_available, remaining_capacity)
            if requested_qty > max_pickable:
                return {
                    'success': False,
                    'result': ActionResult.FAILED_CAPACITY.value,
                    'reason': f"Requested qty exceeds limit (max={max_pickable})",
                    'items_picked': {},
                    'agent_name': agent_name
                }
            random_quantity = requested_qty
            print(f"DEBUG: [ENHANCED_PICK_UP][EXPLICIT] Using specified quantity: {selected_type} x{random_quantity} (max={max_pickable})")
        else:
            # Random mode for legacy compatibility.
            available_types = list(compatible_items.keys())
            selected_type = random.choice(available_types)
            max_available = compatible_items[selected_type]
            max_pickable = min(max_available, remaining_capacity)
            if max_pickable <= 0:
                return {
                    'success': False,
                    'result': ActionResult.FAILED_CAPACITY.value,
                    'reason': "No capacity or items available",
                    'items_picked': {},
                    'agent_name': agent_name
                }
            random_quantity = random.randint(1, max_pickable)
            print(f"DEBUG: [ENHANCED_PICK_UP] Random selection: {selected_type} x{random_quantity} (range: 1-{max_pickable})")

        # 5. Perform pickup.
        # Remove items from terrain.
        success_removed = self._remove_items_from_position(agent_pos, selected_type, random_quantity)
        if not success_removed:
            return {
                'success': False,
                'result': ActionResult.FAILED_INVALID.value,
                'reason': "Failed to remove items from terrain",
                'items_picked': {},
                'agent_name': agent_name
            }

        # Add to agent inventory.
        success_added = agent.add_to_inventory(selected_type, random_quantity)
        if not success_added:
            # Roll back terrain removal if inventory addition fails.
            self.add_item_at_position(agent_pos, selected_type, random_quantity)
            return {
                'success': False,
                'result': ActionResult.FAILED_CAPACITY.value,
                'reason': "Failed to add items to agent inventory",
                'items_picked': {},
                'agent_name': agent_name
            }

        print(f"DEBUG: [ENHANCED_PICK_UP] Success: {agent_name} picked up {selected_type} x{random_quantity}")

        return {
            'success': True,
            'result': ActionResult.SUCCESS.value,
            'reason': f"Successfully picked up {random_quantity} {selected_type}",
            'items_picked': {selected_type: random_quantity},
            'agent_name': agent_name,
            'random_quantity': random_quantity,
            'selected_type': selected_type,
            'limit_max_pickable': (max_pickable if 'max_pickable' in locals() else None),
            'remaining_capacity': remaining_capacity,
            'max_available': (max_available if 'max_available' in locals() else None)
        }

    def _execute_enhanced_put_down(self, agent_name: str, requested_qty: int | None = None) -> Dict[str, Any]:
        """Execute an enhanced PUT_DOWN action."""
        print(f"DEBUG: [ENHANCED_PUT_DOWN] Executing PUT_DOWN for {agent_name}")

        # 1. Validate preconditions.
        valid, reason, context = self._validate_put_down_preconditions(agent_name)
        if not valid:
            print(f"DEBUG: [ENHANCED_PUT_DOWN] Preconditions failed: {reason}")
            return {
                'success': False,
                'result': ActionResult.FAILED_INVALID.value,
                'reason': reason,
                'items_put_down': {},
                'agent_name': agent_name
            }

        # 2. Skip enhanced processing for the legacy system.
        if reason == "Traditional system":
            return {
                'success': True,
                'result': ActionResult.SUCCESS.value,
                'reason': "Traditional system handled",
                'items_put_down': {},
                'agent_name': agent_name
            }

        # 3. Get validation context.
        agent = self.agents[agent_name]
        agent_pos = context['agent_pos']

        # 4. Get currently carried items.
        if not agent.inventory_item:
            return {
                'success': False,
                'result': ActionResult.FAILED_NO_ITEMS.value,
                'reason': "Agent is not carrying any items",
                'items_put_down': {},
                'agent_name': agent_name
            }

        carrying_type = agent.inventory_item.item_type
        carrying_quantity = agent.inventory_item.quantity

        # 5. Select put-down quantity, preferring explicit values.
        if getattr(self, 'explicit_interaction_enabled', False):
            if requested_qty is None or requested_qty <= 0:
                return {
                    'success': False,
                    'result': ActionResult.FAILED_INVALID.value,
                    'reason': "QTY required and must be > 0",
                    'items_put_down': {},
                    'agent_name': agent_name
                }
            # Compute placement capacity from total and optional per-type remaining capacity.
            free_cap = 0
            rem_type = None
            if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                terrain = self.terrain_adapter.get_terrain_at(agent_pos)
                if terrain and getattr(terrain, 'can_store_items', False):
                    try:
                        free_cap = int(terrain.get_remaining_total()) if hasattr(terrain, 'get_remaining_total') else max(0, int(getattr(terrain, 'capacity', 0)) - int(terrain.get_items_total_quantity() if hasattr(terrain, 'get_items_total_quantity') else 0))
                    except Exception:
                        free_cap = 0
                    # Remaining capacity for the type, if limited.
                    try:
                        rem_type = terrain.get_remaining_for_type(carrying_type) if hasattr(terrain, 'get_remaining_for_type') else None
                    except Exception:
                        rem_type = None
            # Combined limit.
            type_cap_limit = rem_type if (rem_type is not None) else free_cap
            max_puttable = min(carrying_quantity, type_cap_limit)
            if requested_qty > max_puttable:
                return {
                    'success': False,
                    'result': ActionResult.FAILED_NO_SPACE.value,
                    'reason': f"Requested qty exceeds placeable/carry limit (max={max_puttable})",
                    'items_put_down': {},
                    'agent_name': agent_name
                }
            random_quantity = requested_qty
            print(f"DEBUG: [ENHANCED_PUT_DOWN][EXPLICIT] Using specified put-down: {carrying_type} x{random_quantity} (carried={carrying_quantity}, total_remaining={free_cap}, type_remaining={rem_type})")
        else:
            # Random mode also respects maximum placement capacity.
            # Recalculate the limit, as in explicit mode.
            free_cap = 0
            rem_type = None
            if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                terrain = self.terrain_adapter.get_terrain_at(agent_pos)
                if terrain and getattr(terrain, 'can_store_items', False):
                    try:
                        free_cap = int(terrain.get_remaining_total()) if hasattr(terrain, 'get_remaining_total') else max(0, int(getattr(terrain, 'capacity', 0)) - int(terrain.get_items_total_quantity() if hasattr(terrain, 'get_items_total_quantity') else 0))
                    except Exception:
                        free_cap = 0
                    try:
                        rem_type = terrain.get_remaining_for_type(carrying_type) if hasattr(terrain, 'get_remaining_for_type') else None
                    except Exception:
                        rem_type = None
            type_cap_limit = rem_type if (rem_type is not None) else free_cap
            max_puttable = min(carrying_quantity, type_cap_limit if type_cap_limit > 0 else carrying_quantity)
            if max_puttable <= 0:
                return {
                    'success': False,
                    'result': ActionResult.FAILED_NO_SPACE.value,
                    'reason': "No placeable capacity for this type",
                    'items_put_down': {},
                    'agent_name': agent_name
                }
            random_quantity = random.randint(1, max_puttable)
            print(f"DEBUG: [ENHANCED_PUT_DOWN] Random put-down: {carrying_type} x{random_quantity} (carried={carrying_quantity}, total_remaining={free_cap}, type_remaining={rem_type})")

        # 6. Perform put-down.
        # Remove from agent inventory.
        success_removed = agent.remove_from_inventory(carrying_type, random_quantity)
        if not success_removed:
            return {
                'success': False,
                'result': ActionResult.FAILED_INVALID.value,
                'reason': "Failed to remove items from agent inventory",
                'items_put_down': {},
                'agent_name': agent_name
            }

        # Add to terrain.
        success_added = self.add_item_at_position(agent_pos, carrying_type, random_quantity)
        if not success_added:
            # Restore agent inventory if terrain addition fails.
            agent.add_to_inventory(carrying_type, random_quantity)
            return {
                'success': False,
                'result': ActionResult.FAILED_NO_SPACE.value,
                'reason': "Failed to place items at current position",
                'items_put_down': {},
                'agent_name': agent_name
            }

        # Accumulate individual contribution for deliveries to tables.
        try:
            terrain = self.terrain_adapter.get_terrain_at(agent_pos) if hasattr(self, 'terrain_adapter') and self.terrain_adapter else None
            if terrain and getattr(terrain, 'terrain_type', '') == 'table':
                self.agent_contributions.setdefault(agent_name, {})
                self.agent_contributions[agent_name][carrying_type] = int(self.agent_contributions[agent_name].get(carrying_type, 0)) + int(random_quantity)
                print(f"DEBUG: [CONTRIB] {agent_name} +{random_quantity} {carrying_type} (table)")
        except Exception as e:
            print(f"WARNING: [CONTRIB] Failed to accumulate individual contribution: {e}")

        print(f"DEBUG: [ENHANCED_PUT_DOWN] Success: {agent_name} put down {carrying_type} x{random_quantity}")

        return {
            'success': True,
            'result': ActionResult.SUCCESS.value,
            'reason': f"Successfully put down {random_quantity} {carrying_type}",
            'items_put_down': {carrying_type: random_quantity},
            'agent_name': agent_name,
            'random_quantity': random_quantity,
            'selected_type': carrying_type,
            'limit_max_puttable': (max_puttable if 'max_puttable' in locals() else carrying_quantity),
            'free_capacity': (free_cap if 'free_cap' in locals() else None),
            'carrying_before': carrying_quantity
        }

    def _remove_items_from_position(self, position: Tuple[int, int], item_type: str, quantity: int) -> bool:
        """Remove items from the specified position."""
        try:
            print(f"DEBUG: [REMOVE_ITEMS] At {position}, removing {item_type} x{quantity}")
            y, x = position
            if hasattr(self, 'mesh_map') and 0 <= y < self.mesh_map.shape[0] and 0 <= x < self.mesh_map.shape[1]:
                mesh_obj = self.mesh_map[y, x]
                if hasattr(mesh_obj, 'remove_item'):
                    success = mesh_obj.remove_item(item_type, quantity)
                    print(f"DEBUG: [REMOVE_ITEMS] Terrain removal result: {success}")
                    return bool(success)
            print(f"DEBUG: [REMOVE_ITEMS] Removal failed: position {position} does not support removal or has insufficient items")
            return False
        except Exception as e:
            print(f"ERROR: [REMOVE_ITEMS] Removal error: {str(e)}")
            return False

    def add_item_at_position(self, position, item_type, quantity=1):
        """Add items at the specified position."""
        print(f"DEBUG: [ITEM] Adding items at {position}: {item_type} x{quantity}")

        # Enforce capacity strictly; support mesh_map before terrain_adapter is initialized.
        terrain = None
        # 1) Prefer permanent terrain from terrain_adapter.
        if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
            terrain = self.terrain_adapter.get_terrain_at(position)
        # 2) Fall back to mesh_map during early custom-map construction before adapter integration.
        if terrain is None and hasattr(self, 'mesh_map') and self.mesh_map is not None:
            y, x = position
            if 0 <= y < self.mesh_map.shape[0] and 0 <= x < self.mesh_map.shape[1]:
                mesh_obj = self.mesh_map[y, x]
                if mesh_obj is not None and hasattr(mesh_obj, 'can_store_items'):
                    terrain = mesh_obj

        if terrain and getattr(terrain, 'can_store_items', False):
            # Enforce capacity checks when terrain already exists.
            try:
                if hasattr(terrain, 'get_items_total_quantity'):
                    cur_total = int(terrain.get_items_total_quantity())
                else:
                    cur_total = sum(int(getattr(it, 'quantity', 0) or 0) for it in getattr(terrain, 'items', []) or [])
            except Exception:
                cur_total = 0
            cap = int(getattr(terrain, 'capacity', 0))
            free_cap = max(0, cap - cur_total)
            if quantity > free_cap:
                print(f"DEBUG: [ITEM] Addition failed - exceeds remaining capacity: requested={quantity}, remaining={free_cap}, capacity={cap}, occupied={cur_total}")
                return False

            # Update the terrain system first.
            terrain_success = False
            if hasattr(terrain, 'add_item'):
                terrain_success = terrain.add_item(item_type, quantity)
                print(f"DEBUG: [ITEM] Terrain item addition result: {terrain_success}")
            if not terrain_success:
                print(f"DEBUG: [ITEM] Addition failed - terrain system rejected the items")
                return False

            print(f"DEBUG: [ITEM] Item addition complete - terrain: ✅")
            return True
        else:
            # If terrain is absent, create resource terrain with an r placeholder in mesh_map, then add items through the new system.
            try:
                if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                    base = self.terrain_adapter.create_terrain('resource', position)
                    # Increase capacity if the default is insufficient.
                    if hasattr(base, 'capacity') and int(base.capacity) < int(quantity):
                        base.capacity = int(quantity)
                    ok = base.add_item(item_type, quantity)
                    print(f"DEBUG: [ITEM] Automatic empty-terrain creation and addition result: {ok}")
                    return bool(ok)
            except Exception as e:
                print(f"WARNING: [ITEM] Automatic terrain creation failed: {e}")
            return False

    def remove_item_at_position(self, position, item_type, quantity=1):
        """Remove items from the specified position."""
        print(f"DEBUG: [ITEM] Removing items from {position}: {item_type} x{quantity}")

        # 1) Prefer enhanced terrain; removal must succeed there when storage terrain exists.
        terrain = self.terrain_adapter.get_terrain_at(position) if hasattr(self, 'terrain_adapter') and self.terrain_adapter else None
        if terrain and hasattr(terrain, 'remove_item'):
            terrain_success = terrain.remove_item(item_type, quantity)
            print(f"DEBUG: [ITEM] Terrain item removal result: {terrain_success}")
            return bool(terrain_success)
        print(f"DEBUG: [ITEM] Cannot remove items: no storage terrain here or removal is unsupported")
        return False

    def get_items_at_position(self, position):
        """Return all items at a position, using only the new terrain system."""
        try:
            y, x = position
            mesh_obj = self.mesh_map[y, x] if hasattr(self, 'mesh_map') else None
            if mesh_obj is not None and hasattr(mesh_obj, 'items'):
                return list(mesh_obj.items)
        except Exception:
            pass
        return []

    def has_items_at_position(self, position):
        """Check for items at a position, using only the new terrain system."""
        try:
            y, x = position
            mesh_obj = self.mesh_map[y, x] if hasattr(self, 'mesh_map') else None
            has_items = bool(mesh_obj and hasattr(mesh_obj, 'items') and mesh_obj.items)
            print(f"DEBUG: [ITEM] Items present at {position}: {has_items}")
            return has_items
        except Exception:
            return False

    def move_agent(self, name, direction):
        if name not in self.agent_meshes or direction not in dirc_map:
            return False

        di, dj = dirc_map[direction]
        agent_mesh = self.agent_meshes[name]

        # Target coordinates: x is pos[1] (column), y is pos[0] (row).
        # di changes rows, dj changes columns; mesh.pos = (row, column).
        target_x = agent_mesh.pos[1] + dj  # x is pos[1], the column.
        target_y = agent_mesh.pos[0] + di  # y is pos[0], the row.

        # Check whether the target is within the grid.
        if not (0 <= target_y < self.height and 0 <= target_x < self.width):
             return False # Cannot move outside the map.

        # Check target passability with the terrain adapter.
        if not self.terrain_adapter.is_traversable((target_y, target_x)):
            print(f"DEBUG: Movement failed - position x={target_x}, y={target_y} is impassable")
            return False # Target is impassable.

        # Check for other agents at the target, including STAY cases.
        target_mesh = self.mesh_map[target_y, target_x]
        if target_mesh is not None and isinstance(target_mesh, Mesh) and target_mesh.name.startswith('Agent_'):
            print(f"DEBUG: Movement failed - another agent occupies x={target_x}, y={target_y}")
            return False # Another agent occupies the target.

        # Check for CapacityMesh (enhanced terrain) at the target.
        target_mesh = self.mesh_map[target_y, target_x]
        if isinstance(target_mesh, CapacityMesh):
            # Check passability; walls, rivers, and mountains are obstacles.
            if not self.terrain_adapter.is_traversable((target_y, target_x)):
                print(f"DEBUG: Movement failed - obstacle at x={target_x}, y={target_y} is impassable")
                return False

            # Allow movement onto passable CapacityMesh terrain.
            # 1. Remove from the current position.
            agent_mesh.remove(self.grid, self.mesh_map)
            # 2. Update position.
            agent_mesh.pos = (target_y, target_x)
            # 3. Record the underlying CapacityMesh for subsequent interactions.
            agent_mesh.standing_on = target_mesh
            # 4. Update only grid, preserving the CapacityMesh reference in mesh_map.
            agent_mesh.place(self.grid, None)  # Update grid only, not mesh_map.
            # 5. Update grid display; shape already contains the agent symbol.
            print(f"DEBUG: Movement succeeded - agent {name} moved onto enhanced terrain")
            return True # Movement succeeded.

        # Otherwise, use the original physics engine path.
        res = agent_mesh.move(self.grid, self.mesh_map, di, dj) # May not apply because physics is global.
        # Record push force for the physics engine.
        (self.push_i if dj == 0 else self.push_j)[agent_mesh] = (di if dj == 0 else dj) * 2

        success = len(res) == 0
        print(f"DEBUG: Physics engine movement result: {success}")
        return success # Return the movement outcome.

    def pick_up(self, agent_name):
        """Pick up an item by checking the agent's position directly."""
        print(f"DEBUG: [PICK_UP] Agent {agent_name} attempting PICK_UP")

        if agent_name not in self.agent_meshes:
            print(f"DEBUG: [PICK_UP] Failed - agent {agent_name} does not exist")
            return False

        agent_mesh = self.agent_meshes[agent_name]
        pos = agent_mesh.pos
        print(f"DEBUG: [PICK_UP] Agent {agent_name} current position: x={pos[1]}, y={pos[0]}")

        # Tables allow deposits only, not pickups.
        try:
            if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                terrain = self.terrain_adapter.get_terrain_at((pos[0], pos[1]))
                if terrain and getattr(terrain, 'terrain_type', None) == 'table':
                    print(f"DEBUG: [PICK_UP] Failed - tables allow deposits only; pickup is forbidden")
                    return False
        except Exception as e:
            print(f"WARNING: [PICK_UP] Table pickup restriction check failed: {e}")

        # Check directly for items at the position.
        if not self.has_items_at_position(pos):
            print(f"DEBUG: [PICK_UP] Failed - no items at {pos}")
            return False

        # Check whether the agent already carries an item.
        current_item = self.tasks[self.task].resource_state.get(agent_name)
        if current_item is not None:
            print(f"DEBUG: [PICK_UP] Failed - agent {agent_name} already carries: {current_item}")
            return False

        print(f"DEBUG: [PICK_UP] Agent {agent_name} carries nothing and can pick up an item")

        # Get items at the position.
        items = self.get_items_at_position(pos)
        if not items:
            print(f"DEBUG: [PICK_UP] Failed - item list at {pos} is empty")
            return False

        # Pick up the first item.
        item_to_pick = items[0]
        print(f"DEBUG: [PICK_UP] Attempting to pick up: {item_to_pick}")
        success = self.remove_item_at_position(pos, item_to_pick.item_type, 1)

        if success:
            self.tasks[self.task].resource_state[agent_name] = item_to_pick.item_type
            print(f"DEBUG: [PICK_UP] ✅ Success - agent {agent_name} picked up {item_to_pick.item_type}")
            print(f"DEBUG: [PICK_UP] Agent {agent_name} state updated - now carrying: {self.tasks[self.task].resource_state.get(agent_name)}")

            # Update the agent object's carrying state.
            agent = self.agents[agent_name]
            agent.is_carrying = True
            agent.carrying_item = item_to_pick.item_type
            agent.item_quantity = 1
            print(f"DEBUG: [PICK_UP] Agent object updated - is_carrying: {agent.is_carrying}, carrying_item: {agent.carrying_item}, item_quantity: {agent.item_quantity}")

            # Update the agent's appearance.
            agent_mesh.shape[:, :] = 'A'
            agent_mesh.place(self.grid, None)
            print(f"DEBUG: [PICK_UP] Agent {agent_name} appearance updated to carrying (A)")
        else:
            print(f"DEBUG: [PICK_UP] ❌ Failed - item removal failed")

        return success

    def put_down(self, agent_name):
        """Put down an item directly at the agent's position."""
        print(f"DEBUG: [PUT_DOWN] Agent {agent_name} attempting PUT_DOWN")

        if agent_name not in self.agent_meshes:
            print(f"DEBUG: [PUT_DOWN] Failed - agent {agent_name} does not exist")
            return False

        agent_mesh = self.agent_meshes[agent_name]
        pos = agent_mesh.pos
        print(f"DEBUG: [PUT_DOWN] Agent {agent_name} current position: x={pos[1]}, y={pos[0]}")

        # Check whether the agent carries an item.
        carried_item = self.tasks[self.task].resource_state.get(agent_name)
        if carried_item is None:
            print(f"DEBUG: [PUT_DOWN] Failed - agent {agent_name} carries nothing")
            print(f"DEBUG: [PUT_DOWN] Agent {agent_name} current state: {self.tasks[self.task].resource_state}")
            return False

        print(f"DEBUG: [PUT_DOWN] Agent {agent_name} currently carrying: {carried_item}")

        # Optionally check whether terrain allows placement here.
        terrain = self.terrain_adapter.get_terrain_at((pos[0], pos[1]))
        can_place_here = True

        if terrain:
            # Reject put-down if terrain explicitly prohibits storage.
            if hasattr(terrain, 'can_store_items') and not terrain.can_store_items:
                print(f"DEBUG: [PUT_DOWN] Failed - terrain {terrain.terrain_type} prohibits item storage")
                can_place_here = False

        if not can_place_here:
            return False

        # Put down the item directly at the position.
        print(f"DEBUG: [PUT_DOWN] At {pos}, putting down {carried_item}")
        success = self.add_item_at_position(pos, carried_item, 1)

        if success:
            self.tasks[self.task].resource_state[agent_name] = None
            print(f"DEBUG: [PUT_DOWN] ✅ Success - agent {agent_name} put down {carried_item}")
            print(f"DEBUG: [PUT_DOWN] Agent {agent_name} state updated - now carrying: {self.tasks[self.task].resource_state.get(agent_name)}")

            # Update the agent object's carrying state.
            agent = self.agents[agent_name]
            agent.is_carrying = False
            agent.carrying_item = None
            agent.item_quantity = 0
            print(f"DEBUG: [PUT_DOWN] Agent object updated - is_carrying: {agent.is_carrying}, carrying_item: {agent.carrying_item}, item_quantity: {agent.item_quantity}")

            # Update the agent's appearance.
            agent_mesh.shape[:, :] = 'a'
            agent_mesh.place(self.grid, None)
            print(f"DEBUG: [PUT_DOWN] Agent {agent_name} appearance updated to not carrying (a)")

            # Check the current terrain state and display.
            terrain = self.terrain_adapter.get_terrain_at((pos[0], pos[1]))
            if terrain:
                print(f"DEBUG: [PUT_DOWN] Terrain details at ({pos[0]}, {pos[1]}): {terrain.get_terrain_info()}")
                print(f"DEBUG: [PUT_DOWN] Terrain display name: '{terrain.get_display_name()}'")

            # Inspect mesh information in mesh_map.
            y, x = pos[0], pos[1]
            mesh_in_map = self.mesh_map[y, x] if 0 <= y < self.mesh_map.shape[0] and 0 <= x < self.mesh_map.shape[1] else None
            if mesh_in_map:
                print(f"DEBUG: [PUT_DOWN] mesh_map[{y},{x}] mesh type: {type(mesh_in_map).__name__}")
                print(f"DEBUG: [PUT_DOWN] mesh_map[{y},{x}] mesh: {mesh_in_map}")
                print(f"DEBUG: [PUT_DOWN] mesh_map[{y},{x}] is EnhancedCapacityMesh: {isinstance(mesh_in_map, EnhancedCapacityMesh)}")
                print(f"DEBUG: [PUT_DOWN] mesh_map[{y},{x}] is CapacityMesh: {isinstance(mesh_in_map, CapacityMesh)}")
                if hasattr(mesh_in_map, 'items'):
                    print(f"DEBUG: [PUT_DOWN] mesh_map[{y},{x}].items: {mesh_in_map.items}")
                if hasattr(mesh_in_map, 'shape'):
                    print(f"DEBUG: [PUT_DOWN] mesh_map[{y},{x}].shape: {mesh_in_map.shape}")
                if hasattr(mesh_in_map, 'get_display_name'):
                    display_name = mesh_in_map.get_display_name()
                    print(f"DEBUG: [PUT_DOWN] mesh_map[{y},{x}].get_display_name(): '{display_name}'")
            else:
                print(f"DEBUG: [PUT_DOWN] ❌ mesh_map[{y},{x}] is None")

            # Mark only the PUT_DOWN position for rendering diagnostics.
            if not hasattr(self, '_debug_put_down_positions'):
                self._debug_put_down_positions = set()
            self._debug_put_down_positions.add((y, x))
            print(f"DEBUG: [PUT_DOWN] 🎯 Marked ({y},{x}) for rendering diagnostics")

            # Show a local grid preview.
            y, x = pos[0], pos[1]
            min_i = max(0, y-2); max_i = min(self.height, y+3)
            min_j = max(0, x-2); max_j = min(self.width, x+3)
            preview = self.grid[min_i:max_i, min_j:max_j]
            print(f"DEBUG: [PUT_DOWN] Local grid preview centered at ({y},{x}):")
            for i in range(preview.shape[0]):
                row_str = "  ".join(str(preview[i, j]) for j in range(preview.shape[1]))
                print(f"DEBUG: [PUT_DOWN]   {row_str}")

            # Disable legacy scoring in favor of target-item quantity scoring.
            # Calculate scores once at round end to prevent double counting.
        else:
            print(f"DEBUG: [PUT_DOWN] ❌ Failed - item addition failed")

        return success

    def _can_pick_up(self, agent_name):
        """Check pickup eligibility directly from items at the position; return (success, reason)."""
        print(f"DEBUG: [CHECK] Checking pickup preconditions for {agent_name}")

        if agent_name not in self.agent_meshes:
            return False, f"Agent {agent_name} not found in environment"

        agent_mesh = self.agent_meshes[agent_name]
        pos = agent_mesh.pos

        # Check whether an item is already carried.
        if self.tasks[self.task].resource_state.get(agent_name) is not None:
            carried_item = self.tasks[self.task].resource_state[agent_name]
            return False, f"Agent already carrying item: {carried_item}"

        # Check directly for items at the position.
        if not self.has_items_at_position(pos):
            return False, f"No items available at position {pos}"

        # Check item quantity.
        items = self.get_items_at_position(pos)
        total_items = sum(item.quantity for item in items)
        if total_items <= 0:
            return False, f"All items depleted at position {pos}"

        print(f"DEBUG: [CHECK] {agent_name} pickup preconditions passed")
        return True, "All pre-conditions satisfied"

    def _can_put_down(self, agent_name):
        """Check put-down eligibility with simplified terrain checks; return (success, reason)."""
        print(f"DEBUG: [CHECK] Checking put-down preconditions for {agent_name}")

        if agent_name not in self.agent_meshes:
            return False, f"Agent {agent_name} not found in environment"

        # Check whether an item is carried.
        carried_item = self.tasks[self.task].resource_state.get(agent_name)
        if carried_item is None:
            return False, f"Agent not carrying any item"

        agent_mesh = self.agent_meshes[agent_name]
        pos = agent_mesh.pos

        # Optional terrain check: reject only explicit storage prohibitions.
        terrain = self.terrain_adapter.get_terrain_at((pos[0], pos[1]))
        if terrain and hasattr(terrain, 'can_store_items') and not terrain.can_store_items:
            return False, f"Terrain type '{terrain.terrain_type}' does not allow item storage"

        print(f"DEBUG: [CHECK] {agent_name} put-down preconditions passed")
        return True, "All pre-conditions satisfied"

    def update_B(self):
        print(f"DEBUG: [UPDATE_B] Starting physics update, round={self.round}")
        for di, dj in dirc_map.values():
            print(f"DEBUG: [UPDATE_B] Processing direction ({di}, {dj})")
            print(f"DEBUG: [UPDATE_B] Before detect_all")
            Mesh.detect_all(self.mesh_map, di, dj)
            print(f"DEBUG: [UPDATE_B] detect_all complete")

            push_forces = self.push_i if dj == 0 else self.push_j
            print(f"DEBUG: [UPDATE_B] Push force count: {len(push_forces)}")
            for mesh, force in push_forces.items():
                if mesh.node is None:
                    print(f"DEBUG: [UPDATE_B] ❌ mesh.node is None; skipping force assignment: mesh={mesh}")
                    continue
                mesh.node.force = force * (di if dj == 0 else dj)
                print(f"DEBUG: [UPDATE_B] Setting push force: mesh={mesh}, force={mesh.node.force}")

            print(f"DEBUG: [UPDATE_B] Before build_dag")
            heads = Mesh.build_dag(self.mesh_map)
            print(f"DEBUG: [UPDATE_B] build_dag complete; head node count: {len(heads)}")

            nodes = Node.pass_forward(heads)
            print(f"DEBUG: [UPDATE_B] pass_forward complete; movable node count: {len(nodes)}")

            Mesh.move_all(self.grid, self.mesh_map, nodes, di, dj)
            print(f"DEBUG: [UPDATE_B] move_all complete")
            # print(f'Pass forward ({di}, {dj}): {nodes}')
        self.push_i = {}
        self.push_j = {}
        print(f"DEBUG: [UPDATE_B] Physics update complete")


    def update(self):
        self.update_B()

        # Optimization 1: collect intents in dictionaries.
        print(f"DEBUG: [UPDATE] Collecting agent intents (Round {self.round})")
        agent_intentions = {}  # name -> intention_dict

        for name, action in self.actions.items():
            agent_mesh = self.agent_meshes.get(name)
            if not agent_mesh:
                continue

            move_action = action.get('move', 'STAY')
            current_pos = agent_mesh.pos

            # Optimization 1: dictionary-based intent structure.
            intention = {
                'move': move_action,
                'interaction': None,
                'current_pos': current_pos,
                'target_pos': current_pos,
                'original_action': move_action  # Preserve the original action for logging.
            }

            # Process movement actions.
            if move_action in dirc_map and move_action != 'STAY':
                di, dj = dirc_map[move_action]
                ty, tx = current_pos[0] + di, current_pos[1] + dj
                if 0 <= ty < self.height and 0 <= tx < self.width:
                    intention['target_pos'] = (ty, tx)
                print(f"DEBUG: [INTENTION] {name} movement intent: {move_action} -> {intention['target_pos']}")

            # Optimization 1: process interaction actions.
            elif move_action == 'PICK_UP':
                intention['move'] = 'STAY'
                intention['interaction'] = 'PICK_UP'
                print(f"DEBUG: [INTENTION] {name} intends PICK_UP (converted to STAY + interaction)")

            elif move_action == 'PUT_DOWN':
                intention['move'] = 'STAY'
                intention['interaction'] = 'PUT_DOWN'
                print(f"DEBUG: [INTENTION] {name} intends PUT_DOWN (converted to STAY + interaction)")

            elif move_action == 'STAY':
                print(f"DEBUG: [INTENTION] {name} intends STAY")

            agent_intentions[name] = intention
            print(f"DEBUG: [INTENTION] {name} final intent: {intention}")

        # 2. Compute preliminary final positions, considering shared targets and wall collisions only.
        preliminary_positions = {}

        # 2a. Resolve conflicts over the same destination.
        print(f"DEBUG: [UPDATE] Resolving position conflicts")
        claims = {}
        for name, intention in agent_intentions.items():
            target_pos = intention['target_pos']
            if target_pos not in claims:
                claims[target_pos] = []
            claims[target_pos].append(name)
            print(f"DEBUG: [CLAIMS] {name} claims position {target_pos}")

        successful_claims = set()
        for target_pos, contenders in claims.items():
            if len(contenders) == 1:
                successful_claims.add(contenders[0])
                continue
            stayers = [name for name in contenders if agent_intentions[name]['move'] == 'STAY']
            if len(stayers) >= 1:
                winner = sorted(stayers)[0]
            else:
                winner = sorted(contenders)[0]
            successful_claims.add(winner)
            for name in contenders:
                if name != winner:
                    print(f"DEBUG: [CONFLICT] Movement conflict: {name} targets {target_pos}, but {winner} won the cell")

        # 2b. Compute preliminary positions, considering wall collisions only.
        print(f"DEBUG: [UPDATE] Computing preliminary positions")
        for name, intention in agent_intentions.items():
            current_pos = intention['current_pos']
            target_pos = intention['target_pos']
            if name in successful_claims:
                target_mesh = self.mesh_map[target_pos]
                target_x = target_pos[1]
                target_y = target_pos[0]
                is_wall = target_mesh and not self.terrain_adapter.is_traversable((target_y, target_x))
                if is_wall:
                    preliminary_positions[name] = current_pos
                else:
                    preliminary_positions[name] = target_pos
            else:
                preliminary_positions[name] = current_pos

        # 3. Resolve chain collisions iteratively until positions stabilize.
        final_positions = preliminary_positions.copy()
        while True:
            collided = False
            agent_names = sorted(list(final_positions.keys()))
            # pos -> [occupants]
            end_pos_map = {}
            for name in agent_names:
                pos = final_positions[name]
                if pos not in end_pos_map:
                    end_pos_map[pos] = []
                end_pos_map[pos].append(name)

            for pos, occupants in end_pos_map.items():
                if len(occupants) > 1:
                    # On collision, return every colliding agent to its original position.
                    collided = True
                    for agent_name in occupants:
                        final_positions[agent_name] = agent_intentions[agent_name]['current_pos'] # Revert to current_pos

            if not collided:
                break

        # 3.5 Record actual movement success.
        move_action_success = {}
        for name, intention in agent_intentions.items():
            if intention['move'] != 'STAY' and intention['move'] in dirc_map:
                # Movement succeeds if the final position equals the target.
                current_pos = intention['current_pos']
                target_pos = intention['target_pos']
                final_pos = final_positions[name]

                move_success = (final_pos == target_pos)
                move_action_success[name] = move_success
                print(f"DEBUG: [MOVE_SUCCESS] {name} actual movement success={move_success}, current={current_pos}, target={target_pos}, final={final_pos}")
                # Record action feedback for the next round's prompt.
                try:
                    self.last_action_feedback = getattr(self, 'last_action_feedback', {})
                    self.last_action_feedback[name] = {
                        'action': intention['move'],
                        'success': move_success,
                        'reason': (None if move_success else 'Move blocked or conflict')
                    }
                except Exception:
                    pass
                # Record the action trace.
                try:
                    self.delayed_actions[name].setdefault('action_trace', [])
                    trace = {
                        'phase': 'move',
                        'current_pos': current_pos,
                        'target_pos': target_pos,
                        'final_pos': final_pos,
                        'success': move_success
                    }
                    # Approximate failure cause: wall/mountain if impassable, otherwise conflict.
                    if not move_success:
                        target_mesh = self.mesh_map[target_pos]
                        traversable = self.terrain_adapter.is_traversable((target_pos[0], target_pos[1])) if hasattr(self, 'terrain_adapter') and self.terrain_adapter else True
                        if not traversable:
                            trace['reason'] = 'Not traversable (wall/mountain/obstacle)'
                        else:
                            trace['reason'] = 'Position conflict'
                    self.delayed_actions[name]['action_trace'].append(trace)
                except Exception:
                    pass

        # 4. Phase three: apply state changes.
        # 4a. Remove all agents from old positions.
        for agent_mesh in self.agent_meshes.values():
            if hasattr(agent_mesh, 'standing_on') and agent_mesh.standing_on is not None:
                agent_mesh.remove(self.grid, None)
            else:
                agent_mesh.remove(self.grid, self.mesh_map)

        # 4b. Place all agents at new positions.
        for name, new_pos in final_positions.items():
            agent_mesh = self.agent_meshes[name]
            agent_mesh.pos = new_pos
            mesh_underneath = self.mesh_map[new_pos]

            if isinstance(mesh_underneath, CapacityMesh):
                agent_mesh.place(grid=self.grid)
                agent_mesh.standing_on = mesh_underneath
            else:
                agent_mesh.place(self.grid, self.mesh_map)
                if hasattr(agent_mesh, 'standing_on'):
                    agent_mesh.standing_on = None

        # 6. Process item interactions after position updates.
        print(f"DEBUG: [INTERACTION] Processing item interactions")

        for name, intention in agent_intentions.items():
            interaction_type = intention.get('interaction')
            if interaction_type:
                print(f"DEBUG: [INTERACTION] Processing {name}: {interaction_type}")

                if interaction_type == 'PICK_UP':
                    valid, reason, context = self._validate_pick_up_preconditions(name)
                    if valid:
                        req_qty = None
                        req_typ = None
                        try:
                            req_qty = self.actions.get(name, {}).get('interaction_qty')
                            req_typ = self.actions.get(name, {}).get('interaction_type')
                        except Exception:
                            pass
                        # Record detailed pre-action context.
                        try:
                            self.delayed_actions[name].setdefault('action_trace', [])
                            self.delayed_actions[name]['action_trace'].append({'phase': 'preconditions', 'context': context})
                        except Exception:
                            pass
                        enhanced_result = self._execute_enhanced_pick_up(name, requested_qty=req_qty, requested_type=req_typ)
                        success = bool(enhanced_result.get('success'))
                        print(f"DEBUG: [INTERACTION] {name} enhanced PICK_UP result: {success}")
                        # (Optional) we no longer record plan here for response memory only
                        # Record interaction requests and outcomes.
                        try:
                            self.delayed_actions[name]['interaction_request'] = {'action': 'PICK_UP', 'qty': req_qty, 'type': req_typ}
                            self.delayed_actions[name]['interaction_result'] = {
                                'success': success,
                                'effective_qty': (enhanced_result.get('random_quantity') if success else 0),
                                'type': enhanced_result.get('selected_type'),
                                'reason': enhanced_result.get('reason')
                            }
                            # Trace constraints and outcomes.
                            self.delayed_actions[name]['action_trace'].append({
                                'phase': 'execute',
                                'limits': {
                                    'max_pickable': enhanced_result.get('limit_max_pickable'),
                                    'remaining_capacity': enhanced_result.get('remaining_capacity'),
                                    'max_available': enhanced_result.get('max_available')
                                },
                                'result': self.delayed_actions[name]['interaction_result']
                            })
                            # Record the post-action snapshot.
                            agent_pos = tuple(self.agent_meshes[name].pos)
                            after_items = self._get_items_at_position(agent_pos)
                            self.delayed_actions[name]['action_trace'].append({
                                'phase': 'after',
                                'tile_items_after': after_items,
                                'inventory_after': self.agents[name].get_inventory_display() if hasattr(self.agents[name], 'get_inventory_display') else None
                            })
                            # Set previous-action feedback.
                            self.last_action_feedback = getattr(self, 'last_action_feedback', {})
                            self.last_action_feedback[name] = {
                                'action': 'PICK_UP',
                                'success': success,
                                'reason': enhanced_result.get('reason')
                            }
                        except Exception:
                            pass

                        # Update agent appearance after success.
                        if success:
                            # Refresh this agent's memory/KB immediately to avoid stale information after collecting all items.
                            try:
                                agent = self.agents.get(name)
                                if agent is not None and hasattr(agent, '_kb_merge_entry'):
                                    agent_pos_rc = tuple(self.agent_meshes[name].pos)  # (row=y_img, col=x_img)
                                    x_img = int(agent_pos_rc[1]); y_img = int(agent_pos_rc[0])
                                    # Convert to math coords for KB
                                    pos_xy = (x_img, int(self.height - 1 - y_img))
                                    # Read current remaining items at the position and aggregate them into an items dictionary.
                                    after_items_now = self._get_items_at_position(agent_pos_rc) or {}
                                    # Apply the complete items dictionary as a patch; an empty dictionary triggers clearing.
                                    items_patch = {k: int(v) for k, v in after_items_now.items() if int(v) > 0}
                                    patch = {
                                        'terrain_type': 'resource',
                                        'items': items_patch
                                    }
                                    agent._kb_merge_entry(pos_xy, patch, 'env', int(self.round))
                                    print(f"DEBUG: [KB][ENV] on PICK_UP pos={pos_xy} items={items_patch}")
                            except Exception as e:
                                print(f"WARNING: [KB][ENV] Immediate refresh failed: {e}")

                            agent_mesh = self.agent_meshes[name]
                            agent_mesh.shape[:, :] = 'A'
                            agent_mesh.place(self.grid, None)
                            print(f"DEBUG: [INTERACTION] ✅ {name} PICK_UP succeeded; appearance updated to carrying (A)")
                            # Display enhanced inventory state.
                            agent = self.agents[name]
                            print(f"DEBUG: [INTERACTION] {name} current inventory: {agent.get_inventory_display()}")
                    else:
                        print(f"DEBUG: [INTERACTION] {name} PICK_UP preconditions failed: {reason}")

                elif interaction_type == 'PUT_DOWN':
                    valid, reason, context = self._validate_put_down_preconditions(name)
                    if valid:
                        req_qty = None
                        try:
                            req_qty = self.actions.get(name, {}).get('interaction_qty')
                        except Exception:
                            pass
                        # Record detailed pre-action context.
                        try:
                            self.delayed_actions[name].setdefault('action_trace', [])
                            self.delayed_actions[name]['action_trace'].append({'phase': 'preconditions', 'context': context})
                        except Exception:
                            pass
                        enhanced_result = self._execute_enhanced_put_down(name, requested_qty=req_qty)
                        success = bool(enhanced_result.get('success'))
                        print(f"DEBUG: [INTERACTION] {name} enhanced PUT_DOWN result: {success}")
                        # (Optional) not recording plan; response memory records only raw LLM outputs
                        # Record interaction requests and outcomes.
                        try:
                            self.delayed_actions[name]['interaction_request'] = {'action': 'PUT_DOWN', 'qty': req_qty}
                            self.delayed_actions[name]['interaction_result'] = {
                                'success': success,
                                'effective_qty': (enhanced_result.get('random_quantity') if success else 0),
                                'type': enhanced_result.get('selected_type'),
                                'reason': enhanced_result.get('reason')
                            }
                            # Trace constraints and outcomes.
                            self.delayed_actions[name]['action_trace'].append({
                                'phase': 'execute',
                                'limits': {
                                    'max_puttable': enhanced_result.get('limit_max_puttable'),
                                    'free_capacity': enhanced_result.get('free_capacity'),
                                    'carrying_before': enhanced_result.get('carrying_before')
                                },
                                'result': self.delayed_actions[name]['interaction_result']
                            })
                            # Record the post-action snapshot.
                            agent_pos = tuple(self.agent_meshes[name].pos)
                            after_items = self._get_items_at_position(agent_pos)
                            self.delayed_actions[name]['action_trace'].append({
                                'phase': 'after',
                                'tile_items_after': after_items,
                                'inventory_after': self.agents[name].get_inventory_display() if hasattr(self.agents[name], 'get_inventory_display') else None
                            })
                            # Set previous-action feedback.
                            self.last_action_feedback = getattr(self, 'last_action_feedback', {})
                            self.last_action_feedback[name] = {
                                'action': 'PUT_DOWN',
                                'success': success,
                                'reason': enhanced_result.get('reason')
                            }
                        except Exception:
                            pass

                        if success:
                            # Immediately refresh table remaining capacity at this position in the agent's KB.
                            try:
                                agent = self.agents.get(name)
                                if agent is not None and hasattr(agent, '_kb_merge_entry'):
                                    agent_pos_rc = tuple(self.agent_meshes[name].pos)  # (row=y_img, col=x_img)
                                    x_img = int(agent_pos_rc[1]); y_img = int(agent_pos_rc[0])
                                    pos_xy = (x_img, int(self.height - 1 - y_img))
                                    terrain = None
                                    try:
                                        if hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                                            terrain = self.terrain_adapter.get_terrain_at(agent_pos_rc)
                                    except Exception:
                                        terrain = None
                                    if terrain and getattr(terrain, 'terrain_type', '') == 'table':
                                        try:
                                            cur_total = int(terrain.get_items_total_quantity()) if hasattr(terrain, 'get_items_total_quantity') else 0
                                        except Exception:
                                            cur_total = 0
                                        cap = int(getattr(terrain, 'capacity', 0) or 0)
                                        rem = max(0, cap - cur_total)
                                        patch = {'terrain_type': 'table', 'remaining_capacity': int(rem)}
                                        # Attach per-type remaining capacity when supported.
                                        try:
                                            if hasattr(terrain, 'get_remaining_by_type') and (getattr(terrain, 'capacity_by_type', None) or getattr(terrain, 'per_type_default', None) is not None):
                                                rb = terrain.get_remaining_by_type(['wood', 'sand', 'stone'])
                                                if rb:
                                                    patch['remaining_by_type'] = rb
                                        except Exception:
                                            pass
                                        agent._kb_merge_entry(pos_xy, patch, 'env', int(self.round))
                                        print(f"DEBUG: [KB][ENV] on PUT_DOWN pos={pos_xy} table remaining={rem}")
                                    elif terrain and getattr(terrain, 'can_store_items', False):
                                        # For non-table cells, update items at this position in the KB.
                                        try:
                                            after_items = self._get_items_at_position(agent_pos_rc) or {}
                                            items_patch = {k: int(v) for k, v in after_items.items() if int(v) > 0}
                                            patch = {
                                                'terrain_type': getattr(terrain, 'terrain_type', 'resource'),
                                                'items': items_patch
                                            }
                                            agent._kb_merge_entry(pos_xy, patch, 'env', int(self.round))
                                            print(f"DEBUG: [KB][ENV] on PUT_DOWN pos={pos_xy} items={items_patch}")
                                        except Exception as e:
                                            print(f"WARNING: [KB][ENV] PUT_DOWN non-table item refresh failed: {e}")
                            except Exception as e:
                                print(f"WARNING: [KB][ENV] PUT_DOWN immediate refresh failed: {e}")
                            agent_mesh = self.agent_meshes[name]
                            agent_mesh.shape[:, :] = 'a'
                            agent_mesh.place(self.grid, None)
                            print(f"DEBUG: [INTERACTION] ✅ {name} PUT_DOWN succeeded; appearance updated to not carrying (a)")
                            agent = self.agents[name]
                            print(f"DEBUG: [INTERACTION] {name} current inventory: {agent.get_inventory_display()}")
                    else:
                        print(f"DEBUG: [INTERACTION] {name} PUT_DOWN preconditions failed: {reason}")

        print(f"DEBUG: [INTERACTION] Item interactions complete")

        # Write deferred PICK_UP/PUT_DOWN action logs.
        if hasattr(self, 'delayed_actions') and self.delayed_actions:
            print(f"DEBUG: [DELAYED_LOG] Writing deferred action logs")
            for name, action_data in self.delayed_actions.items():
                if name not in self.actions:
                    print(f"DEBUG: [DELAYED_LOG] Skipping message-only placeholder: {name}")
                    continue
                # Get the actual execution outcome.
                actual_success = False
                if name in agent_intentions:
                    intention = agent_intentions[name]
                    if intention.get('interaction') == 'PICK_UP':
                        # Enhanced inventory: determine success from quantity changes.
                        agent_now = self.agents.get(name)
                        if agent_now and hasattr(agent_now, 'enable_inventory') and agent_now.enable_inventory:
                            before = action_data.get('inv_before_total')
                            try:
                                after = agent_now.get_total_carrying()
                            except Exception:
                                after = None
                            actual_success = (before is not None and after is not None and after > before)
                            print(f"DEBUG: [DELAYED_LOG] {name} actual PICK_UP success (enhanced): before={before}, after={after}, success={actual_success}")
                        else:
                            # Legacy system fallback.
                            actual_success = self.tasks[self.task].resource_state.get(name) is not None
                    elif intention.get('interaction') == 'PUT_DOWN':
                        # Enhanced inventory: determine success from a quantity decrease.
                        agent_now = self.agents.get(name)
                        if agent_now and hasattr(agent_now, 'enable_inventory') and agent_now.enable_inventory:
                            before = action_data.get('inv_before_total')
                            try:
                                after = agent_now.get_total_carrying()
                            except Exception:
                                after = None
                            actual_success = (before is not None and after is not None and after < before)
                            print(f"DEBUG: [DELAYED_LOG] {name} actual PUT_DOWN success (enhanced): before={before}, after={after}, success={actual_success}")
                        else:
                            # Legacy system fallback.
                            carried_before = action_data.get('carried_before_put_down')
                            if carried_before is not None:
                                actual_success = self.tasks[self.task].resource_state.get(name) is None
                            else:
                                actual_success = False
                    elif action_data['action'] in dirc_map and action_data['action'] != 'STAY':
                        # Check whether movement actually succeeded.
                        if name in move_action_success:
                            actual_success = move_action_success[name]
                        else:
                            # Default to failure if no movement outcome was recorded.
                            actual_success = False

                print(f"DEBUG: [DELAYED_LOG] {name} deferred action log: {action_data['action']}, actual_success={actual_success}")
                self.logger.log_agent_action(
                    self, name,
                    action_data['prompt'],
                    action_data['response'],
                    action_data['action'],
                    action_data['message'],
                    action_data['api_call_time'],
                    actual_success,  # Use the actual execution outcome.
                    action_data['view'],
                    action_data['usage'],
                    extra={
                        'messages_received': action_data.get('messages_received', []),
                        'subactions': action_data.get('subactions', {}),
                        'interaction_request': action_data.get('interaction_request'),
                        'interaction_result': action_data.get('interaction_result'),
                        'capacity_snapshot': action_data.get('capacity_snapshot'),
                        'action_trace': action_data.get('action_trace', [])
                    }
                )
            # Clear deferred actions.
            self.delayed_actions = {}

        # Summarize current states of all agents.
        print(f"DEBUG: [STATE_SUMMARY] === Round {self.round} Agent State Summary ===")
        for name, mesh in self.agent_meshes.items():
            pos = mesh.pos
            carrying_item = self.tasks[self.task].resource_state.get(name)
            status = f"Carrying: {carrying_item}" if carrying_item else "Empty-handed"
            print(f"DEBUG: [STATE_SUMMARY] {name} @ x={pos[0]}, y={pos[1]} - {status}")
        print(f"DEBUG: [STATE_SUMMARY] ================================")

        # 6. Update task logic and logs.
        self.tasks[self.task].update(self, self.actions)
        for name, mesh in self.agent_meshes.items():
            i, j = mesh.pos[0], mesh.pos[1]
            if not (0 <= i < self.height and 0 <= j < self.width):
                mesh.remove(self.grid, self.mesh_map)
                self.agents[name].escaped = True
        self.round += 1
        self.logger.log_game_state(self, self.round, time.time())
        print(self.render(use_color=True))  # Use colors in terminal output.
        self.actions = {}

    def get_view(self, name):
        if name not in self.agent_meshes:
            return None
        x, y = self.agent_meshes[name].pos[1], self.agent_meshes[name].pos[0]
        view_size = self.view_size
        half_view = view_size // 2

        view = np.full((view_size, view_size), '.', dtype=self.grid_dtype)

        # Dynamically build a temporary grid for rendering
        temp_grid = self._render_grid_state()

        for i in range(view_size):
            for j in range(view_size):
                grid_x = x + (j - half_view)
                grid_y = y + (i - half_view)

                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    # Render normally within map bounds.
                    view[i, j] = temp_grid[grid_y, grid_x]
                else:
                    # Outside the map, render '*' only in the first ring beyond the walls; leave farther cells blank.
                    is_one_ring_outside = (
                        (-1 <= grid_x <= self.width) and (-1 <= grid_y <= self.height)
                        and not (0 <= grid_x < self.width and 0 <= grid_y < self.height)
                    )
                    view[i, j] = '*' if is_one_ring_outside else ''

        return view

    def _render_grid_state(self):
        temp_grid = np.full_like(self.grid, '.', dtype=object)
        # Map grid positions to mesh types to resolve symbol conflicts.
        self._grid_mesh_types = {}

        # Flag for debugging a specific position only.
        debug_specific_positions = getattr(self, '_debug_put_down_positions', set())

        # Prefer mesh_map rendering; fall back to permanent terrain in terrain_adapter when no mesh exists.
        for i in range(self.height):
            for j in range(self.width):
                mesh = self.mesh_map[i, j]
                pos_key = (i, j)

                if mesh is not None:
                    # Detailed diagnostics only for PUT_DOWN positions.
                    if pos_key in debug_specific_positions:
                        print(f"DEBUG: [RENDER] 🎯PUT_DOWN position {pos_key} - mesh type: {type(mesh).__name__}, name: {mesh.name}")
                        print(f"DEBUG: [RENDER] 🎯PUT_DOWN position {pos_key} - is EnhancedCapacityMesh: {isinstance(mesh, EnhancedCapacityMesh)}")
                        if hasattr(mesh, 'items'):
                            print(f"DEBUG: [RENDER] 🎯PUT_DOWN position {pos_key} - items: {mesh.items}")
                        if hasattr(mesh, 'shape'):
                            print(f"DEBUG: [RENDER] 🎯PUT_DOWN position {pos_key} - shape: {mesh.shape}")

                    # Determine the display name.
                    display_name = mesh.name  # Default to the mesh name.

                    # Use get_display_name for CapacityMesh objects.
                    if isinstance(mesh, CapacityMesh) and hasattr(mesh, 'get_display_name'):
                        display_name = mesh.get_display_name()

                    # Update grid display.
                    temp_grid[i, j] = display_name

                    # Record actual mesh type for coloring, preferring item type.
                    mesh_type_for_color = mesh.name
                    if hasattr(mesh, 'items') and mesh.items:
                        # When items exist, use the first item's type for coloring.
                        first_item = mesh.items[0]
                        mesh_type_for_color = first_item.item_type
                    self._grid_mesh_types[pos_key] = mesh_type_for_color
                elif pos_key in debug_specific_positions:
                    print(f"DEBUG: [RENDER] 🎯PUT_DOWN position {pos_key} - no mesh; using default '.'")

                # Fall back to terrain_adapter display when the mesh is absent but permanent terrain exists.
                if mesh is None and hasattr(self, 'terrain_adapter') and self.terrain_adapter:
                    terrain = self.terrain_adapter.get_terrain_at(pos_key)
                    if terrain is not None:
                        display_name = terrain.get_display_name() if hasattr(terrain, 'get_display_name') else terrain.name
                        temp_grid[i, j] = display_name
                        # Record type for coloring.
                        mesh_type_for_color = terrain.name
                        if hasattr(terrain, 'items') and terrain.items:
                            first_item = terrain.items[0]
                            mesh_type_for_color = first_item.item_type
                        self._grid_mesh_types[pos_key] = mesh_type_for_color

        # Add agent display.
        for agent_name, agent_mesh in self.agent_meshes.items():
            x_a, y_a = agent_mesh.pos[1], agent_mesh.pos[0]
            if 0 <= y_a < self.height and 0 <= x_a < self.width:
                temp_grid[y_a, x_a] = agent_mesh.shape[0, 0]

        return temp_grid

    def _get_render_score_text(self):
        """Return a compact score string for human-facing render panels."""
        try:
            task_obj = self.tasks.get(self.task) if isinstance(self.tasks, dict) else None
            score = getattr(task_obj, 'score', None)
            if score is None:
                return None
            if isinstance(score, (int, np.integer)):
                return str(int(score))
            if isinstance(score, (float, np.floating)):
                score_f = float(score)
                if score_f.is_integer():
                    return str(int(score_f))
                return f"{score_f:.2f}".rstrip('0').rstrip('.')
            return str(score)
        except Exception:
            return None

    def _extract_render_char(self, cell):
        """Collapse a rendered cell to a single display character."""
        if cell is None:
            return '.'
        text = str(cell)
        if text == '':
            return ' '
        return text[0]

    def _decorate_render_char(self, display_char, mesh_type=None, use_color=True, emphasize=False, rendered_symbol=None):
        """Apply existing terrain/resource colors to a visible tile string."""
        rendered = rendered_symbol if rendered_symbol is not None else display_char
        if not use_color or display_char == ' ':
            return rendered

        if display_char == 'W':
            rendered = Back.RED + rendered + Style.RESET_ALL
        elif display_char == 'M':
            rendered = Back.YELLOW + rendered + Style.RESET_ALL
        elif display_char == 'B':
            rendered = Back.YELLOW + rendered + Style.RESET_ALL
        elif display_char == 'G':
            rendered = Back.GREEN + rendered + Style.RESET_ALL
        elif display_char == 'F':
            rendered = Back.LIGHTGREEN_EX + rendered + Style.RESET_ALL
        elif display_char == 'f':
            rendered = Back.LIGHTCYAN_EX + rendered + Style.RESET_ALL
        elif display_char == 'T':
            rendered = Back.MAGENTA + rendered + Style.RESET_ALL
        elif display_char == 'w':
            if mesh_type == 'wood':
                rendered = Back.YELLOW + Fore.BLACK + rendered + Style.RESET_ALL
            else:
                rendered = Back.BLUE + rendered + Style.RESET_ALL
        elif display_char == 'R':
            rendered = Back.BLUE + rendered + Style.RESET_ALL
        elif display_char == 's':
            if mesh_type == 'sand':
                rendered = Back.LIGHTYELLOW_EX + Fore.BLACK + rendered + Style.RESET_ALL
            elif mesh_type == 'stone':
                rendered = Back.WHITE + Fore.BLACK + rendered + Style.RESET_ALL
        elif display_char == 'S' or mesh_type == 'stone':
            rendered = Back.WHITE + Fore.BLACK + rendered + Style.RESET_ALL

        if emphasize:
            rendered = Fore.CYAN + Style.BRIGHT + rendered + Style.RESET_ALL
        return rendered

    def _get_legend_items(self, seen_symbols):
        """Build a stable legend, filtering out absent symbols."""
        specs = [
            (('a', 'A'), lambda syms: ('a' if 'a' in syms else 'A') + " Agent"),
            (('T',), lambda syms: "T Table"),
            (('W',), lambda syms: "W Wall"),
            (('G',), lambda syms: "G Grass"),
            (('F',), lambda syms: "F Forest"),
            (('M',), lambda syms: "M Mountain"),
            (('B',), lambda syms: "B Block"),
            (('w',), lambda syms: "w Wood"),
            (('s',), lambda syms: "s Sand"),
            (('S',), lambda syms: "S Stone"),
            (('R',), lambda syms: "R River"),
            (('f',), lambda syms: "f Food"),
            (('.',), lambda syms: ". Empty"),
            (('*',), lambda syms: "* Out"),
        ]
        legend_items = []
        for candidates, formatter in specs:
            if any(symbol in seen_symbols for symbol in candidates):
                legend_items.append(formatter(seen_symbols))
        return legend_items

    def _format_tactical_axis_value(self, value):
        """Format x-axis values in a compact tactical style."""
        try:
            num = int(value)
            if 0 <= num < 100:
                return f"{num:02d}"
            return str(num)
        except Exception:
            return str(value)

    def _format_tactical_row_label(self, value):
        """Format y-axis labels as padded numbers without a y-prefix."""
        try:
            num = int(value)
            if 0 <= num < 100:
                return f"{num:02d}"
            return str(num)
        except Exception:
            return str(value)

    def _stylize_render_char(self, symbol):
        """Render letters in a different typographic style while keeping the same semantics."""
        style_map = {
            'W': '𝚆',
            'M': '𝙼',
            'B': '𝙱',
            'G': '𝙶',
            'F': '𝙵',
            'T': '𝚃',
            'R': '𝚁',
            'S': '𝚂',
            'a': '𝚊',
            'w': '𝚠',
            's': '𝚜',
            'f': '𝚏',
        }
        return style_map.get(symbol, symbol)

    def _format_symbol_tile(self, symbol, cell_width=3):
        """Render a symbol with width that adapts to the current map."""
        if symbol == ' ':
            return " " * cell_width
        return f"{symbol:^{cell_width}}"

    def _wrap_panel_items(self, prefix, items, max_width):
        """Wrap pipe-delimited items without changing visible semantics."""
        if not items:
            return []
        lines = []
        continuation_prefix = ' ' * len(prefix)
        current = prefix
        for item in items:
            proposal = f"{current}{item}" if current == prefix else f"{current} | {item}"
            if len(proposal) <= max_width or current == prefix:
                current = proposal
            else:
                lines.append(current)
                current = f"{continuation_prefix}{item}"
        lines.append(current)
        return lines

    def _build_tactical_map_lines(
        self,
        matrix,
        row_labels,
        col_labels,
        use_color=True,
        mesh_type_lookup=None,
    ):
        """Build the left tactical-map pane as plain/decorated line pairs."""
        rows = int(matrix.shape[0]) if hasattr(matrix, 'shape') else len(matrix)
        cols = int(matrix.shape[1]) if hasattr(matrix, 'shape') and len(matrix.shape) > 1 else (len(matrix[0]) if rows else 0)
        mesh_type_lookup = mesh_type_lookup or {}

        row_texts = [self._format_tactical_row_label(v) for v in row_labels]
        row_label_width = max((len(text) for text in row_texts), default=3)
        col_texts = [self._format_tactical_axis_value(v) for v in col_labels]
        cell_width = max(3, max((len(text) for text in col_texts), default=1) + 1)
        coord_plain = f"{'':<{row_label_width}}  " + ''.join(f"{text:>{cell_width}}" for text in col_texts)
        coord_decorated = coord_plain
        guide_plain = f"{'':<{row_label_width}}  " + ''.join('─' * cell_width for _ in col_texts)
        guide_decorated = guide_plain

        line_pairs = []
        seen_symbols = set()

        for row_idx in range(rows):
            plain_cells = []
            decorated_cells = []
            for col_idx in range(cols):
                raw_cell = matrix[row_idx, col_idx] if hasattr(matrix, 'shape') else matrix[row_idx][col_idx]
                display_char = self._extract_render_char(raw_cell)
                if display_char != ' ':
                    seen_symbols.add(display_char)
                mesh_type = mesh_type_lookup.get((row_idx, col_idx))
                styled_char = self._stylize_render_char(display_char)
                tile_plain = self._format_symbol_tile(styled_char, cell_width=cell_width)
                tile_decorated = self._decorate_render_char(
                    display_char,
                    mesh_type=mesh_type,
                    use_color=use_color,
                    rendered_symbol=tile_plain,
                )
                plain_cells.append(tile_plain)
                decorated_cells.append(tile_decorated)
            row_prefix_plain = f"{row_texts[row_idx]:<{row_label_width}}  "
            row_prefix_decorated = row_prefix_plain
            line_pairs.append((row_prefix_plain + ''.join(plain_cells), row_prefix_decorated + ''.join(decorated_cells)))

        line_pairs.append((guide_plain, guide_decorated))
        line_pairs.append((coord_plain, coord_decorated))

        return line_pairs, seen_symbols

    def _compact_legend_items(self, legend_items, columns=2, gap=3):
        """Compact legend rows to keep the sidebar shorter on small maps."""
        legend_items = list(legend_items or [])
        if not legend_items:
            return []
        if len(legend_items) <= columns:
            return legend_items

        rows = []
        chunks = [legend_items[idx:idx + columns] for idx in range(0, len(legend_items), columns)]
        first_col_width = max((len(chunk[0]) for chunk in chunks if chunk), default=0)
        for chunk in chunks:
            if len(chunk) == 1:
                rows.append(chunk[0])
            else:
                rows.append(f"{chunk[0]:<{first_col_width}}{' ' * gap}{chunk[1]}")
        return rows

    def _get_table_status_lines(self):
        """Collect compact per-table capacity summaries for the global HUD."""
        table_lines = []
        tables = []
        for mesh in getattr(self, 'meshes', []) or []:
            if not isinstance(mesh, CapacityMesh):
                continue
            if getattr(mesh, 'name', None) != 'T':
                continue
            try:
                mesh_x, mesh_y = int(mesh.pos[1]), int(mesh.pos[0])
                total_qty = 0
                if hasattr(mesh, 'items') and mesh.items:
                    for it in mesh.items:
                        total_qty += int(getattr(it, 'quantity', 0) or 0)
                capacity = int(getattr(mesh, 'capacity', 0) or 0)
                remaining = max(0, capacity - total_qty)
                out_x, out_y = to_math_xy(mesh_x, mesh_y, self.height)
                remaining_by_type = None
                if hasattr(mesh, 'get_remaining_by_type') and (
                    getattr(mesh, 'capacity_by_type', None) or getattr(mesh, 'per_type_default', None) is not None
                ):
                    remaining_by_type = mesh.get_remaining_by_type(['wood', 'sand', 'stone'])
                tables.append((int(out_y), int(out_x), int(remaining), remaining_by_type or {}))
            except Exception:
                continue

        tables.sort(key=lambda item: (-item[0], item[1]))
        for out_y, out_x, remaining, remaining_by_type in tables:
            table_lines.append(f"T@({out_x},{out_y}) total={remaining}")
            parts = []
            for resource_key in ('wood', 'sand', 'stone'):
                if resource_key in remaining_by_type:
                    parts.append(f"{resource_key}={remaining_by_type.get(resource_key)}")
            if parts:
                table_lines.append(", ".join(parts))
        return table_lines

    def _build_sidebar_card(self, title, lines, width):
        """Render a compact HUD card with its own inner border."""
        width = max(width, len(title) + 4)
        title_text = f" {title} "
        top = "╭" + title_text + ("─" * max(0, width - len(title_text))) + "╮"
        body = [f"│ {line:<{width - 2}} │" for line in lines] if lines else [f"│ {'':<{width - 2}} │"]
        bottom = "╰" + ("─" * width) + "╯"
        return [top] + body + [bottom]

    def _build_sidebar_lines(self, info_pairs=None, legend_items=None, notes=None, table_status_lines=None):
        """Build the right tactical HUD pane."""
        info_pairs = info_pairs or []
        legend_items = self._compact_legend_items(legend_items or [], columns=2, gap=4)
        table_status_lines = [str(line) for line in (table_status_lines or []) if line]
        notes = [str(note) for note in (notes or []) if note]

        mission_lines = []
        for label, value in info_pairs:
            if value is None or value == '':
                continue
            mission_lines.append(f"{label:<8} {value}")

        cards = []
        max_content_width = 18
        for collection in (mission_lines, legend_items, table_status_lines, notes):
            for item in collection:
                max_content_width = max(max_content_width, len(str(item)) + 2)

        if mission_lines:
            cards.extend(self._build_sidebar_card("MISSION DATA", mission_lines, max_content_width))
        if legend_items:
            if cards:
                cards.append("")
            cards.extend(self._build_sidebar_card("LEGEND", legend_items, max_content_width))
        if table_status_lines:
            if cards:
                cards.append("")
            cards.extend(self._build_sidebar_card("TABLES STATUS", table_status_lines, max_content_width))
        if notes:
            if cards:
                cards.append("")
            cards.extend(self._build_sidebar_card("NOTES", notes, max_content_width))

        return cards

    def _make_titled_border_segment(self, width, title):
        """Create a top-border segment with an inline title."""
        total = max(1, int(width))
        if not title:
            return "─" * total
        title_text = f"─ {title} "
        if len(title_text) >= total:
            return title_text[:total]
        return title_text + ("─" * (total - len(title_text)))

    def _render_tactical_dashboard(
        self,
        matrix,
        title,
        row_labels,
        col_labels,
        use_color=True,
        mesh_type_lookup=None,
        hud_title="TACTICAL HUD",
        info_pairs=None,
        notes=None,
        table_status_lines=None,
    ):
        """Render a human-facing tactical layout: left map, right telemetry."""
        def _center_pad_pairs(pairs, target_len):
            diff = max(0, target_len - len(pairs))
            top = diff // 2
            bottom = diff - top
            return [("", "")] * top + pairs + [("", "")] * bottom

        def _center_pad_lines(lines, target_len):
            diff = max(0, target_len - len(lines))
            top = diff // 2
            bottom = diff - top
            return [""] * top + lines + [""] * bottom

        map_pairs, seen_symbols = self._build_tactical_map_lines(
            matrix,
            row_labels,
            col_labels,
            use_color=use_color,
            mesh_type_lookup=mesh_type_lookup,
        )
        legend_items = self._get_legend_items(seen_symbols)
        sidebar_lines = self._build_sidebar_lines(
            info_pairs=info_pairs,
            legend_items=legend_items,
            notes=notes,
            table_status_lines=table_status_lines,
        )

        left_width = max((len(plain) for plain, _ in map_pairs), default=0)
        right_width = max((len(line) for line in sidebar_lines), default=0)
        terminal_width = shutil.get_terminal_size(fallback=(120, 40)).columns
        use_stacked_layout = right_width > 0 and (left_width + right_width + 7 > terminal_width)

        if use_stacked_layout:
            lines = ["┌" + self._make_titled_border_segment(left_width + 2, title) + "┐"]
            for left_plain, left_decorated in map_pairs:
                left_pad = left_width - len(left_plain)
                lines.append(f"│ {left_decorated}{' ' * left_pad} │")
            lines.append("└" + ("─" * (left_width + 2)) + "┘")

            if sidebar_lines:
                lines.append("┌" + self._make_titled_border_segment(right_width + 2, hud_title) + "┐")
                for line in sidebar_lines:
                    right_pad = right_width - len(line)
                    lines.append(f"│ {line}{' ' * right_pad} │")
                lines.append("└" + ("─" * (right_width + 2)) + "┘")
            return "\n".join(lines) + "\n"

        top_border = (
            "┌"
            + self._make_titled_border_segment(left_width + 2, title)
            + "┬"
            + self._make_titled_border_segment(right_width + 2, hud_title)
            + "┐"
        )
        lines = [top_border]
        total_rows = max(len(map_pairs), len(sidebar_lines))
        map_pairs = _center_pad_pairs(map_pairs, total_rows)
        sidebar_lines = _center_pad_lines(sidebar_lines, total_rows)
        for row_idx in range(total_rows):
            left_plain, left_decorated = map_pairs[row_idx]
            right_plain = right_decorated = sidebar_lines[row_idx]

            left_pad = left_width - len(left_plain)
            right_pad = right_width - len(right_plain)
            lines.append(
                f"│ {left_decorated}{' ' * left_pad} │ {right_decorated}{' ' * right_pad} │"
            )

        lines.append("└" + ("─" * (left_width + 2)) + "┴" + ("─" * (right_width + 2)) + "┘")
        return "\n".join(lines) + "\n"

    def render_for_file(self, name=None):
        """Render without colors for file output."""
        return self.render(name=name, use_color=False)

    def render_no_color(self, name=None):
        """Compatibility alias for callers expecting an explicit no-color render path."""
        return self.render(name=name, use_color=False)

    def render(self, name=None, use_color=True):
        if name:
            view = self.get_view(name)
            if view is None:
                return "Agent not found"
            x_center, y_center = self.agent_meshes[name].pos[1], self.agent_meshes[name].pos[0]
            half_view = view.shape[0] // 2
            mesh_type_lookup = {}
            for row_idx in range(view.shape[0]):
                for col_idx in range(view.shape[1]):
                    grid_x = x_center + (col_idx - half_view)
                    grid_y = y_center + (row_idx - half_view)
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        mesh_type_lookup[(row_idx, col_idx)] = self._grid_mesh_types.get((grid_y, grid_x))
            return self._render_tactical_dashboard(
                view,
                title="SENSOR WINDOW",
                row_labels=list(range(view.shape[0])),
                col_labels=list(range(view.shape[1])),
                use_color=use_color,
                mesh_type_lookup=mesh_type_lookup,
                hud_title="LOCAL HUD",
                info_pairs=[
                    ("Agent", name),
                    ("Round", self.round),
                    ("Window", f"{view.shape[1]}x{view.shape[0]}"),
                    ("Scope", "Local"),
                ],
                notes=None,
            )

        temp_grid = self._render_grid_state()
        score_text = self._get_render_score_text()
        table_status_lines = self._get_table_status_lines()
        info_pairs = [
            ("Round", self.round),
            ("Size", f"{self.width}x{self.height}"),
        ]
        if score_text is not None:
            info_pairs.append(("Score", score_text))
        info_pairs.extend([
            ("Agents", len(self.agent_meshes)),
            ("Msgs", len(self.messages)),
        ])
        return self._render_tactical_dashboard(
            temp_grid,
            title="TACTICAL MAP",
            row_labels=[self.height - 1 - i for i in range(self.height)],
            col_labels=list(range(self.width)),
            use_color=use_color,
            mesh_type_lookup=self._grid_mesh_types,
            hud_title="MISSION HUD",
            info_pairs=info_pairs,
            notes=None,
            table_status_lines=table_status_lines,
        )


if __name__ == '__main__':
    grid = np.full((10, 10), '.', dtype=str)
    shape_h = np.array(
        [['B', 'B', 'B', 'B']]
    )
    shape_v = np.array(
        [['B'], ['B'], ['B'], ['B']]
    )
    meshes = [
        Mesh(pos=(1, 1), shape=shape_h, static=False),
        Mesh(pos=(4, 1), shape=shape_v, static=False)
    ]
    for mesh in meshes:
        grid[mesh.pos[0]:mesh.pos[0] + mesh.shape.shape[0], mesh.pos[1]:mesh.pos[1] + mesh.shape[1]] = mesh
    print(grid)


SwarmEnvironment = MASEnvironment
