import os
from framework.framework import Framework, ModelConfig
from masbench.agent import MASAgent
from masbench.environment import MASEnvironment
from masbench.logger import MASLogger
from masbench.mechanisms import build_effective_config, get_preset
from contextlib import contextmanager

from queue import Queue
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

stdout = sys.stdout


@contextmanager
def silence():
    # Save the current stdout
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Redirect stdout to devnull
    yield  # Allow the code to run
    sys.stdout = original_stdout  # Restore stdout


def output(s):
    stdout.write(f'{s}')


class MASFramework:

    instances = {}
    submission = {}

    def __init__(self, name=''):
        self.name = name
        self.framework = None

    @property
    def status(self):
        if self.framework is None or self.framework.env is None:
            return 'pending'
        if self.framework.env.done:
            return 'done'
        return 'running'

    def run_task(self, model, task, log_dir=None,
                  num_agents=10, max_round=100, width=12, height=12, seed=42, view_size=9,
                  communication_size=15, food_capacities=None, nest_capacities=None,
                  custom_map=None,
                  custom_map_path=None,
                  message_phase_enabled=True,
                  messages_enabled=True,
                  message_phase_timeout_s=60,
                  message_phase_concurrency=None,
                  explicit_interaction_enabled=True,
                  auto_clamp=False,
                  agent_view_memory=None,
                  agent_history_memory=None,
                  # === Memory, protocol, and rate-limiting mechanisms ===
                  enable_structured_view_memory=False,
                  view_memory_k=5,
                  view_memory_max_lines=12,
                  enable_message_memory=False,
                  message_memory_k_sent=3,
                  message_memory_k_recv=3,
                  single_message_per_round=False,
                  single_message_prompt_mode='basic',
                  ring_mode='angle_halves',
                  # Protocol V2.
                  protocol_v2_enabled=False,
                  protocol_v2_auto=True,
                  ask_cooldown_rounds=2,
                  max_items_in_message=5,
                  max_tables_in_message=3,
                  # Protocol V2: free-text portion and header length limits.
                  proto_v2_free_enable=False,
                  proto_v2_free_slot='C',
                  proto_v2_free_max=150,
                  proto_v2_header_max_len=0,
                  proto_v2_total_max_len=220,
                  # Agent carrying capacity override.
                  agent_carrying_capacity=None,
                  mechanism_preset='base',
                  mechanism_raw_config=None):
        if self.status != 'pending':
            raise RuntimeError(f'Cannot run task because task is already {self.status}.')
        env_name = self.name
        mechanism_preset = str(mechanism_preset or 'base')
        mechanism_raw_config = dict(mechanism_raw_config or {})
        mechanism_effective_config = build_effective_config(mechanism_preset, mechanism_raw_config)
        memory_config = mechanism_effective_config.get('memory', {}) or {}
        routing_config = mechanism_effective_config.get('routing', {}) or {}
        protocol_config = mechanism_effective_config.get('protocol', {}) or {}
        runtime_config = mechanism_effective_config.get('runtime', {}) or {}
        base_no_mechanisms = mechanism_preset == 'base'

        def _cfg_bool(config, key, fallback):
            return bool(config.get(key, fallback))

        def _cfg_int(config, key, fallback):
            try:
                return int(config.get(key, fallback))
            except Exception:
                return int(fallback)

        def _cfg_str(config, key, fallback):
            value = config.get(key, fallback)
            return str(value if value is not None else fallback)

        effective_structured_view_memory = False if base_no_mechanisms else _cfg_bool(memory_config, 'enable_structured_view_memory', enable_structured_view_memory)
        effective_view_memory_k = _cfg_int(memory_config, 'view_memory_k', view_memory_k)
        effective_view_memory_max_lines = _cfg_int(memory_config, 'view_memory_max_lines', view_memory_max_lines)
        effective_message_memory = False if base_no_mechanisms else _cfg_bool(memory_config, 'enable_message_memory', enable_message_memory)
        effective_message_memory_k_sent = _cfg_int(memory_config, 'message_memory_k_sent', message_memory_k_sent)
        effective_message_memory_k_recv = _cfg_int(memory_config, 'message_memory_k_recv', message_memory_k_recv)

        effective_single_message_per_round = False if base_no_mechanisms else _cfg_bool(routing_config, 'single_message_per_round', single_message_per_round)
        effective_single_message_prompt_mode = _cfg_str(routing_config, 'single_message_prompt_mode', single_message_prompt_mode)
        effective_ring_mode = _cfg_str(routing_config, 'ring_mode', ring_mode)
        effective_ask_cooldown_rounds = _cfg_int(routing_config, 'ask_cooldown_rounds', ask_cooldown_rounds)

        effective_protocol_v2_enabled = False if base_no_mechanisms else _cfg_bool(protocol_config, 'protocol_v2_enabled', protocol_v2_enabled)
        effective_protocol_v2_auto = False if base_no_mechanisms else _cfg_bool(protocol_config, 'protocol_v2_auto', protocol_v2_auto)
        effective_protocol_ask_cooldown_rounds = _cfg_int(protocol_config, 'ask_cooldown_rounds', effective_ask_cooldown_rounds)
        effective_max_items_in_message = _cfg_int(protocol_config, 'max_items_in_message', max_items_in_message)
        effective_max_tables_in_message = _cfg_int(protocol_config, 'max_tables_in_message', max_tables_in_message)
        effective_proto_v2_free_enable = False if base_no_mechanisms else _cfg_bool(protocol_config, 'proto_v2_free_enable', proto_v2_free_enable)
        effective_proto_v2_free_slot = _cfg_str(protocol_config, 'proto_v2_free_slot', proto_v2_free_slot)
        effective_proto_v2_free_max = _cfg_int(protocol_config, 'proto_v2_free_max', proto_v2_free_max)
        effective_proto_v2_header_max_len = _cfg_int(protocol_config, 'proto_v2_header_max_len', proto_v2_header_max_len)
        effective_proto_v2_total_max_len = _cfg_int(protocol_config, 'proto_v2_total_max_len', proto_v2_total_max_len)
        effective_messages_enabled = False if base_no_mechanisms else (bool(messages_enabled) and _cfg_bool(runtime_config, 'messages_enabled', True))
        effective_message_phase_enabled = False if base_no_mechanisms else (bool(message_phase_enabled) and effective_messages_enabled)

        # Prepare agent parameters
        sys_prompt = "You are a agent. You need to cooperate with other agents and finish a given task."
        agent_args = {"sys_prompt": sys_prompt}

        # Enable enhanced inventory for all agents; read capacity from override/env/json
        try:
            capacity_from_map = None
            if isinstance(custom_map, dict):
                map_cfg = custom_map.get('map', {}) or {}
                agent_cfg = map_cfg.get('agent_config', {}) or {}
                item_number = agent_cfg.get('item_number')
                if isinstance(item_number, int) and item_number > 0:
                    capacity_from_map = item_number
            if agent_carrying_capacity is not None and int(agent_carrying_capacity) > 0:
                carrying_capacity = int(agent_carrying_capacity)
                cap_src = 'override'
            elif capacity_from_map is not None:
                carrying_capacity = int(capacity_from_map)
                cap_src = 'json'
            else:
                carrying_capacity = 4
                cap_src = 'default'
            agent_args.update({
                'enable_inventory': True,
                'carrying_capacity': carrying_capacity
            })
            print(f"DEBUG: MASFramework.run_task - enhanced inventory enabled: True, carrying capacity: {carrying_capacity} (src={cap_src})")
        except Exception as e:
            print(f"WARNING: MASFramework.run_task - failed to parse custom-map carrying capacity: {e}; using default enhanced configuration")
            agent_args.update({
                'enable_inventory': True,
                'carrying_capacity': 4
            })

        # Configure view and action history windows; callers may override them.
        vm = 0 if agent_view_memory is None else int(agent_view_memory)
        hm = 1 if agent_history_memory is None else int(agent_history_memory)
        if base_no_mechanisms:
            vm = 0
            hm = 0
        agent_args.update({'view_memory': vm, 'history_memory': hm})
        print(f"DEBUG: MASFramework.run_task - memory windows: view_memory={vm}, history_memory={hm}")

        # Pass memory/protocol mechanism parameters to the agent.
        agent_args.update({
            'enable_structured_view_memory': effective_structured_view_memory,
            'view_memory_k': effective_view_memory_k,
            'view_memory_max_lines': effective_view_memory_max_lines,
            'enable_message_memory': effective_message_memory,
            'mechanism_preset': mechanism_preset,
            'mechanism_raw_config': mechanism_raw_config,
        })
        print(f"DEBUG: MASFramework.run_task - structured view memory={effective_structured_view_memory}, K={effective_view_memory_k}, max_lines={effective_view_memory_max_lines}")
        print(f"DEBUG: MASFramework.run_task - message memory={effective_message_memory}")
        print(f"DEBUG: MASFramework.run_task - Protocol V2: enabled={effective_protocol_v2_enabled}, auto={effective_protocol_v2_auto}, cooldown={effective_protocol_ask_cooldown_rounds}, max_items={effective_max_items_in_message}, max_tables={effective_max_tables_in_message}, total_max={effective_proto_v2_total_max_len}")
        print(f"DEBUG: MASFramework.run_task - one message per round: enabled={effective_single_message_per_round}, prompt_mode={effective_single_message_prompt_mode}, ring_mode={effective_ring_mode}")

        try:
            preset_spec = get_preset(mechanism_preset)
            mechanism_classes = {
                'memory': preset_spec.memory_cls.__name__,
                'routing': preset_spec.routing_cls.__name__,
                'protocol': preset_spec.protocol_cls.__name__,
            }
        except Exception:
            mechanism_classes = {}

        # Prepare logger parameters
        meta = {
            'model': model.model if isinstance(model, ModelConfig) else [m.model for m in model],
            'num_agents': num_agents,
            'max_round': max_round,
            'width': width,
            'height': height,
            'seed': seed,
            'view_size': view_size,
            'communication_size': communication_size,  # Record the additional parameter.
            'enhanced_inventory': True,
            'carrying_capacity': agent_args.get('carrying_capacity'),
            'mechanism_preset': mechanism_preset,
            'mechanism_classes': mechanism_classes,
            'mechanism_raw_config': mechanism_raw_config,
            'mechanism_effective_config': mechanism_effective_config,
            'message_phase_enabled': effective_message_phase_enabled,
            'messages_enabled': effective_messages_enabled,
        }
        # Record the map filename, if provided.
        try:
            import os as _os
            meta['map'] = _os.path.basename(str(custom_map_path)) if custom_map_path else None
        except Exception:
            meta['map'] = None
        # Attach task requirements from the custom map for comparison with final deliveries.
        try:
            if isinstance(custom_map, dict):
                task_cfg = custom_map.get('task')
                if isinstance(task_cfg, dict) and task_cfg:
                    meta['task_config'] = task_cfg
        except Exception:
            pass

        print(f"DEBUG: MASFramework.run_task - parameters: observation_range={view_size}, communication_range={communication_size}")
        logger_args = {"log_dir": log_dir, "meta": meta}

        # Prepare environment parameters
        env_args = {"task": task, "seed": seed, "max_round": max_round,
                    'width': width, 'height': height, 'view_size': view_size,
                    'communication_size': communication_size,  # Pass the additional parameter.
                    'food_capacities': food_capacities, 'nest_capacities': nest_capacities,
                    'custom_map': custom_map,
                    'message_phase_enabled': effective_message_phase_enabled,
                    'messages_enabled': effective_messages_enabled,
                    'message_phase_timeout_s': message_phase_timeout_s,
                    'message_phase_concurrency': message_phase_concurrency,
                    'explicit_interaction_enabled': explicit_interaction_enabled,
                    'auto_clamp': auto_clamp,
                    # Message memory and one-message-per-round mode.
                    'enable_message_memory': effective_message_memory,
                    'message_memory_k_sent': effective_message_memory_k_sent,
                    'message_memory_k_recv': effective_message_memory_k_recv,
                    'single_message_per_round': effective_single_message_per_round,
                    'single_message_prompt_mode': effective_single_message_prompt_mode,
                    'ring_mode': effective_ring_mode,
                    # Protocol V2.
                    'protocol_v2_enabled': effective_protocol_v2_enabled,
                    'protocol_v2_auto': effective_protocol_v2_auto,
                    'ask_cooldown_rounds': effective_protocol_ask_cooldown_rounds,
                    'max_items_in_message': effective_max_items_in_message,
                    'max_tables_in_message': effective_max_tables_in_message,
                    # Protocol V2 free text.
                    'proto_v2_free_enable': effective_proto_v2_free_enable,
                    'proto_v2_free_slot': effective_proto_v2_free_slot,
                    'proto_v2_free_max': effective_proto_v2_free_max,
                    'proto_v2_header_max_len': effective_proto_v2_header_max_len,
                    'proto_v2_total_max_len': effective_proto_v2_total_max_len,
                    'mechanism_preset': mechanism_preset,
                    'mechanism_raw_config': mechanism_raw_config,
                    'mechanism_effective_config': mechanism_effective_config,
                    }

        self.framework = Framework()
        # Use the framework to start the game
        self.framework.start(
            agent_cls=MASAgent,
            env_cls=MASEnvironment,
            logger_cls=MASLogger,
            env_name=env_name,
            num_agents=num_agents,
            model=model,
            agent_args=agent_args,
            logger_args=logger_args,
            env_args=env_args
        )

    @classmethod
    def model_config(cls, model, api_key, api_base, provider: str | None = None):
        cfg = ModelConfig()
        cfg.api_key = api_key
        cfg.api_base = api_base
        cfg.model = model
        try:
            if provider is not None:
                cfg.provider = str(provider)
        except Exception:
            pass
        return cfg

    @classmethod
    def submit(cls, name, model, task, log_dir=None,
               num_agents=10, max_round=100, width=12, height=12, seed=42, view_size=9,
               communication_size=15, food_capacities=None, nest_capacities=None,
               custom_map=None,
               custom_map_path=None,
               message_phase_enabled=True,
               messages_enabled=True,
               message_phase_timeout_s=60,
               message_phase_concurrency=None,
               explicit_interaction_enabled=True,
               auto_clamp=False,
               agent_view_memory=None,
               agent_history_memory=None,
               # === Memory, protocol, and rate-limiting mechanisms ===
               enable_structured_view_memory=False,
               view_memory_k=5,
               view_memory_max_lines=12,
               enable_message_memory=False,
               message_memory_k_sent=3,
               message_memory_k_recv=3,
               single_message_per_round=False,
               single_message_prompt_mode='basic',
               ring_mode='angle_halves',
               # Protocol V2.
               protocol_v2_enabled=False,
               protocol_v2_auto=True,
               ask_cooldown_rounds=2,
               max_items_in_message=5,
               max_tables_in_message=3,
               # Protocol V2: free-text portion and header length limits.
               proto_v2_free_enable=False,
               proto_v2_free_slot='C',
               proto_v2_free_max=150,
               proto_v2_header_max_len=0,
               proto_v2_total_max_len=220,
               # Agent carrying capacity override.
               agent_carrying_capacity=None,
               mechanism_preset='base',
               mechanism_raw_config=None):
        kwargs = {
            'model': model,
            'task': task,
            'log_dir': log_dir,
            'num_agents': num_agents,
            'max_round': max_round,
            'width': width,
            'height': height,
            'seed': seed,
            'view_size': view_size,
            'communication_size': communication_size,  # Pass the additional parameter.
            'food_capacities': food_capacities,
            'nest_capacities': nest_capacities,
            'custom_map': custom_map,
            'custom_map_path': custom_map_path,
            'message_phase_enabled': message_phase_enabled,
            'messages_enabled': messages_enabled,
            'message_phase_timeout_s': message_phase_timeout_s,
            'message_phase_concurrency': message_phase_concurrency,
            'explicit_interaction_enabled': explicit_interaction_enabled,
            'auto_clamp': auto_clamp,
            'agent_view_memory': agent_view_memory,
            'agent_history_memory': agent_history_memory,
            # Memory, protocol, and rate-limiting mechanisms.
            'enable_structured_view_memory': enable_structured_view_memory,
            'view_memory_k': view_memory_k,
            'view_memory_max_lines': view_memory_max_lines,
            'enable_message_memory': enable_message_memory,
            'message_memory_k_sent': message_memory_k_sent,
            'message_memory_k_recv': message_memory_k_recv,
            'single_message_per_round': single_message_per_round,
            'single_message_prompt_mode': single_message_prompt_mode,
            'ring_mode': ring_mode,
            # Protocol V2.
            'protocol_v2_enabled': protocol_v2_enabled,
            'protocol_v2_auto': protocol_v2_auto,
            'ask_cooldown_rounds': ask_cooldown_rounds,
            'max_items_in_message': max_items_in_message,
            'max_tables_in_message': max_tables_in_message,
            # Protocol V2 free-text parameters.
            'proto_v2_free_enable': proto_v2_free_enable,
            'proto_v2_free_slot': proto_v2_free_slot,
            'proto_v2_free_max': proto_v2_free_max,
            'proto_v2_header_max_len': proto_v2_header_max_len,
            'proto_v2_total_max_len': proto_v2_total_max_len,
            'agent_carrying_capacity': agent_carrying_capacity,
            'mechanism_preset': mechanism_preset,
            'mechanism_raw_config': mechanism_raw_config,
        }

        if name in cls.submission:
            raise ValueError(f"Name ({name}) already exists.")
        cls.submission[name] = kwargs

    @classmethod
    def run_all(cls, max_parallel=None):
        for name, args in cls.submission.items():
            cls.instances[name] = cls(name=name)

        def wrapper(name):
            cls.instances[name].run_task(**cls.submission[name])

        def daemon():
            max_name_len = max([len(name) for name in cls.submission])
            max_progress_len = max([len(f'{d["max_round"]}/{d["max_round"]}')
                                    for d in cls.submission.values()])
            fmt_str = f'{{:<{max_name_len}}} - {{:>{max_progress_len}}}'
            prev_prog = -1

            while True:
                dones = 0
                total_progress = 0
                progress = 0
                brief = []
                for name, instance in cls.instances.items():
                    total_progress += cls.submission[name]['max_round']
                    if instance.status == 'running':
                        cur_progress = instance.framework.env.round
                    elif instance.status == 'done':
                        dones += 1
                        cur_progress = cls.submission[name]['max_round']
                    else:
                        cur_progress = 0
                    progress += cur_progress
                    brief.append(fmt_str.format(name, f"{cur_progress}/{cls.submission[name]['max_round']}"))
                prog = progress / total_progress
                brief.append(f'Progress: {prog:.2%}')

                if prog != prev_prog:
                    output('\n'.join(brief))
                    output('\n')
                    prev_prog = prog

                if dones == len(cls.submission):
                    break
                time.sleep(1)

        with silence(), ThreadPoolExecutor(
                max_workers=len(cls.instances) if max_parallel is None else max_parallel
        ) as executor:
            for name in cls.instances:
                executor.submit(wrapper, name)
            daemon_thread = Thread(target=daemon)
            daemon_thread.start()
        daemon_thread.join()

        cls.instances = {}
        cls.submission = {}


SwarmFramework = MASFramework
