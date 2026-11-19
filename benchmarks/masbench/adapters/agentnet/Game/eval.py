from masbench import MASFramework
from masbench.debug_silencer import silence_debug_prints
from datetime import datetime
from dotenv import load_dotenv
import os
import json
import argparse

# Load the .env file into environment variables automatically.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(ROOT_DIR, '.env')
load_dotenv(dotenv_path=DOTENV_PATH)
# Silence DEBUG: prints during runs unless explicitly enabled
silence_debug_prints(True)

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--provider", type=str, default="QWEN",
                    help="Provider/env prefix, e.g. QWEN or API_YI2. "
                         "For OpenAI-compatible providers, credentials are read from "
                         "<PROVIDER>_OPENAI_API_KEY and <PROVIDER>_OPENAI_BASE.")
parser.add_argument("--model", type=str, default="deepseek-v3.2-exp")
parser.add_argument("--max_round", type=int, default=None,
                    help="Override max_round; precedence: args > OVERRIDE > default(50)")
args = parser.parse_args()
args.provider = str(args.provider or '').strip().upper()
if not args.provider:
    parser.error("--provider must not be empty")

def _resolve_provider_credentials(provider: str):
    provider_name = str(provider or '').strip().upper()
    if provider_name == 'GEMINI':
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_OPENAI_API_KEY')
        api_base = os.getenv('GEMINI_API_BASE', '')
        return api_key, api_base, 'GEMINI'

    api_key = os.getenv(f"{provider_name}_OPENAI_API_KEY")
    api_base = os.getenv(f"{provider_name}_OPENAI_BASE")
    return api_key, api_base, provider_name


API_KEY, API_BASE, RESOLVED_PROVIDER_ENV_PREFIX = _resolve_provider_credentials(args.provider)

if args.provider == 'GEMINI':
    if not API_KEY:
        raise RuntimeError(
            "Missing Gemini credentials. Expected one of GEMINI_API_KEY / GOOGLE_API_KEY / GEMINI_OPENAI_API_KEY in .env."
        )
else:
    if not API_KEY or not API_BASE:
        provider_name = str(args.provider or '').strip().upper()
        expected = f"{provider_name}_OPENAI_API_KEY / {provider_name}_OPENAI_BASE"
        raise RuntimeError(
            f"Missing provider credentials for {provider_name}. "
            f"Expected {expected} in {DOTENV_PATH}. "
            "Provider selects which key/api_base to use; model is passed through unchanged."
        )

def load_custom_map(path):
    print(f"DEBUG: [EVAL] Loading custom map: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    # ===== User configuration =====
    # Model/API (prefer environment variables over plaintext credentials).

    if args.model == '':
        MODEL = os.getenv('MODEL', 'gpt-5-mini')
    else:
        MODEL = args.model

    VIEW_SIZE = os.getenv('VIEW_SIZE', None)

    log_suffix = f"{MODEL}_view{VIEW_SIZE}" if VIEW_SIZE is not None else f"{MODEL}_viewjson"

    # Custom map path (empty uses a default random task).
    CUSTOM_MAP_PATH = os.getenv('CUSTOM_MAP_PATH', os.path.join(ROOT_DIR, 'custom_map', '18x18_v1.json'))

    # Agent memory windows (environment variables allow quick checks).
    AGENT_VIEW_MEMORY = os.getenv('AGENT_VIEW_MEMORY')
    AGENT_HISTORY_MEMORY = os.getenv('AGENT_HISTORY_MEMORY')
    try:
        AGENT_VIEW_MEMORY = int(AGENT_VIEW_MEMORY) if AGENT_VIEW_MEMORY is not None else 1
    except Exception:
        AGENT_VIEW_MEMORY = None
    try:
        AGENT_HISTORY_MEMORY = int(AGENT_HISTORY_MEMORY) if AGENT_HISTORY_MEMORY is not None else 0
    except Exception:
        AGENT_HISTORY_MEMORY = None

    # Startup switches and tunable stable_fair values; registry presets control mechanism switches.
    def _to_bool(val, default=False):
        if val is None:
            return default
        v = str(val).strip().lower()
        return v in ('1', 'true', 'yes', 'y', 'on')

    def _int_env(name, default):
        try:
            raw = os.getenv(name)
            if raw is None or str(raw).strip() == "":
                return int(default)
            return int(raw)
        except Exception:
            return int(default)

    MECHANISM_PRESET = os.getenv('MECHANISM_PRESET', 'base').strip() or 'base'

    VIEW_MEMORY_K = _int_env('VIEW_MEMORY_K', 8)
    VIEW_MEMORY_MAX_LINES = _int_env('VIEW_MEMORY_MAX_LINES', 8)
    KB_TTL_ROUNDS = _int_env('KB_TTL_ROUNDS', 8)
    KB_MEMORY_MAX_ENTRIES = _int_env('KB_MEMORY_MAX_ENTRIES', VIEW_MEMORY_MAX_LINES)
    KB_NEAREST_TABLES_K = _int_env('KB_NEAREST_TABLES_K', 1)
    KB_NEAREST_ITEMS_K = _int_env('KB_NEAREST_ITEMS_K', 2)
    RESPONSE_MEMORY_K = _int_env('RESPONSE_MEMORY_K', 3)
    MESSAGE_MEMORY_SENT_K = _int_env('MESSAGE_MEMORY_SENT_K', 3)
    MESSAGE_MEMORY_RECV_K = _int_env('MESSAGE_MEMORY_RECV_K', 3)
    ASK_COOLDOWN_ROUNDS = _int_env('ASK_COOLDOWN_ROUNDS', 2)
    MAX_ITEMS_IN_MESSAGE = _int_env('MAX_ITEMS_IN_MESSAGE', 5)
    MAX_TABLES_IN_MESSAGE = _int_env('MAX_TABLES_IN_MESSAGE', 3)
    PROTO_V2_FREE_SLOT = os.getenv('PROTO_V2_FREE_SLOT', 'C')
    PROTO_V2_FREE_MAX = _int_env('PROTO_V2_FREE_MAX', 150)
    PROTO_V2_HEADER_MAX_LEN = _int_env('PROTO_V2_HEADER_MAX_LEN', 0)
    PROTO_V2_TOTAL_MAX_LEN = _int_env('PROTO_V2_MAX_LEN', 220)

    # Master message switch (overrides all message paths for testing).
    SEND_MESSAGES = _to_bool(os.getenv('SEND_MESSAGES'), True)

    # Task and parameter overrides (empty/None uses JSON or defaults).
    TASKS = ('Foraging',)  # Options: 'Transport','Pursuit','Synchronization','Foraging','Flocking'
    SEEDS = (42,)
    OVERRIDE = {
        # None means no override: use JSON when available, otherwise defaults.
        'num_agents': None,      # If unset, use the number of JSON map.agents.
        'max_round': 50,        # Default if omitted from JSON; a specified number overrides it.
        'width': None,           # If unset, use JSON map.width.
        'height': None,          # If unset, use JSON map.height.
        'view_size': None,       # If unset, use JSON map.agent_config.view_size.
        'communication_size': None,  # If unset, use JSON map.agent_config.communication_size.
        'agent_item_capacity': None, # If unset, use JSON map.agent_config.item_number.
    }

    # ===== Submit evaluation =====
    name = 1

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    map_label = "default_map"
    if CUSTOM_MAP_PATH:
        try:
            map_label = os.path.splitext(os.path.basename(CUSTOM_MAP_PATH))[0] or "custom_map"
        except Exception:
            map_label = "custom_map"
    safe_map_label = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in str(map_label))

    safe_preset_name = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in MECHANISM_PRESET)
    log_dir = os.path.join('.preset_logs', f"{timestamp}_{safe_map_label}_{safe_preset_name}_{log_suffix}")

    # Preload the custom map, if provided.
    custom_map = None
    if CUSTOM_MAP_PATH:
        try:
            custom_map = load_custom_map(CUSTOM_MAP_PATH)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to load custom map: {CUSTOM_MAP_PATH}") from e

    for task in TASKS:
        for seed in SEEDS:
            # Resolve parameters: eval overrides take precedence over JSON and defaults.
            if custom_map:
                map_cfg = custom_map.get('map', {})
                agent_cfg = map_cfg.get('agent_config', {})
                json_agents_list_len = len(map_cfg.get('agents', []) or [])
                json_agent_number = agent_cfg.get('agent_number') if isinstance(agent_cfg, dict) else None
                json_width = map_cfg.get('width')
                json_height = map_cfg.get('height')
                json_view = agent_cfg.get('view_size') if isinstance(agent_cfg, dict) else None
                json_comm = agent_cfg.get('communication_size') if isinstance(agent_cfg, dict) else None
                json_item_capacity = agent_cfg.get('item_number') if isinstance(agent_cfg, dict) else None
            else:
                json_agents_list_len = None
                json_agent_number = None
                json_width = None
                json_height = None
                json_view = None
                json_comm = None
                json_item_capacity = None

            # num_agents priority: env > override > json(agent_number/list_len) > default.
            num_src = 'default'
            if os.getenv('NUM_AGENTS') is not None:
                try:
                    env_num = int(os.getenv('NUM_AGENTS'))
                    if env_num > 0:
                        num_agents = env_num
                        num_src = 'env'
                    else:
                        raise ValueError
                except Exception:
                    num_agents = None
            if num_src == 'default':
                if OVERRIDE['num_agents'] is not None:
                    num_agents = OVERRIDE['num_agents']
                    num_src = 'override'
                else:
                    if isinstance(json_agent_number, int) and json_agent_number > 0:
                        base_agents = json_agent_number
                    else:
                        base_agents = json_agents_list_len or 2
                    # Use the larger value if the agents list exceeds agent_number to avoid placement failures.
                    if json_agents_list_len and base_agents < json_agents_list_len:
                        print(f"DEBUG: [EVAL] agent_number({base_agents}) < agents list length({json_agents_list_len}); using the larger value")
                        base_agents = json_agents_list_len
                    num_agents = base_agents
                    num_src = 'json' if (json_agent_number or json_agents_list_len) else 'default'
            # max_round priority: args > OVERRIDE > default(50).
            if isinstance(args.max_round, int) and args.max_round > 0:
                max_round = int(args.max_round)
                max_round_src = 'args'
            elif OVERRIDE['max_round'] is not None:
                max_round = OVERRIDE['max_round']
                max_round_src = 'override'
            else:
                max_round = 50
                max_round_src = 'default'
            width = OVERRIDE['width'] if OVERRIDE['width'] is not None else (json_width or 12)
            height = OVERRIDE['height'] if OVERRIDE['height'] is not None else (json_height or 12)
            # view_size:env > override > json > default
            view_src = 'default'
            view_env_raw = os.getenv('VIEW_SIZE')
            if view_env_raw is not None:
                try:
                    v = int(view_env_raw)
                    if v > 0:
                        view_size = v
                        view_src = 'env'
                    else:
                        raise ValueError
                except Exception:
                    view_size = None
            if view_src == 'default':
                if OVERRIDE['view_size'] is not None:
                    view_size = OVERRIDE['view_size']
                    view_src = 'override'
                elif json_view is not None:
                    view_size = json_view
                    view_src = 'json'
                else:
                    view_size = 9
                    view_src = 'default'

            # communication_size:env > override > json > default(max(9, view+3))
            comm_src = 'default'
            comm_env_raw = os.getenv('COMMUNICATION_SIZE')
            if comm_env_raw is not None:
                try:
                    c = int(comm_env_raw)
                    if c > 0:
                        communication_size = c
                        comm_src = 'env'
                    else:
                        raise ValueError
                except Exception:
                    communication_size = None
            if comm_src == 'default':
                if OVERRIDE['communication_size'] is not None:
                    communication_size = OVERRIDE['communication_size']
                    comm_src = 'override'
                elif json_comm is not None:
                    communication_size = json_comm
                    comm_src = 'json'
                else:
                    communication_size = max(9, view_size + 3)
                    comm_src = 'default'

            # agent item capacity:env > override > json(item_number) > default(4)
            cap_src = 'default'
            cap_env_raw = os.getenv('AGENT_ITEM_CAPACITY')
            agent_item_capacity = None
            if cap_env_raw is not None:
                try:
                    cap_val = int(cap_env_raw)
                    if cap_val > 0:
                        agent_item_capacity = cap_val
                        cap_src = 'env'
                except Exception:
                    agent_item_capacity = None
            if agent_item_capacity is None:
                if OVERRIDE.get('agent_item_capacity') is not None:
                    agent_item_capacity = OVERRIDE['agent_item_capacity']
                    cap_src = 'override'
                elif isinstance(json_item_capacity, int) and json_item_capacity > 0:
                    agent_item_capacity = json_item_capacity
                    cap_src = 'json'
                else:
                    agent_item_capacity = 4
                    cap_src = 'default'

            print(f"DEBUG: [EVAL] Submitting task: task={task}, seed={seed}, preset={MECHANISM_PRESET}, num_agents={num_agents}({num_src}), max_round={max_round}({max_round_src}), width={width}, height={height}, view={view_size}({view_src}), comm={communication_size}({comm_src}), item_cap={agent_item_capacity}({cap_src}), custom_map={'Y' if custom_map else 'N'}, view_mem={AGENT_VIEW_MEMORY}, hist_mem={AGENT_HISTORY_MEMORY}, kb_ttl={KB_TTL_ROUNDS}, kb_max={KB_MEMORY_MAX_ENTRIES}, kb_hints={KB_NEAREST_TABLES_K}/{KB_NEAREST_ITEMS_K}, response_mem={RESPONSE_MEMORY_K}")
            mechanism_raw_config = {
                'view_memory_k': VIEW_MEMORY_K,
                'view_memory_max_lines': VIEW_MEMORY_MAX_LINES,
                'kb_ttl_rounds': KB_TTL_ROUNDS,
                'kb_memory_max_entries': KB_MEMORY_MAX_ENTRIES,
                'kb_nearest_tables_k': KB_NEAREST_TABLES_K,
                'kb_nearest_items_k': KB_NEAREST_ITEMS_K,
                'response_memory_k': RESPONSE_MEMORY_K,
                'message_memory_k_sent': MESSAGE_MEMORY_SENT_K,
                'message_memory_k_recv': MESSAGE_MEMORY_RECV_K,
                'ask_cooldown_rounds': ASK_COOLDOWN_ROUNDS,
                'max_items_in_message': MAX_ITEMS_IN_MESSAGE,
                'max_tables_in_message': MAX_TABLES_IN_MESSAGE,
                'proto_v2_free_slot': PROTO_V2_FREE_SLOT,
                'proto_v2_free_max': PROTO_V2_FREE_MAX,
                'proto_v2_header_max_len': PROTO_V2_HEADER_MAX_LEN,
                'proto_v2_total_max_len': PROTO_V2_TOTAL_MAX_LEN,
                'seed': seed,
            }

            MASFramework.submit(
                f'exp_{name}',
                MASFramework.model_config(MODEL, API_KEY, API_BASE, provider=args.provider),
                task,
                log_dir=log_dir,
                num_agents=num_agents,
                max_round=max_round,
                width=width,
                height=height,
                seed=seed,
                view_size=view_size,
                communication_size=communication_size,
                custom_map=custom_map,
                custom_map_path=CUSTOM_MAP_PATH if custom_map is not None else None,
                agent_view_memory=AGENT_VIEW_MEMORY,
                agent_history_memory=AGENT_HISTORY_MEMORY,
                # Override agent carrying capacity.
                agent_carrying_capacity=agent_item_capacity,
                # Master message sending switch.
                messages_enabled=SEND_MESSAGES,
                mechanism_preset=MECHANISM_PRESET,
                mechanism_raw_config=mechanism_raw_config,
            )
            name += 1

    MASFramework.run_all(max_parallel=1)
