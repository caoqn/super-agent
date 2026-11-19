from util import safe_open, SafeOpenWrapper
from datetime import datetime
from masbench.physics import CapacityMesh

import os
import time
import json
import numpy as np
from framework.logger import Logger
from threading import Lock, current_thread
from io import TextIOWrapper
import builtins
from typing import Optional, Dict, Any
from masbench.token_utils import collect_message_exchange_stats

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


class MASLogger(Logger):

    def __init__(self, name, meta, log_dir="structured_logs"):
        super().__init__(name)
        self.log_dir = os.path.realpath(log_dir)
        with SafeOpenWrapper.get_lock(self.log_dir):
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        # init logger
        timestamp = f'{int(time.time())}-{current_thread().getName()}'
        self.timestamp = timestamp
        self.start_time_ts = time.time()
        self.game_log_file = os.path.join(log_dir, f"game_log_{timestamp}.json")
        self.agent_log_file = os.path.join(log_dir, f"agent_log_{timestamp}.json")
        self.memory_log_file = os.path.join(log_dir, f"memory_log_{timestamp}.json")
        with safe_open(os.path.join(self.log_dir, f"meta_log.json"), "a") as f:
            pass
        with safe_open(os.path.join(self.log_dir, f"meta_log.json"), "r+") as f:
            try:
                meta_log = json.load(f)
            except json.decoder.JSONDecodeError:
                meta_log = {}
            # Write initial settings and log paths once at startup.
            start_meta = dict(meta or {})
            # Document external mathematical coordinates: bottom-left origin, y upward.
            start_meta['coord_sys'] = 'math'
            start_meta.update({
                'log_files': {
                    'game_log': self.game_log_file,
                    'agent_log': self.agent_log_file
                },
                # Record a human-readable start time.
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            # Record the parent shell script launching this evaluation and its start time, if detectable.
            launcher = self._detect_launcher_info()
            if launcher:
                start_meta['launcher'] = launcher
            # Record the batch/tag passed by the external shell.
            launch_tag = os.getenv('EVAL_LAUNCH_TAG') or os.getenv('EVAL_BATCH_TAG') or os.getenv('EXPERIMENT_TAG')
            if launch_tag:
                start_meta['launch_tag'] = str(launch_tag)
            meta_log[timestamp] = start_meta
            f.truncate(0)
            f.seek(0)
            # Use multiline formatting for readability.
            json.dump(meta_log, f, indent=2)

        # init log data
        self.game_logs = []
        self.agent_logs = []
        self.memory_logs = []

    def _detect_launcher_info(self) -> Optional[Dict[str, Any]]:
        """Infer the parent shell script and its start time on a best-effort basis.
        Prefer environment variables:
          - EVAL_LAUNCH_SCRIPT: Shell script path.
          - EVAL_LAUNCH_TS: Shell start time in epoch seconds.
          - EVAL_LAUNCH_TIME: Human-readable shell start time (fallback).
        If environment variables are missing, use psutil to search parent process arguments for a .sh script.
        """
        try:
            script_env = os.getenv('EVAL_LAUNCH_SCRIPT')
            ts_env = os.getenv('EVAL_LAUNCH_TS')
            time_env = os.getenv('EVAL_LAUNCH_TIME')
            if script_env or ts_env or time_env:
                start_time_str = None
                if ts_env:
                    try:
                        t = float(ts_env)
                        start_time_str = datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        start_time_str = None
                if not start_time_str and time_env:
                    start_time_str = str(time_env)
                return {
                    'script': script_env or '',
                    'start_time': start_time_str or '',
                    'source': 'env'
                }

            if psutil is None:
                return None
            proc = psutil.Process(os.getpid())
            for parent in proc.parents():
                try:
                    cmd = parent.cmdline() or []
                    # Look for arguments such as /path/to/script.sh.
                    script_args = [c for c in cmd if isinstance(c, str) and c.endswith('.sh')]
                    if script_args:
                        script_path = script_args[0]
                        ctime = parent.create_time() if hasattr(parent, 'create_time') else None
                        start_time_str = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S') if ctime else ''
                        return {
                            'script': script_path,
                            'start_time': start_time_str,
                            'pid': parent.pid,
                            'source': 'psutil'
                        }
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def finalize_meta(self, env):
        """Merge final experiment information into this timestamp's meta_log.json entry."""
        try:
            meta_path = os.path.join(self.log_dir, "meta_log.json")
            with safe_open(meta_path, "r+") as f:
                try:
                    meta_log = json.load(f)
                except json.decoder.JSONDecodeError:
                    meta_log = {}
                entry = meta_log.get(self.timestamp, {})
                # Retrieve final task statistics if possible.
                end_ts = time.time()
                # Token statistics:
                # - overall: prompt/completion/total across all phases; completion excludes reasoning.
                # - total retains the provider total, which may exceed prompt+completion for reasoning models.
                # - message_phase: prompt/completion/total for messaging only; completion excludes reasoning.
                # - *_raw / reasoning: original provider completion counts and reasoning details, if available.
                overall = {
                    'prompt_total': 0,
                    'completion_total': 0,
                    'completion_raw_total': 0,
                    'reasoning_total': 0,
                    'total': 0,
                    'by_agent': {},
                }
                msg_only = {'prompt_total': 0, 'completion_total': 0, 'total': 0, 'by_agent': {}}
                try:
                    for name, agent in (getattr(env, 'agents', {}) or {}).items():
                        bp = int(getattr(agent.brain, 'usage_prompt', 0) or 0)
                        bc = int(getattr(agent.brain, 'usage_completion', 0) or 0)
                        bcr = int(getattr(agent.brain, 'usage_completion_raw', bc) or bc)
                        br = int(getattr(agent.brain, 'usage_reasoning', 0) or 0)
                        bt = int(getattr(agent.brain, 'usage_total', bp + bc) or (bp + bc))
                        mp = int(getattr(agent.brain, 'message_prompt_tokens', 0) or 0)
                        mc = int(getattr(agent.brain, 'message_completion_tokens', 0) or 0)
                        mt = mp + mc
                        overall['by_agent'][name] = {'prompt': bp, 'completion': bc, 'completion_raw': bcr, 'reasoning': br, 'total': bt}
                        msg_only['by_agent'][name] = {'prompt': mp, 'completion': mc, 'total': mt}
                        overall['prompt_total'] += bp
                        overall['completion_total'] += bc
                        overall['completion_raw_total'] += bcr
                        overall['reasoning_total'] += br
                        overall['total'] += bt
                        msg_only['prompt_total'] += mp
                        msg_only['completion_total'] += mc
                        msg_only['total'] += mt
                except Exception:
                    pass

                # Elapsed runtime.
                duration_seconds = max(0.0, float(end_ts - getattr(self, 'start_time_ts', end_ts)))
                def _fmt_hms(sec: float) -> str:
                    s = int(round(sec))
                    hh = s // 3600
                    mm = (s % 3600) // 60
                    ss = s % 60
                    return f"{hh:02d}:{mm:02d}:{ss:02d}"

                message_exchange = collect_message_exchange_stats(getattr(env, 'messages', []) or [])

                summary = {
                    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'end_round': int(getattr(env, 'round', -1)),
                    'duration_seconds': round(duration_seconds, 3),
                    'duration_hms': _fmt_hms(duration_seconds),
                    'tokens': {
                        'overall': overall,
                        'message_phase': msg_only,
                        'message_exchange': message_exchange,
                    }
                }
                try:
                    task_obj = env.tasks.get(env.task)
                    if hasattr(task_obj, 'score'):
                        summary['final_score'] = int(task_obj.score)
                    # Snapshot table items if a statistics method is available.
                    if hasattr(task_obj, 'count_items_on_tables'):
                        summary['items_on_tables'] = task_obj.count_items_on_tables(env)
                except Exception:
                    pass
                # Write back.
                entry['finish'] = summary
                meta_log[self.timestamp] = entry
                f.truncate(0)
                f.seek(0)
                json.dump(meta_log, f, indent=2)
        except Exception as e:
            print(f"WARNING: [LOGGER] finalize_meta failed: {e}")

    def log_game_state(self, env, round_num, timestamp):
        """Log game state"""
        # Convert numpy arrays to lists, retaining only the first character for visualization.
        temp_grid = env._render_grid_state()
        # Simplify display to each cell's first character.
        simplified_grid = np.full_like(temp_grid, '.', dtype=object)
        for i in range(temp_grid.shape[0]):
            for j in range(temp_grid.shape[1]):
                cell = str(temp_grid[i, j])
                simplified_grid[i, j] = cell[0] if cell else '.'
        grid_list = simplified_grid.tolist()

        # Get agent information
        agents_info = []
        for name, mesh in env.agent_meshes.items():
            # Get the agent's carrying state.
            carrying_item = None
            carrying_qty = 0
            # Prefer the enhanced inventory system.
            agent_obj = env.agents.get(name)
            if agent_obj is not None and hasattr(agent_obj, 'enable_inventory') and agent_obj.enable_inventory:
                try:
                    carrying_qty = agent_obj.get_total_carrying()
                    ctype = agent_obj.get_carrying_item_type()
                    carrying_item = ctype if carrying_qty > 0 else None
                except Exception:
                    pass
            else:
                # Legacy system compatibility.
                if hasattr(env, 'tasks') and env.task in env.tasks:
                    task = env.tasks[env.task]
                    if hasattr(task, 'resource_state'):
                        carrying_item = task.resource_state.get(name)
                        carrying_qty = 1 if carrying_item is not None else 0

            # Expose positions in mathematical coordinates.
            _x_img, _y_img = mesh.pos[1], mesh.pos[0]
            try:
                from masbench.coords import to_math_xy
                _x_out, _y_out = to_math_xy(_x_img, _y_img, env.height)
            except Exception:
                _x_out, _y_out = _x_img, _y_img

            agent_info = {
                "id": name,
                "x": _x_out,
                "y": _y_out,
                "shape": mesh.shape[0, 0],
                # Agent state information.
                "carrying_item": carrying_item,
                "is_carrying": carrying_item is not None,
                "carrying_qty": carrying_qty
            }
            agents_info.append(agent_info)
            print(f"DEBUG: [GAME_LOG] Agent {name} @ row/column({mesh.pos[0]}, {mesh.pos[1]}) carrying: {carrying_item}")

        # Get non-agent mesh information
        meshes_info = []
        for mesh in env.meshes:
            if mesh.name not in env.agents: # Exclude agents themselves
                mesh_data = {
                    "name": mesh.name,
                    "pos": mesh.pos,
                    "shape": mesh.shape.tolist()
                }
                if isinstance(mesh, CapacityMesh):
                    mesh_data['capacity'] = mesh.capacity
                # meshes_info.append(mesh_data)

        # Get messages
        messages = [m for m in env.messages if m["round"] == round_num-1]
        # Debug print messages with recipients
        for m in messages:
            print(f"DEBUG: [GAME_LOG][MSG] round={m.get('round')} from={m.get('agent_name')} recipients={m.get('recipients', None)} text='{m.get('message','')[:60]}'")

        # Collect global table status (remaining total and per-type if available)
        tables_status = []
        try:
            for mesh in env.meshes:
                try:
                    if hasattr(mesh, 'terrain_type') and str(getattr(mesh, 'terrain_type')) == 'table':
                        x_img = int(mesh.pos[1]); y_img = int(mesh.pos[0])
                        try:
                            from masbench.coords import to_math_xy
                            x, y = to_math_xy(x_img, y_img, env.height)
                        except Exception:
                            x, y = x_img, y_img
                        # total remaining
                        try:
                            rem_total = int(mesh.get_remaining_total()) if hasattr(mesh, 'get_remaining_total') else (
                                max(0, int(getattr(mesh, 'capacity', 0)) - int(mesh.get_items_total_quantity() if hasattr(mesh, 'get_items_total_quantity') else 0))
                            )
                        except Exception:
                            rem_total = None
                        entry = {"position": {"x": x, "y": y}, "remaining_capacity": rem_total}
                        # per-type remaining (optional)
                        try:
                            if hasattr(mesh, 'get_remaining_by_type') and (getattr(mesh, 'capacity_by_type', None) or getattr(mesh, 'per_type_default', None) is not None):
                                rb = mesh.get_remaining_by_type(['wood', 'sand', 'stone'])
                                if rb:
                                    entry['remaining_by_type'] = {k: int(v) for k, v in rb.items()}
                        except Exception:
                            pass
                        tables_status.append(entry)
                except Exception:
                    continue
        except Exception as e:
            print(f"WARNING: [GAME_LOG][TABLES] Failed to collect table state: {e}")

        game_state = {
            "timestamp": timestamp,
            "round": round_num,
            "grid": grid_list,
            "agents": agents_info,
            "meshes": meshes_info, # Add meshes info to the log
            **env.tasks[env.task].task_obs(None),
            "messages": messages,
            "tables_status": tables_status,
            "coord_sys": "math"
        }
        self.game_logs.append(game_state)
        self.save_game_logs()

        # === Persist per-agent memory snapshot by round ===
        try:
            mem_round = {"round": round_num, "timestamp": timestamp, "agents": {}}
            for name, agent_obj in env.agents.items():
                # position
                pos = env.agent_meshes.get(name).pos if name in env.agent_meshes else (None, None)
                ax, ay = (pos[1], pos[0]) if pos else (None, None)
                tables = []
                items = []
                sent = []
                recv = []
                try:
                    if hasattr(env, '_get_memory_state'):
                        mem_state = env._get_memory_state(name) or {}
                    else:
                        mem_state = {}
                    tables = list(mem_state.get('tables', [])) if isinstance(mem_state, dict) else []
                    items = list(mem_state.get('items', [])) if isinstance(mem_state, dict) else []
                    msg_state = mem_state.get('messages', {}) if isinstance(mem_state, dict) else {}
                    sent = list(msg_state.get('sent', [])) if isinstance(msg_state, dict) else []
                    recv = list(msg_state.get('recv', [])) if isinstance(msg_state, dict) else []
                except Exception:
                    pass
                if not tables and not items:
                    try:
                        km = getattr(agent_obj, 'known_map', {}) or {}
                        for (x, y), rec in km.items():
                            ter = rec.get('terrain_type')
                            rem = rec.get('remaining_capacity')
                            its = rec.get('items') or {}
                            entry = {"x": int(x), "y": int(y)}
                            if ter:
                                entry["terrain"] = str(ter)
                            if rem is not None:
                                entry["remaining_capacity"] = int(rem)
                            if its:
                                try:
                                    entry["items"] = {str(k): int(v) for k, v in its.items()}
                                except Exception:
                                    entry["items"] = its
                            if str(ter) == 'table':
                                tables.append(entry)
                            elif its:
                                items.append(entry)
                    except Exception:
                        pass
                if not sent and not recv:
                    try:
                        rm = env.recent_msgs.get(name, {})
                        sent = list(rm.get('sent', [])) if rm else []
                        recv = list(rm.get('recv', [])) if rm else []
                    except Exception:
                        sent, recv = [], []
                # Plan memory snapshot (lightweight)
                plan_mem = {}
                try:
                    if hasattr(agent_obj, 'enable_plan_memory') and agent_obj.enable_plan_memory:
                        cur = getattr(agent_obj, 'current_plan', None)
                        entries = list(getattr(agent_obj, 'plan_entries', []) or [])
                        def _strip(e):
                            if not isinstance(e, dict): return {}
                            # don't dump full response to keep logs compact; include summary and parsed
                            return {
                                'round': e.get('round'),
                                'summary': e.get('summary'),
                                'parsed': e.get('parsed'),
                                'status': e.get('status'),
                                'progress': e.get('progress')
                            }
                        plan_mem = {
                            'current': _strip(cur) if cur else None,
                            'recent': [_strip(e) for e in entries[-3:]]
                        }
                except Exception:
                    plan_mem = {}

                # Response memory snapshot (last 3 raw)
                response_mem = []
                try:
                    if hasattr(agent_obj, 'enable_response_memory') and agent_obj.enable_response_memory:
                        entries = list(getattr(agent_obj, 'response_memory', []) or [])
                        for e in entries[-3:]:
                            response_mem.append({
                                'round': e.get('round'),
                                'phase': e.get('phase'),
                                'response': e.get('response')
                            })
                except Exception:
                    response_mem = []

                # Expose positions in mathematical coordinates.
                try:
                    from masbench.coords import to_math_xy
                    ax_out, ay_out = to_math_xy(ax, ay, env.height)
                except Exception:
                    ax_out, ay_out = ax, ay

                mem_round["agents"][name] = {
                    "position": {"x": ax_out, "y": ay_out},
                    "tables": tables,
                    "items": items,
                    "messages": {"sent": sent, "recv": recv},
                    "mechanism_memory": mem_state if isinstance(mem_state, dict) else {},
                    "plan_memory": plan_mem,
                    "response_memory": response_mem
                }
            self.memory_logs.append(mem_round)
            with safe_open(self.memory_log_file, 'w') as f:
                json.dump(self.memory_logs, f, indent=2)
        except Exception as e:
            print(f"WARNING: [MEM_LOG] persist failed: {e}")

    def log_agent_action(self, env, agent_id, prompt, response, action, message,
                         api_call_time, action_success, view, total_llm_tokens, extra=None):
        """Record agent actions"""
        # Convert the vision numpy array to a clipped list (align with agent prompt):
        # Only keep rows/cols whose global coords fall within [-1, H] x [-1, W]
        view_list = None
        if view is not None:
            try:
                H, W = int(env.height), int(env.width)
                # agent global position
                mesh = env.agent_meshes.get(agent_id)
                if mesh is not None:
                    y, x = int(mesh.pos[0]), int(mesh.pos[1])
                    vh, vw = view.shape[0], view.shape[1]
                    i1, j1 = y - vh // 2, x - vw // 2
                    i_lo = max(0, -1 - i1)
                    i_hi = min(vh - 1, H - i1)
                    j_lo = max(0, -1 - j1)
                    j_hi = min(vw - 1, W - j1)
                    if i_lo <= i_hi and j_lo <= j_hi:
                        clipped = view[i_lo:i_hi+1, j_lo:j_hi+1]
                        view_list = clipped.tolist()
                    else:
                        view_list = []
                else:
                    view_list = view.tolist()
            except Exception:
                view_list = view.tolist()

        # Get current carried items, preferring enhanced inventory.
        carrying_item = None
        item_quantity = 0
        agent_obj = env.agents.get(agent_id)
        if agent_obj is not None and hasattr(agent_obj, 'enable_inventory') and agent_obj.enable_inventory:
            try:
                item_quantity = agent_obj.get_total_carrying()
                ctype = agent_obj.get_carrying_item_type()
                carrying_item = ctype if item_quantity > 0 else None
            except Exception:
                pass
        else:
            if hasattr(env, 'tasks') and env.task in env.tasks:
                task = env.tasks[env.task]
                if hasattr(task, 'resource_state') and agent_id in task.resource_state:
                    carrying_item = task.resource_state.get(agent_id)
                    item_quantity = 1 if carrying_item is not None else 0

        print(f"DEBUG: [LOGGER] Agent {agent_id} state - carried item: {carrying_item}, quantity: {item_quantity}")

        agent_log = {
            "timestamp": time.time(),
            "round": env.round,
            "agent_id": agent_id,
            "view": view_list,
            "prompt": prompt,
            "response": response,
            "action": action,
            "message": message,
            "api_call_time": api_call_time,
            "action_success": action_success,
            'total_llm_tokens': total_llm_tokens,
            # Additional fields: agent state information.
            "agent_state": {
                "is_carrying": carrying_item is not None,
                "carrying_item": carrying_item,
                "item_quantity": item_quantity
            }
        }
        # Allow extra debugging and structured fields.
        if isinstance(extra, dict):
            try:
                agent_log.update(extra)
            except Exception:
                pass
        self.agent_logs.append(agent_log)
        self.save_agent_logs()

    def save_game_logs(self):
        """Save game logs"""
        with safe_open(self.game_log_file, 'w') as f:
            json.dump(self.game_logs, f, indent=2)

    def save_agent_logs(self):
        """Save agent logs"""
        with safe_open(self.agent_log_file, 'w') as f:
            json.dump(self.agent_logs, f, indent=2)

    def info(self, msg):
        """
        Output informational logs
        Overriding the info method of Logger
        """
        super().info(msg)
        print(msg)

    def debug(self, msg):
        """
        Output debug logs
        Overriding the debug method of Logger
        """
        super().debug(msg)
        if os.environ.get('DEBUG'):
            print(f"[DEBUG] {msg}")

    def error(self, msg):
        """
        Output error logs
        Overriding the error method of Logger
        """
        super().error(msg)
        print(f"[ERROR] {msg}")


SwarmLogger = MASLogger
