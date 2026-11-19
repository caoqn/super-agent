import asyncio

class Environment:
    def __init__(self, env_name, agents, logger):
        self.agents = agents
        self.env_name = env_name
        self.logger = logger
        self._round_obs_cache = {}
        self._round_action_intents = {}
        # Default to the current refactored behavior. Individual environments
        # can override these hooks when they need compatibility timing.
        self.message_phase_order = 'before_update'
        self.message_phase_use_cached_obs = True
        self.message_phase_include_action_intent = False

    def start(self):
        asyncio.run(self.loop())

    async def loop(self):
        while not self.is_done():
            for agent in self.agents.values():
                self.notify(agent)
            await self.step()
            if getattr(self, 'message_phase_order', 'before_update') == 'after_update':
                self.update()
                self.clear_round_obs_cache()
                await self.message_phase()
            else:
                if not getattr(self, 'message_phase_use_cached_obs', True):
                    self.clear_round_obs_cache()
                await self.message_phase()
                self.update()
            self.clear_round_obs_cache()

    async def step(self):
        self._round_obs_cache = {}
        self._round_action_intents = {}
        tasks = {
            name: agent.decision(self.get_round_obs(agent)) for name, agent in self.agents.items()
        }

        results = await asyncio.gather(*tasks.values())
        actions = {name: result for name, result in zip(tasks.keys(), results)}
        if getattr(self, 'message_phase_include_action_intent', False):
            self._round_action_intents = {
                name: self._summarize_round_action_intent(name, action)
                for name, action in actions.items()
            }
            self._inject_round_action_intents_into_obs_cache()

        act_results = {}
        for name, action in actions.items():
            act_results[name] = await self.act(self.agents[name], action)
        return act_results

    async def act(self, agent, action):
        return {}

    def notify(self, agent):
        pass

    def reset(self):
        pass

    def obs(self, agent):
        return {}

    def get_round_obs(self, agent_or_name):
        if isinstance(agent_or_name, str):
            name = agent_or_name
            agent = self.agents[name]
        else:
            agent = agent_or_name
            name = agent.name

        if name not in self._round_obs_cache:
            self._round_obs_cache[name] = self.obs(agent)
        return self._round_obs_cache[name]

    def clear_round_obs_cache(self):
        self._round_obs_cache = {}
        self._round_action_intents = {}

    def _summarize_round_action_intent(self, name, action):
        action = dict(action or {})
        obs = dict(self._round_obs_cache.get(name) or {})
        pos = dict(obs.get('position') or {})
        try:
            x = int(pos.get('x', 0) or 0)
            y = int(pos.get('y', 0) or 0)
        except Exception:
            x, y = 0, 0

        move = str(action.get('move') or 'STAY').strip().upper()
        interaction_qty = action.get('interaction_qty')
        interaction_type = action.get('interaction_type')
        target_position = None
        summary = f"Planned action this round: {move}."

        if move == 'UP':
            target_position = {'x': x, 'y': y + 1}
            summary = f"Planned action this round: move UP toward ({target_position['x']},{target_position['y']})."
        elif move == 'DOWN':
            target_position = {'x': x, 'y': y - 1}
            summary = f"Planned action this round: move DOWN toward ({target_position['x']},{target_position['y']})."
        elif move == 'LEFT':
            target_position = {'x': x - 1, 'y': y}
            summary = f"Planned action this round: move LEFT toward ({target_position['x']},{target_position['y']})."
        elif move == 'RIGHT':
            target_position = {'x': x + 1, 'y': y}
            summary = f"Planned action this round: move RIGHT toward ({target_position['x']},{target_position['y']})."
        elif move == 'PICK_UP':
            detail = []
            if interaction_qty is not None:
                detail.append(f"qty={interaction_qty}")
            if interaction_type:
                detail.append(f"type={interaction_type}")
            suffix = f" ({', '.join(detail)})" if detail else ""
            summary = f"Planned action this round: PICK_UP at current position{suffix}."
        elif move == 'PUT_DOWN':
            detail = []
            if interaction_qty is not None:
                detail.append(f"qty={interaction_qty}")
            suffix = f" ({', '.join(detail)})" if detail else ""
            summary = f"Planned action this round: PUT_DOWN at current position{suffix}."
        elif move == 'STAY':
            summary = "Planned action this round: STAY at current position."

        return {
            'move': move,
            'interaction_qty': interaction_qty,
            'interaction_type': interaction_type,
            'target_position': target_position,
            'summary': summary,
        }

    def _derive_round_post_action_self_summary(self, name, action):
        action = dict(action or {})
        obs = dict(self._round_obs_cache.get(name) or {})
        pos = dict(obs.get('position') or {})
        inv = dict(obs.get('inventory') or {})
        tile_items = dict(obs.get('tile_items_counts') or {})
        try:
            x = int(pos.get('x', 0) or 0)
            y = int(pos.get('y', 0) or 0)
        except Exception:
            x, y = 0, 0

        move = str(action.get('move') or 'STAY').strip().upper()
        interaction_qty = action.get('interaction_qty')
        interaction_type = str(action.get('interaction_type') or '').strip().lower() or None

        try:
            inv_qty = int(inv.get('quantity', 0) or 0)
        except Exception:
            inv_qty = 0
        inv_type = str(inv.get('type') or '').strip().lower() or None
        inv_display = str(inv.get('display') or 'empty')
        try:
            inv_capacity = int(inv.get('capacity', 0) or 0)
        except Exception:
            inv_capacity = 0

        post_x, post_y = x, y
        if move == 'UP':
            post_y = y + 1
        elif move == 'DOWN':
            post_y = y - 1
        elif move == 'LEFT':
            post_x = x - 1
        elif move == 'RIGHT':
            post_x = x + 1

        post_inv_type = inv_type
        post_inv_qty = inv_qty
        interaction_anchor = 'unchanged'

        if move == 'PICK_UP':
            chosen_type = interaction_type
            if not chosen_type:
                available_types = []
                for item_type in ('wood', 'sand', 'stone'):
                    try:
                        qty = int(tile_items.get(item_type, 0) or 0)
                    except Exception:
                        qty = 0
                    if qty > 0:
                        available_types.append(item_type)
                if len(available_types) == 1:
                    chosen_type = available_types[0]
            try:
                chosen_qty = int(interaction_qty or 0)
            except Exception:
                chosen_qty = 0
            if chosen_type:
                try:
                    available_qty = int(tile_items.get(chosen_type, 0) or 0)
                except Exception:
                    available_qty = 0
                remaining_capacity = max(0, inv_capacity - inv_qty) if inv_capacity > 0 else 0
                if chosen_qty <= 0:
                    chosen_qty = available_qty
                effective_qty = max(0, min(chosen_qty, available_qty, remaining_capacity))
                if effective_qty > 0 and (not inv_type or inv_type == chosen_type):
                    post_inv_type = chosen_type
                    post_inv_qty = inv_qty + effective_qty
                interaction_anchor = f"PICK_UP {chosen_type} x{effective_qty}" if effective_qty > 0 else f"PICK_UP {chosen_type or 'unknown'}"
            else:
                interaction_anchor = "PICK_UP at current tile"
        elif move == 'PUT_DOWN':
            try:
                chosen_qty = int(interaction_qty or 0)
            except Exception:
                chosen_qty = 0
            if chosen_qty <= 0:
                chosen_qty = inv_qty
            effective_qty = max(0, min(chosen_qty, inv_qty))
            remaining_qty = max(0, inv_qty - effective_qty)
            post_inv_qty = remaining_qty
            post_inv_type = inv_type if remaining_qty > 0 else None
            if inv_type:
                interaction_anchor = f"PUT_DOWN {inv_type} x{effective_qty}"
            else:
                interaction_anchor = "PUT_DOWN at current tile"

        post_inventory_display = f"{post_inv_type} x{post_inv_qty}" if post_inv_type and post_inv_qty > 0 else "empty"
        summary_lines = [
            "Deterministic post-action self summary:",
            f"- Current self position: ({x},{y})",
            f"- Planned post-action self position anchor: ({post_x},{post_y})",
            f"- Current carrying: {inv_display}",
            f"- Planned post-action carrying anchor: {post_inventory_display}",
        ]
        if move in ('PICK_UP', 'PUT_DOWN'):
            summary_lines.append(f"- Chosen interaction anchor: {interaction_anchor}")
        else:
            summary_lines.append(f"- Chosen movement anchor: {move}")
        summary_lines.append(
            "- Use this only as a self-state anchor. Do not infer any new resources, tables, agents, or visibility beyond the current observation."
        )

        return {
            'move': move,
            'post_position': {'x': post_x, 'y': post_y},
            'post_position_display': f"({post_x},{post_y})",
            'post_inventory_type': post_inv_type,
            'post_inventory_quantity': post_inv_qty,
            'post_inventory_display': post_inventory_display,
            'interaction_anchor': interaction_anchor,
            'summary': "\n".join(summary_lines),
        }

    def _inject_round_action_intents_into_obs_cache(self):
        for name, obs in list(self._round_obs_cache.items()):
            if not isinstance(obs, dict):
                continue
            intent = dict(self._round_action_intents.get(name) or {})
            if not intent:
                continue
            obs['current_round_action_intent'] = intent
            obs['current_round_action_summary'] = str(intent.get('summary') or '')
            post_action = self._derive_round_post_action_self_summary(name, intent)
            obs['current_round_post_action_self'] = post_action
            obs['current_round_post_action_self_summary'] = str(post_action.get('summary') or '')

    def is_done(self):
        pass

    def update(self):
        pass

    async def message_phase(self):
        """Subclasses may override the message phase after each action round; environment hooks determine the order."""
        return None
