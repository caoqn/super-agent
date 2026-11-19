import numpy as np
from scipy.optimize import linear_sum_assignment
from masbench.physics import Mesh, CapacityMesh
from masbench.enhanced_terrain import create_terrain

from random import Random


class Task:

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = Random(seed)

    def reset(self, env):
        pass

    def is_done(self, env):
        return False

    def update(self, env, actions):
        pass

    def task_obs(self, agent):
        return {}

    def desc(self):
        return ''


def spread(env, x0, x1, y0, y1, rng, symbol='A'):
    # place agent
    used_positions = set()
    for name, _ in env.agents.items():
        while True:
            x = rng.randint(x0, x1)
            y = rng.randint(y0, y1)
            pos = (y, x)
            if pos not in used_positions and env.grid[y, x] == '.':
                used_positions.add(pos)
                # self.grid[y, x] = name
                # self.agents_pos[name] = {'x': x, 'y': y}
                agent_mesh = Mesh(pos, np.array([[symbol]], dtype=str), False, name=name)
                # agent_mesh.mass = 0
                agent_mesh.place(env.grid, env.mesh_map)
                env.agent_meshes[name] = agent_mesh
                env.meshes.append(agent_mesh)
                break

class Transport(Task):

    boxes = [
        {
            'shape': [
                ['B', 'B', 'B', '.', '.'],
                ['B', 'B', 'B', '.', '.'],
                ['B', 'B', 'B', '.', '.'],
                ['.', 'B', 'B', 'B', 'B'],
            ],
            'pos': (-4, -5),
            'gap': ((-1, None), (-4, None)),
        },
        {
            'shape': [
                ['.', '.', 'B', 'B', 'B'],
                ['.', '.', 'B', 'B', 'B'],
                ['.', '.', 'B', 'B', 'B'],
                ['B', 'B', 'B', 'B', '.'],
            ],
            'pos': (-4, 0),
            'gap': ((-1, None), (None, 4)),
        },
        {
            'shape': [
                ['.', '.', '.', 'B'],
                ['.', '.', '.', 'B'],
                ['B', 'B', 'B', 'B'],
                ['B', 'B', 'B', 'B'],
                ['B', 'B', 'B', '.'],
            ],
            'pos': (0, -4),
            'gap': ((None, 4), (-1, None)),
        },
        {
            'shape': [
                ['B', 'B', 'B', '.'],
                ['B', 'B', 'B', 'B'],
                ['B', 'B', 'B', 'B'],
                ['.', '.', '.', 'B'],
                ['.', '.', '.', 'B'],
            ],
            'pos': (-5, -4),
            'gap': ((-4, None), (-1, None)),
        },
        {
            'shape': [
                ['B', '.', '.', '.'],
                ['B', '.', '.', '.'],
                ['B', 'B', 'B', 'B'],
                ['B', 'B', 'B', 'B'],
                ['.', 'B', 'B', 'B'],
            ],
            'pos': (0, 0),
            'gap': ((None, 4), (None, 1)),
        },
        {
            'shape': [
                ['.', 'B', 'B', 'B'],
                ['B', 'B', 'B', 'B'],
                ['B', 'B', 'B', 'B'],
                ['B', '.', '.', '.'],
                ['B', '.', '.', '.'],
            ],
            'pos': (-5, 0),
            'gap': ((-4, None), (None, 1)),
        },
        {
            'shape': [
                ['B', 'B', 'B', 'B', '.'],
                ['.', '.', 'B', 'B', 'B'],
                ['.', '.', 'B', 'B', 'B'],
                ['.', '.', 'B', 'B', 'B'],
            ],
            'pos': (0, 0),
            'gap': ((None, 1), (None, 4)),
        },
        {
            'shape': [
                ['.', 'B', 'B', 'B', 'B'],
                ['B', 'B', 'B', '.', '.'],
                ['B', 'B', 'B', '.', '.'],
                ['B', 'B', 'B', '.', '.'],
            ],
            'pos': (0, -5),
            'gap': ((None, 1), (-4, None)),
        }
    ]

    def __init__(self, seed=42):
        super().__init__(seed)
        self.escaped = set()
        self.score = 0

    def reset(self, env):
        self.escaped = set()
        box = self.rng.choice(Transport.boxes)
        box_mesh = Mesh(
            (
                (box['pos'][0] + env.grid.shape[0]) % env.grid.shape[0],
                (box['pos'][1] + env.grid.shape[1]) % env.grid.shape[1]
            ),
            np.array(box['shape'], dtype=str), False, name='B'
        )
        wall_shape = np.full_like(env.grid, 'W', dtype=str)
        wall_shape[1:-1, 1:-1] = '.'
        wall_shape[box['gap'][0][0]:box['gap'][0][1], box['gap'][1][0]:box['gap'][1][1]] = '.'

        # Use EnhancedCapacityMesh consistently for walls.
        wall_mesh = create_terrain('wall', (0, 0), shape=wall_shape)
        wall_mesh.place(env.grid, env.mesh_map)
        env.meshes.append(wall_mesh)
        box_mesh.place(env.grid, env.mesh_map)
        box_mesh.mass = 5
        env.meshes.append(box_mesh)

        spread(env, 1, env.width - 2, 1, env.height - 2, self.rng)
        self.score = 0

    def is_done(self, env):
        # Check if all agents have exited through the exit
        all_out = True
        for name, mesh in env.agent_meshes.items():
            y, x = mesh.pos
            if 0 <= y < env.height and 0 <= x < env.width:
                all_out = False
        return all_out

    def update(self, env, actions):
        for name, mesh in env.agent_meshes.items():
            if name in self.escaped:
                continue
            y, x = mesh.pos
            if not (0 <= y < env.height and 0 <= x < env.width):
                self.escaped.add(name)
                self.score += (env.max_round - env.round) / env.max_round

    def task_obs(self, agent):
        if agent is None:
            return {'score': self.score}
        return {}

    def desc(self):
        return (
            "The boundary of the map is surrounded by walls (denoted as W), with a gap leading to the outside of the map (denoted as '*'). The gap is blocked by an obstacle (denoted as B).\n"
            "The goal is to first locate the obstacle (B), then have five robots simultaneously push it through the exit, and finally escape to the outside of the map (denoted as '*')."
        )


class Flocking(Task):
    def __init__(self, seed=42):
        super().__init__()
        self.target_shape = None
        self.shape_desc = ''
        self.score = 0
        self.init_dis = 0
        self.cur_dis = 1e10

    def emd(self, grid):
        # Extract the coordinates of the target and source points
        tgt = np.argwhere(self.target_shape == 'A')  # shape (n,2)
        src = np.argwhere(grid == 'A')  # shape (m,2)
        n, m = len(tgt), len(src)
        if n == 0:
            return 0, 0, 0

        # Pre-compute the difference matrices D_x, D_y
        # D_x[i,j] = src[i,0] - tgt[j,0], D_y[i,j] = src[i,1] - tgt[j,1]
        D_x = src[:, 0:1] - tgt[:, 0]  # shape (m,n) via broadcasting
        D_y = src[:, 1:2] - tgt[:, 1]  # shape (m,n)

        # Generate all candidate translation vectors (dx,dy)
        # Note: Use the D_x, D_y matrices to quickly extract all differences
        cand_set = set(zip(D_x.ravel(), D_y.ravel()))

        best_cost = np.inf
        best_dx = best_dy = 0

        # Enumerate the candidates
        for dx, dy in cand_set:
            # For each candidate translation, the cost matrix is |D_x-dx|+|D_y-dy|
            cost = np.abs(D_x - dx) + np.abs(D_y - dy)
            # Optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            total = cost[row_ind, col_ind].sum()
            if total < best_cost:
                best_cost = total
                best_dx, best_dy = dx, dy

        return int(best_cost)

    def reset(self, env):
        wall_shape = np.full_like(env.grid, 'W', dtype=str)
        wall_shape[1:-1, 1:-1] = '.'
        # Use EnhancedCapacityMesh consistently for walls.
        wall_mesh = create_terrain('wall', (0, 0), shape=wall_shape)
        wall_mesh.place(env.grid, env.mesh_map)
        env.meshes.append(wall_mesh)
        num_agents = len(env.agents)
        self.target_shape = np.full((num_agents // 4 + 1, num_agents // 4 + 1), 'A', dtype=str)
        self.target_shape[1:-1, 1:-1] = '.'
        self.shape_desc = '\n'.join([' '.join([f'{self.target_shape[i, j]}'
                                     for j in range(self.target_shape.shape[1])])
                                     for i in range(self.target_shape.shape[0])])
        spread(env, 1, env.width - 2, 1, env.height - 2, self.rng)
        self.init_dis = self.emd(env.grid)
        self.cur_dis = self.init_dis
        self.score = 0

    def is_done(self, env):
        return self.cur_dis == 0

    def update(self, env, actions):
        self.cur_dis = self.emd(env.grid)
        new_score = self.init_dis - self.cur_dis
        self.score = max(new_score, self.score)
        print(f'init dis: {self.init_dis}, current dis: {self.cur_dis}, score: {self.score}')

    def task_obs(self, agent):
        if agent is None:
            return {'score': self.score}
        return {}

    def desc(self):
        return (f'Your target is to properly align in the map and form'
                f'the following shape with other agents (A for agent):\n'
                f'{self.shape_desc}')


class Pursuit(Task):
    def __init__(self, seed=42):
        super().__init__(seed)
        self.prey = None
        self.score = 0

    def reset(self, env):
        wall_shape = np.full_like(env.grid, 'W', dtype=str)
        wall_shape[1:-1, 1:-1] = '.'
        # Use EnhancedCapacityMesh consistently for walls.
        wall_mesh = create_terrain('wall', (0, 0), shape=wall_shape)
        wall_mesh.place(env.grid, env.mesh_map)
        prey_pos = (self.rng.randint(env.height // 2 + 1, env.height - 2),
                    self.rng.randint(env.width // 2 + 1, env.width - 2))
        self.prey = Mesh(prey_pos, np.full((1, 1), 'P', dtype=str),
                         static=False, name='P')
        self.prey.place(env.grid, env.mesh_map)
        spread(env, 1, max(env.height // 2, 5), 1, max(env.width // 2, 5), self.rng)
        self.score = 0

    def subview(self, env, i, j):
        return env.grid[max(i - 4, 0):i + 4, max(j - 4, 0):j + 4]

    def danger(self, env, i, j):
        subview = self.subview(env, i, j)
        view_agents = (subview == 'A').sum()
        view_walls = (subview == 'W').sum()
        return view_agents + view_walls * 0.9

    def can_move_to(self, env, i, j):
        return 0 <= i < env.height and 0 <= j < env.width and env.grid[i, j] == '.'

    def update(self, env, actions):
        cnt = 0
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            i, j = self.prey.pos
            if 0 <= i + di < env.height and 0 <= j + dj < env.width:
                cnt += env.grid[i + di, j + dj] != '.'
        if cnt == 4:
            self.score += 1
            danger = 1e6
            self.prey.remove(env.grid, env.mesh_map)
            self.prey = Mesh((0, 0), np.full((1, 1), 'P', dtype=str),
                             static=True, name='P')
            t = 0
            while t < env.height * env.width:
                i = self.rng.randint(1, env.height - 2)
                j = self.rng.randint(1, env.width - 2)
                if env.grid[i, j] != '.':
                    continue
                t += 1
                view_danger = self.danger(env, i, j)
                if view_danger < danger:
                    danger = view_danger
                    self.prey.pos = i, j
            self.prey.place(env.grid, env.mesh_map)
        else:
            i, j = self.prey.pos
            target = 0, 0, 0, 0
            danger = self.danger(env, i + di, j + dj)
            for di1, dj1 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                for di2, dj2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    di = di1 + di2
                    dj = dj1 + dj2
                    if (not self.can_move_to(env, i + di, j + dj)
                        or not self.can_move_to(env, i + di1, j + dj1)
                        or not self.can_move_to(env, i + di2, j + dj2)):
                        continue
                    view_danger = self.danger(env, i + di, j + dj)
                    if view_danger < danger:
                        danger = view_danger
                        target = di1, dj1, di2, dj2
            if target != (0, 0, 0, 0):
                self.prey.move(env.grid, env.mesh_map, target[0], target[1])
                self.prey.move(env.grid, env.mesh_map, target[2], target[3])

    def task_obs(self, agent):
        return {'score': self.score}

    def desc(self):
        return (f'Target: There is a prey (denoted as P) somewhere on the map. '
                f'Your target is to cooperate with other agents to chase the prey down. '
                f'A prey is caught only when it is surrounded by 4 agents (A) or walls (W):\n'
                f'.,A/W,.\nA/W,P,A/W\n.,A/W,.\n'
                f'The prey moves faster than all of your kinds (2 steps each round). '
                f'Once it is caught, you will earn a point, and a new prey will spawn somewhere on the map.')


class Synchronization(Task):
    def __init__(self, seed=42):
        super().__init__(seed)
        self.agent_state = {}
        self.score = 0
        self.prev_state = -1

    def reset(self, env):
        wall_shape = np.full_like(env.grid, 'W', dtype=str)
        wall_shape[1:-1, 1:-1] = '.'
        # Use EnhancedCapacityMesh consistently for walls.
        wall_mesh = create_terrain('wall', (0, 0), shape=wall_shape)
        wall_mesh.place(env.grid, env.mesh_map)
        self.score = 0
        self.agent_state = {name: bool(self.rng.randint(0, 1)) for name in env.agents}
        self.prev_state = -1
        spread(env, 1, env.width - 2, 1, env.height - 2, self.rng)
        for agent in env.agents.values():
            agent.valid_actions.append('SWITCH')
        for name in env.agents:
            i, j = env.agent_meshes[name].pos
            env.grid[i, j] = 'A' if self.agent_state[name] else 'a'
        self.score = 0

    def task_obs(self, agent):
        if agent is None:
            return {'score': self.score}
        return {'light_on': self.agent_state[agent.name]}

    def update(self, env, actions):
        for name, action in actions.items():
            if action.get('move', '') == 'SWITCH':
                self.agent_state[name] = not (self.agent_state[name])
            i, j = env.agent_meshes[name].pos
            env.grid[i, j] = 'A' if self.agent_state[name] else 'a'
        self.update_score()

    def update_score(self):
        state = sum(self.agent_state.values())
        if (state == len(self.agent_state) or state == 0) and state != self.prev_state:
            self.prev_state = state
            self.score += 1

    def desc(self):
        return ('Target: Each of you agents has a light that has two possible states: on/off. '
                'All of you need to synchronize your light states, i.e., all on or all off. '
                'You earn one point when all agents (including the ones out of your sight) have the same light states. '
                'And after this, all agents must synchronize their light states to the opposite in ordered to earn one another point. '
                'That means if you earned a point for synchronizing light states to all on, '
                'next time you earn a point is when light states are all off (the opposite), vice versa.\n'
                'There is one more action: SWITCH. '
                'When you choose SWITCH, light states will be changed to the opposite. '
                'In your view, agents with lights on are denoted in the regular way, '
                'while agents with lights off start with a dollar sign ($).')


class Foraging(Task):
    def __init__(self, seed=42, food_capacities=None, nest_capacities=None, target_items=None):
        super().__init__(seed)
        self.food_mesh = None
        self.nest_mesh = None
        # Use resource_state to support multiple resource types.
        # None means carrying nothing; a resource name identifies the carried type.
        self.resource_state = {}
        self.score = 0
        # Configure default capacities.
        self.food_capacities = food_capacities or [5, 5, 5, 5]  # Four food sources, each with capacity 5.
        self.nest_capacities = nest_capacities or [5, 5, 5, 5]  # Four nests, each with capacity 5.

        # Target items: required quantities for each resource type.
        self.target_items = target_items or {'wood': 3, 'sand': 2, 'stone': 1}

        # Additional resource type definitions.
        self.resource_types = {
            'wood': {'name': 'wood', 'display': 'w', 'score': 1},
            'sand': {'name': 'sand', 'display': 's', 'score': 1},
            'stone': {'name': 'stone', 'display': 's', 'score': 1}  # Stone also uses s here; mesh_type distinguishes it.
        }
        print(f"DEBUG: Initializing Foraging with {len(self.resource_types)} resource types: {list(self.resource_types.keys())}")
        print(f"DEBUG: Target item configuration: {self.target_items}")

    def reset(self, env):
        wall_shape = np.full_like(env.grid, 'W', dtype=str)
        wall_shape[1:-1, 1:-1] = '.'

        food_i = self.rng.choice([1, env.height - 3])
        food_j = self.rng.choice([1, env.width - 3])
        nest_i = env.height - 3 if food_i == 1 else 1
        nest_j = env.width - 3 if food_j == 1 else 1

        if self.rng.randint(0, 1) == 0:
            walls = [n for n in range(4, env.height - 4, 4)]
            if len(walls) % 2 == 1:
                nest_j = 1
                food_j = 1
            for n, i in enumerate(walls):
                if n % 2 == 0:
                    wall_shape[i, :-env.width // 3 + 1] = 'W'
                else:
                    wall_shape[i, env.width // 3:] = 'W'
        else:
            walls = [n for n in range(4, env.width - 4, 4)]
            if len(walls) % 2 == 1:
                nest_i = 1
                food_i = 1
            for n, j in enumerate(walls):
                if n % 2 == 0:
                    wall_shape[:-env.height // 3 + 1, j] = 'W'
                else:
                    wall_shape[env.height // 3:, j] = 'W'

        # Remove the original nest wall shape; create separate CapacityMesh objects below.

        # Use EnhancedCapacityMesh consistently for walls.
        wall_mesh = create_terrain('wall', (0, 0), shape=wall_shape)
        wall_mesh.place(env.grid, env.mesh_map)
        env.meshes.append(wall_mesh)

        # Create food CapacityMesh objects with custom capacities.
        self.food_meshes = []
        food_positions = []

        # Create one food position per configured capacity.
        num_foods = len(self.food_capacities)
        if num_foods == 1:
            food_positions = [(food_i, food_j)]
        elif num_foods == 2:
            food_positions = [(food_i, food_j), (food_i, food_j + 1)]
        elif num_foods == 3:
            food_positions = [(food_i, food_j), (food_i, food_j + 1), (food_i + 1, food_j)]
        else:  # Four or more.
            for i_offset in range(2):
                for j_offset in range(2):
                    if len(food_positions) < num_foods:
                        food_positions.append((food_i + i_offset, food_j + j_offset))

        # Match position and capacity counts.
        if len(food_positions) > num_foods:
            food_positions = food_positions[:num_foods]
        elif len(food_positions) < num_foods:
            # Extend capacity configuration with defaults when counts differ.
            self.food_capacities = self.food_capacities[:len(food_positions)]

        # Skip food generation; generate only wood, sand, and stone.
        # for idx, (pos, capacity) in enumerate(zip(food_positions, self.food_capacities)):
        #     food_mesh = CapacityMesh(pos,
        #                             np.full((1, 1), 'f', dtype=str),
        #                             static=True, name='f', capacity=capacity)
        #     food_mesh.place(env.grid, env.mesh_map)
        #     env.meshes.append(food_mesh)
        #     self.food_meshes.append(food_mesh)

        # Create table CapacityMesh objects with custom capacities.
        self.table_meshes = []
        table_positions = []

        # Create one table position per configured capacity.
        num_tables = len(self.nest_capacities)
        if num_tables == 1:
            table_positions = [(nest_i, nest_j)]
        elif num_tables == 2:
            table_positions = [(nest_i, nest_j), (nest_i, nest_j + 1)]
        elif num_tables == 3:
            table_positions = [(nest_i, nest_j), (nest_i, nest_j + 1), (nest_i + 1, nest_j)]
        else:  # Four or more.
            for i_offset in range(2):
                for j_offset in range(2):
                    if len(table_positions) < num_tables:
                        table_positions.append((nest_i + i_offset, nest_j + j_offset))

        # Match position and capacity counts.
        if len(table_positions) > num_tables:
            table_positions = table_positions[:num_tables]
        elif len(table_positions) < num_tables:
            # Extend capacity configuration with defaults when counts differ.
            self.nest_capacities = self.nest_capacities[:len(table_positions)]

        for idx, (pos, capacity) in enumerate(zip(table_positions, self.nest_capacities)):
            table_terrain = create_terrain('table', pos, capacity=capacity)
            table_terrain.place(env.grid, env.mesh_map)
            env.meshes.append(table_terrain)
            self.table_meshes.append(table_terrain)
            print(f"DEBUG: Creating table (T) at {pos}, capacity={capacity}")

        # Use EnhancedCapacityMesh consistently for mountains.
        for _ in range(1): # Add 1 mountain tile
            while True:
                x = self.rng.randint(1, env.width - 2)
                y = self.rng.randint(1, env.height - 2)
                if env.grid[y, x] == '.':
                    mountain_mesh = create_terrain('mountain', (y, x))
                    mountain_mesh.place(env.grid, env.mesh_map)
                    env.meshes.append(mountain_mesh)
                    break

        # Use EnhancedCapacityMesh consistently for rivers.
        for _ in range(1): # Add 1 river tile
            while True:
                x = self.rng.randint(1, env.width - 2)
                y = self.rng.randint(1, env.height - 2)
                if env.grid[y, x] == '.':
                    river_mesh = create_terrain('river', (y, x))
                    river_mesh.place(env.grid, env.mesh_map)
                    env.meshes.append(river_mesh)
                    break

        # Use EnhancedCapacityMesh consistently for grass.
        grass_count = self.rng.randint(3, 5)  # Generate 3-5 grass cells randomly.
        grass_generated = 0
        for _ in range(grass_count):
            attempts = 0
            while attempts < 50:  # Avoid an infinite loop.
                x = self.rng.randint(1, env.width - 2)
                y = self.rng.randint(1, env.height - 2)
                if env.grid[y, x] == '.':
                    grass_mesh = create_terrain('grass', (y, x))
                    grass_mesh.place(env.grid, env.mesh_map)
                    env.meshes.append(grass_mesh)
                    grass_generated += 1
                    break
                attempts += 1

        # Use EnhancedCapacityMesh consistently for forests.
        forest_count = self.rng.randint(2, 4)  # Generate 2-4 forest cells randomly.
        forest_generated = 0
        for _ in range(forest_count):
            attempts = 0
            while attempts < 50:  # Avoid an infinite loop.
                x = self.rng.randint(1, env.width - 2)
                y = self.rng.randint(1, env.height - 2)
                if env.grid[y, x] == '.':
                    forest_mesh = create_terrain('forest', (y, x))
                    forest_mesh.place(env.grid, env.mesh_map)
                    env.meshes.append(forest_mesh)
                    forest_generated += 1
                    break
                attempts += 1

        # Generate resources in two layers: terrain (resource) and items (wood/sand/stone).
        print(f"DEBUG: Generating two-layer resources: terrain (resource) + items (wood/sand/stone)")

        # Define item types and quantities to generate.
        resource_items = [
            {'item_type': 'wood', 'count': self.rng.randint(2, 4), 'initial_quantity': 2},
            {'item_type': 'sand', 'count': self.rng.randint(2, 4), 'initial_quantity': 2},
            {'item_type': 'stone', 'count': self.rng.randint(2, 4), 'initial_quantity': 2}
        ]

        total_generated = 0
        for resource_def in resource_items:
            item_type = resource_def['item_type']
            count = resource_def['count']
            initial_qty = resource_def['initial_quantity']

            generated = 0
            print(f"DEBUG: Generating {count} resource cells for {item_type}")

            for _ in range(count):
                attempts = 0
                while attempts < 50:
                    x = self.rng.randint(1, env.width - 2)
                    y = self.rng.randint(1, env.height - 2)
                    if env.grid[y, x] == '.':
                        # Create resource terrain as the lower layer.
                        resource_terrain = create_terrain('resource', (y, x), capacity=5)
                        resource_terrain.place(env.grid, env.mesh_map)
                        env.meshes.append(resource_terrain)

                        # Place initial items on resource terrain as the upper layer.
                        success = resource_terrain.add_item(item_type, initial_qty)

                        # Also update the environment's item mapping.
                        if success:
                            env.add_item_at_position((y, x), item_type, initial_qty)
                            generated += 1
                            total_generated += 1
                            print(f"DEBUG: Position ({y}, {x}) - terrain: resource, items: {item_type}x{initial_qty}")
                        else:
                            print(f"DEBUG: Warning - at ({y}, {x}), failed to add {item_type}")
                        break
                    attempts += 1

            print(f"DEBUG: Generated {generated} resource cells for {item_type} (target: {count})")

        print(f"DEBUG: Two-layer resource generation complete: {total_generated} resource cells")

        # Initialize resource state to None (carrying nothing).
        self.resource_state = {name: None for name in env.agents}
        print(f"DEBUG: Initialized resource state for {len(env.agents)} agents to None (carrying nothing)")

        spread(env, 1, env.width - 2, 1, env.height - 2, self.rng, 'a')

        self.score = 0

    def task_obs(self, agent):
        if agent is None:
            return {'score': self.score}

        # Return information about the agent's carried resource.
        carrying_resource = self.resource_state[agent.name]
        # Preserve carrying_food compatibility while adding resource state information.
        return {
            # 'carrying_food': carrying_resource == 'f',  # Backward compatibility.
            'carrying_resource': carrying_resource,  # Resource state.
            'carrying_resource_name': self.resource_types.get(carrying_resource, {}).get('name', 'none') if carrying_resource else 'none'
        }

    def update(self, env, actions):
        print(f"DEBUG: Updating Foraging task for {len(env.agent_meshes)} agents")
        print(f"DEBUG: Current action list: {list(actions.keys())}")

        # Explicit actions: environment.pick_up/put_down already handle interactions.
        # Only update the grid after movement here.

        for name, agent_mesh in env.agent_meshes.items():
            # Ensure correct agent symbols in the grid.
            current_resource = self.resource_state[name]
            if current_resource is not None:
                agent_mesh.shape[:, :] = 'A'  # Uppercase A when carrying resources.
            else:
                agent_mesh.shape[:, :] = 'a'  # Lowercase a when carrying nothing.

            # Update grid display.
            agent_mesh.place(env.grid, None)

            print(f"DEBUG: Agent {name} state updated - carried resource: {current_resource}, position: {agent_mesh.pos}")

        # Update the score at the end of each round.
        self.update_score(env)

    def count_items_on_tables(self, env):
        """Count items on all tables."""
        item_counts = {}

        # Inspect all mesh objects.
        for mesh in env.meshes:
            if (hasattr(mesh, 'terrain_type') and
                mesh.terrain_type == 'table' and
                hasattr(mesh, 'items')):
                # Count all items on this table.
                for item in mesh.items:
                    item_counts[item.item_type] = item_counts.get(item.item_type, 0) + item.quantity

        print(f"DEBUG: Table item counts: {item_counts}")
        return item_counts

    def update_score(self, env):
        """Calculate the score from target item quantities."""
        total_items = self.count_items_on_tables(env)

        new_score = 0
        for item_type, target_count in self.target_items.items():
            actual_count = total_items.get(item_type, 0)
            # Count only quantities up to the targets; excess items earn no points.
            new_score += min(actual_count, target_count)

        # Update only when the score changes.
        if new_score != self.score:
            old_score = self.score
            self.score = new_score
            print(f"DEBUG: 🎯 Score updated {old_score} -> {new_score} (targets: {self.target_items}, actual: {total_items})")

    def desc(self):
        return ('Target: You are in a maze where walls (W) blocks your way. '
                f'Your task is to collect resources and bring them to tables (T) to earn points. Target items: {self.target_items} '
                'SCORING SYSTEM: Score is based on TOTAL items on ALL tables. Each item type contributes up to its target amount. '
                'ARCHITECTURE: Two-layer system - Terrain layer (empty/mountain/river/wall/resource/table/forest/grass) + Item layer (wood/sand/stone). '
                'Available resources: wood, sand, stone. These are ITEMS that can be placed on any passable terrain. '
                'You can only carry ONE resource type at a time. '
                'Use PICK_UP action to collect items from any terrain, then use PUT_DOWN action on a table (T) to contribute to the score. '
                'Table is represented as T or T(items) where items show current contents. '
                'You can move through empty (.), grass (G), and forest (F). Mountains (M), Rivers (R) and Walls (W) block movement. '
                'Important: You must explicitly use PICK_UP and PUT_DOWN actions for interaction! ')
