"""
Terrain adapter integrating EnhancedCapacityMesh into the existing environment.
Provides backward-compatible interfaces to keep existing code working.
"""

import numpy as np
from masbench.enhanced_terrain import EnhancedCapacityMesh, create_terrain

class TerrainAdapter:
    """Manage EnhancedCapacityMesh integration with the existing system."""

    def __init__(self, env):
        """Initialize the adapter."""
        print(f"DEBUG: Initializing TerrainAdapter; environment: {env}")
        self.env = env
        self.enhanced_mesh_map = np.full_like(env.mesh_map, None, dtype=object)
        self._setup_terrain_mapping()
        print("DEBUG: TerrainAdapter initialized")

    def _setup_terrain_mapping(self):
        """Map terrain symbols to types in the two-layer architecture."""
        print("DEBUG: Configuring terrain symbols for the two-layer architecture")
        self.symbol_to_terrain = {
            '.': 'empty',       # Empty ground, terrain layer.
            'M': 'mountain',    # Mountain, terrain layer.
            'R': 'river',       # River, terrain layer.
            'W': 'wall',        # Wall, terrain layer.
            'T': 'table',       # Table, terrain layer.
            'F': 'forest',      # Forest, terrain layer.
            'G': 'grass',       # Grass, terrain layer.
            'r': 'resource',    # Resource terrain, supports the item layer.
            # Replace legacy item-specific symbols with resource terrain plus a separate item layer.
        }
        print(f"DEBUG: Symbol mapping (two-layer architecture): {self.symbol_to_terrain}")

    def convert_existing_meshes(self):
        """Convert CapacityMesh to EnhancedCapacityMesh and create permanent terrain at every position."""
        print("DEBUG: [TERRAIN_ADAPTER] Converting existing meshes to EnhancedCapacityMesh and creating a complete terrain map")

        converted_count = 0
        created_count = 0

        for i in range(self.env.mesh_map.shape[0]):
            for j in range(self.env.mesh_map.shape[1]):
                mesh = self.env.mesh_map[i, j]

                if mesh is not None and isinstance(mesh, EnhancedCapacityMesh):
                    # Reuse existing EnhancedCapacityMesh references and ensure they are in meshes.
                    print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): already EnhancedCapacityMesh, mesh={mesh}")
                    if hasattr(mesh, 'node'):
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): EnhancedCapacityMesh.node={mesh.node}")
                    else:
                        print(f"DEBUG: [TERRAIN_ADAPTER] ❌ Position ({i}, {j}): EnhancedCapacityMesh has no node attribute!")
                    self.enhanced_mesh_map[i, j] = mesh
                    # Ensure membership in the meshes list.
                    if mesh not in self.env.meshes:
                        self.env.meshes.append(mesh)
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): added to meshes")
                    converted_count += 1
                elif mesh is not None:
                    # Mesh requiring conversion.
                    print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): converting mesh={mesh}, type={type(mesh).__name__}")
                    if hasattr(mesh, 'node'):
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): original mesh.node={mesh.node}")
                    terrain_type = self._infer_terrain_type(mesh)
                    if terrain_type:
                        enhanced_mesh = self._convert_mesh_to_enhanced(mesh, terrain_type)
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): converted enhanced_mesh={enhanced_mesh}")
                        if hasattr(enhanced_mesh, 'node'):
                            print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): converted enhanced_mesh.node={enhanced_mesh.node}")
                        self.enhanced_mesh_map[i, j] = enhanced_mesh
                        # Update the original mesh_map reference and add to meshes.
                        self.env.mesh_map[i, j] = enhanced_mesh
                        self.env.meshes.append(enhanced_mesh)  # Include in meshes for rendering.
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): updated mesh_map reference and added to meshes")
                        converted_count += 1
                else:
                    # Create permanent terrain for empty positions, checking mesh_map first.
                    mesh_in_map = self.env.mesh_map[i, j]
                    if mesh_in_map is not None:
                        # Use mesh type information when an object exists in mesh_map.
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): mesh_map contains {mesh_in_map}, type={type(mesh_in_map).__name__}")
                        if hasattr(mesh_in_map, 'name'):
                            terrain_type = self.symbol_to_terrain.get(mesh_in_map.name, 'empty')
                            print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): terrain type inferred from mesh name: {terrain_type}")
                        else:
                            terrain_type = 'empty'
                            print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): mesh has no name; defaulting to empty")
                    else:
                        # Infer from the grid symbol only when no mesh object exists.
                        grid_symbol = self.env.grid[i, j]
                        terrain_type = self.symbol_to_terrain.get(grid_symbol, 'empty')
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): mesh_map empty; inferred from grid symbol '{grid_symbol}' -> {terrain_type}")

                    # Create permanent terrain.
                    print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): creating permanent terrain, final type={terrain_type}")
                    permanent_terrain = create_terrain(terrain_type, (i, j))
                    print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): created permanent terrain={permanent_terrain}")
                    if hasattr(permanent_terrain, 'node'):
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): new terrain.node={permanent_terrain.node}")
                    else:
                        print(f"DEBUG: [TERRAIN_ADAPTER] ❌ Position ({i}, {j}): new terrain has no node attribute!")
                    # Update both enhanced_mesh_map and the original mesh_map, and add to meshes.
                    self.enhanced_mesh_map[i, j] = permanent_terrain
                    self.env.mesh_map[i, j] = permanent_terrain
                    self.env.meshes.append(permanent_terrain)  # Include in meshes for rendering.
                    created_count += 1

        print(f"DEBUG: [TERRAIN_ADAPTER] Terrain conversion complete: converted {converted_count} meshes, created {created_count} terrain cells")

        # Ensure every newly created EnhancedCapacityMesh has a valid node.
        print("DEBUG: [TERRAIN_ADAPTER] Ensuring valid node attributes on all EnhancedCapacityMesh objects")
        from masbench.physics import Node
        for i in range(self.enhanced_mesh_map.shape[0]):
            for j in range(self.enhanced_mesh_map.shape[1]):
                mesh = self.enhanced_mesh_map[i, j]
                if mesh is not None and isinstance(mesh, EnhancedCapacityMesh):
                    if mesh.node is None:
                        print(f"DEBUG: [TERRAIN_ADAPTER] Repairing null mesh.node at ({i}, {j})")
                        mesh.node = Node(mesh)
                        print(f"DEBUG: [TERRAIN_ADAPTER] Position ({i}, {j}): created mesh.node={mesh.node}")
        print("DEBUG: [TERRAIN_ADAPTER] Node attributes repaired for all EnhancedCapacityMesh objects")

    def _infer_terrain_type(self, mesh):
        """Infer the terrain type of a mesh."""
        print(f"DEBUG: Inferring terrain type for mesh name: {mesh.name}")

        if hasattr(mesh, 'name') and mesh.name in self.symbol_to_terrain:
            terrain_type = self.symbol_to_terrain[mesh.name]
            print(f"DEBUG: Terrain type inferred from name: {terrain_type}")
            return terrain_type

        # Infer from other attributes.
        if hasattr(mesh, 'static') and mesh.static:
            if hasattr(mesh, 'capacity') and mesh.capacity > 0:
                print("DEBUG: Inferred resource source type")
                return 'resource_source'
            else:
                print("DEBUG: Inferred static obstacle")
                return 'wall'

        print("DEBUG: Cannot infer terrain type; defaulting to empty")
        return 'empty'

    def _convert_mesh_to_enhanced(self, mesh, terrain_type):
        """Convert a standard mesh to EnhancedCapacityMesh."""
        print(f"DEBUG: Converting mesh to EnhancedCapacityMesh, type={terrain_type}")

        # Get configuration parameters.
        config = EnhancedCapacityMesh.TERRAIN_CONFIG.get(terrain_type,
                                                        EnhancedCapacityMesh.TERRAIN_CONFIG['empty'])

        # Create the enhanced mesh.
        enhanced_mesh = EnhancedCapacityMesh(
            terrain_type=terrain_type,
            position=mesh.pos,
            shape=mesh.shape,
            capacity=getattr(mesh, 'capacity', config[3]),
            name=mesh.name
        )

        # Copy other attributes.
        enhanced_mesh.static = mesh.static
        enhanced_mesh.mass = mesh.mass

        print(f"DEBUG: Mesh conversion complete: {enhanced_mesh}")
        return enhanced_mesh

    def create_terrain(self, terrain_type, position, **kwargs):
        """Create terrain and add it to the environment."""
        print(f"DEBUG: Creating terrain: type={terrain_type}, position={position}")

        terrain = create_terrain(terrain_type, position, **kwargs)

        # Place in the environment.
        terrain.place(self.env.grid, self.env.mesh_map)
        self.env.meshes.append(terrain)

        # Update the enhanced mesh mapping.
        y, x = position
        if 0 <= y < self.enhanced_mesh_map.shape[0] and 0 <= x < self.enhanced_mesh_map.shape[1]:
            self.enhanced_mesh_map[y, x] = terrain

        print(f"DEBUG: Terrain created and placed: {terrain}")
        return terrain

    def get_terrain_at(self, position):
        """Return enhanced terrain at the specified position."""
        y, x = position
        if 0 <= y < self.enhanced_mesh_map.shape[0] and 0 <= x < self.enhanced_mesh_map.shape[1]:
            terrain = self.enhanced_mesh_map[y, x]
            print(f"DEBUG: Permanent terrain at {position}: {terrain}")
            return terrain
        print(f"DEBUG: Position {position} is out of bounds")
        return None

    def is_traversable(self, position):
        """Check whether a position is passable."""
        terrain = self.get_terrain_at(position)
        if terrain:
            traversable = terrain.traversable
            print(f"DEBUG: Position {position}, terrain={getattr(terrain, 'terrain_type', 'unknown')}, passable={traversable}")
            return traversable

        # Default behavior: inspect the grid character.
        y, x = position
        if 0 <= y < self.env.grid.shape[0] and 0 <= x < self.env.grid.shape[1]:
            cell = self.env.grid[y, x]
            traversable = cell == '.'  # Only empty ground is passable.
            print(f"DEBUG: Checking passability from grid character '{cell}': {traversable}")
            return traversable

        print(f"DEBUG: Position {position} is invalid and impassable")
        return False

    def add_item_to_terrain(self, position, item_type, quantity=1):
        """Add items to terrain at the specified position."""
        print(f"DEBUG: Adding items at {position}: {item_type} x{quantity}")

        terrain = self.get_terrain_at(position)
        if terrain and hasattr(terrain, 'add_item'):
            success = terrain.add_item(item_type, quantity)
            print(f"DEBUG: Item addition result: {success}")
            return success

        print("DEBUG: Cannot add items - terrain absent or unsupported")
        return False

    def remove_item_from_terrain(self, position, item_type, quantity=1):
        """Remove items from terrain at the specified position."""
        print(f"DEBUG: Removing items from {position}: {item_type} x{quantity}")

        terrain = self.get_terrain_at(position)
        if terrain and hasattr(terrain, 'remove_item'):
            success = terrain.remove_item(item_type, quantity)
            print(f"DEBUG: Item removal result: {success}")
            return success

        print("DEBUG: Cannot remove items - terrain absent or unsupported")
        return False

    def get_terrain_info(self, position):
        """Return detailed terrain information."""
        print(f"DEBUG: Getting terrain information at {position}")

        terrain = self.get_terrain_at(position)
        if terrain and hasattr(terrain, 'get_terrain_info'):
            info = terrain.get_terrain_info()
            print(f"DEBUG: Terrain information: {info}")
            return info

        # Return basic information.
        y, x = position
        if 0 <= y < self.env.grid.shape[0] and 0 <= x < self.env.grid.shape[1]:
            info = {
                'terrain_type': 'unknown',
                'position': position,
                'grid_symbol': self.env.grid[y, x],
                'has_mesh': self.env.mesh_map[y, x] is not None,
                'traversable': self.env.grid[y, x] == '.',
                'can_store_items': False
            }
            print(f"DEBUG: Basic terrain information: {info}")
            return info

        print("DEBUG: Cannot get terrain information - invalid position")
        return None


def integrate_terrain_system(env):
    """Convenience function to integrate terrain into an environment."""
    print("DEBUG: Integrating terrain into the environment")

    # Create the adapter.
    adapter = TerrainAdapter(env)

    # Convert existing meshes.
    adapter.convert_existing_meshes()

    # Attach the adapter to the environment.
    env.terrain_adapter = adapter

    print("DEBUG: Terrain integration complete")
    return adapter


# Test code.
if __name__ == "__main__":
    print("=== Testing TerrainAdapter ===")

    # Mock environment for simplified testing.
    class MockEnv:
        def __init__(self):
            self.grid = np.full((10, 10), '.', dtype=str)
            self.mesh_map = np.full_like(self.grid, None, dtype=object)
            self.meshes = []

    mock_env = MockEnv()

    # Create the adapter.
    adapter = TerrainAdapter(mock_env)

    # Test terrain creation.
    ground = adapter.create_terrain('empty', (5, 5))
    mountain = adapter.create_terrain('mountain', (3, 3))

    # Test passability checks.
    print(f"Ground passable: {adapter.is_traversable((5, 5))}")
    print(f"Mountain passable: {adapter.is_traversable((3, 3))}")

    # Test item operations.
    ground.add_item('wood', 2)
    print(f"Ground information: {adapter.get_terrain_info((5, 5))}")

    print("=== Tests complete ===")
