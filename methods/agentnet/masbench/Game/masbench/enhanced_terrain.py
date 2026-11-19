"""
Enhanced terrain management: unified terrain and item management.
Extends CapacityMesh while preserving backward compatibility.
"""

import numpy as np
from masbench.physics import CapacityMesh

class Item:
    """An item that can be placed in a grid cell."""

    def __init__(self, item_type, quantity=1):
        print(f"DEBUG: Creating Item: type={item_type}, quantity={quantity}")
        self.item_type = item_type  # 'wood', 'stone', 'sand', 'food'
        self.quantity = quantity
        self.score_value = self._get_score_value()
        print(f"DEBUG: Item created; score value: {self.score_value}")

    def _get_score_value(self):
        """Return the score value for the item type."""
        values = {'wood': 1, 'stone': 2, 'sand': 1, 'food': 1}
        return values.get(self.item_type, 0)

    def __repr__(self):
        return f"Item({self.item_type}:{self.quantity})"


class EnhancedCapacityMesh(CapacityMesh):
    """
    Enhanced CapacityMesh with unified terrain and item management.
    Inherits CapacityMesh for physics engine compatibility.
    """

    # Terrain configuration: terrain layer plus item layer.
    TERRAIN_CONFIG = {
        # Terrain type: (symbol, passable, stores_items, default_capacity, static).
        'empty': ('.', True, True, 10, False),      # Empty: passable, stores items.
        'mountain': ('M', False, False, 0, True),   # Mountain: impassable, no storage.
        'river': ('R', False, False, 0, True),       # River: impassable, no storage.
        'wall': ('W', False, False, 0, True),       # Wall: impassable, no storage.
        'resource': ('r', True, True, 5, True),     # Resource: passable, stores items, initially populated.
        'table': ('T', True, True, 10, True),       # Table: passable, stores items.
        'forest': ('F', True, True, 3, True),       # Forest: passable, stores items.
        'grass': ('G', True, True, 2, True),        # Grass: passable, stores items.
    }

    def __init__(self, terrain_type, position, shape=None, capacity=None, name=None,
                 type_caps: dict | None = None,
                 per_type_default: int | None = None):
        """
        Initialize enhanced terrain.

        Args:
            terrain_type: Terrain type, such as 'empty' or 'mountain'.
            position: Position coordinates (y, x).
            shape: Shape array, defaulting to 1x1.
            capacity: Capacity, defaulting to the configured value.
            name: Name, defaulting to the terrain display symbol.
        """
        print(f"DEBUG: Initializing EnhancedCapacityMesh: terrain={terrain_type}, position={position}")

        # Get terrain configuration.
        if terrain_type not in self.TERRAIN_CONFIG:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

        config = self.TERRAIN_CONFIG[terrain_type]
        display_symbol, traversable, can_store_items, default_capacity, is_static = config

        # Set defaults.
        if shape is None:
            shape = np.array([[display_symbol]], dtype=str)
        if name is None:
            name = display_symbol
        if capacity is None:
            capacity = default_capacity

        print(f"DEBUG: Terrain configuration - symbol={display_symbol}, passable={traversable}, stores_items={can_store_items}, capacity={capacity}")

        # Call the parent constructor.
        super().__init__(position, shape, is_static, name=name, capacity=capacity)

        # Extended attributes.
        self.terrain_type = terrain_type
        self.traversable = traversable
        self.can_store_items = can_store_items
        self.items = []  # Item list
        self.display_symbol = display_symbol
        # Table extensions: optional per-type capacities and default per-type capacity.
        self.capacity_total = int(getattr(self, 'capacity', 0))
        try:
            self.capacity_by_type = {str(k): int(v) for k, v in (type_caps or {}).items()} if type_caps else None
        except Exception:
            self.capacity_by_type = None
        try:
            self.per_type_default = int(per_type_default) if per_type_default is not None else None
        except Exception:
            self.per_type_default = None

        print(f"DEBUG: EnhancedCapacityMesh initialized; final name: {self.name}")

    def add_item(self, item_type, quantity=1):
        """Add items to the cell."""
        print(f"DEBUG: Adding items: type={item_type}, quantity={quantity}")

        if not self.can_store_items:
            print(f"DEBUG: Failed - terrain {self.terrain_type} cannot store items")
            return False

        # Check per-type capacity first, if configured, then total capacity.
        total_quantity = sum(item.quantity for item in self.items)
        # Optional remaining capacity by type.
        if self.capacity_by_type is not None or self.per_type_default is not None:
            cap_type = None
            if self.capacity_by_type and item_type in self.capacity_by_type:
                cap_type = int(self.capacity_by_type[item_type])
            elif self.per_type_default is not None:
                cap_type = int(self.per_type_default)
            if cap_type is not None:
                cur_type_q = sum(it.quantity for it in self.items if it.item_type == item_type)
                if cur_type_q + quantity > cap_type:
                    print(f"DEBUG: Failed - insufficient capacity for {item_type}: {cur_type_q}+{quantity} > {cap_type}")
                    return False
        # Total remaining capacity.
        if total_quantity + quantity > self.capacity:
            print(f"DEBUG: Failed - insufficient capacity: current={total_quantity}, requested={quantity}, total_capacity={self.capacity}")
            return False

        # Find items of the same type.
        for item in self.items:
            if item.item_type == item_type:
                item.quantity += quantity
                print(f"DEBUG: Success - increased existing item quantity to {item.quantity}")
                return True

        # Add a new item.
        new_item = Item(item_type, quantity)
        self.items.append(new_item)
        print(f"DEBUG: Success - added new item: {new_item}")
        return True

    # ===== Table capacity helpers: remaining capacity =====
    def get_remaining_total(self) -> int:
        try:
            return max(0, int(getattr(self, 'capacity', 0)) - int(self.get_items_total_quantity()))
        except Exception:
            return 0

    def get_remaining_for_type(self, item_type: str):
        """Return remaining capacity for a type, or None if no type limit is configured (total capacity still applies)."""
        try:
            cap_type = None
            if self.capacity_by_type and item_type in self.capacity_by_type:
                cap_type = int(self.capacity_by_type[item_type])
            elif self.per_type_default is not None:
                cap_type = int(self.per_type_default)
            if cap_type is None:
                return None
            cur_type_q = self.get_item_quantity(item_type)
            return max(0, cap_type - int(cur_type_q))
        except Exception:
            return None

    def get_remaining_by_type(self, types: list[str]) -> dict:
        out = {}
        for t in types:
            rem = self.get_remaining_for_type(t)
            if rem is not None:
                # Total remaining capacity still applies.
                out[t] = max(0, min(int(rem), int(self.get_remaining_total())))
        return out

    def remove_item(self, item_type, quantity=1):
        """Remove items from the cell."""
        print(f"DEBUG: Removing items: type={item_type}, quantity={quantity}")

        for i, item in enumerate(self.items):
            if item.item_type == item_type:
                if item.quantity >= quantity:
                    item.quantity -= quantity
                    print(f"DEBUG: Success - removed items; remaining quantity: {item.quantity}")

                    if item.quantity == 0:
                        removed_item = self.items.pop(i)
                        print(f"DEBUG: Item quantity reached zero; removed: {removed_item}")

                    return True
                else:
                    print(f"DEBUG: Failed - insufficient items: available={item.quantity}, requested={quantity}")
                    return False

        print(f"DEBUG: Failed - item type not found: {item_type}")
        return False

    def get_items_total_quantity(self):
        """Return the total item quantity."""
        total = sum(item.quantity for item in self.items)
        print(f"DEBUG: Total item quantity: {total}")
        return total

    def get_item_quantity(self, item_type):
        """Return the quantity of a specific item type."""
        for item in self.items:
            if item.item_type == item_type:
                print(f"DEBUG: Quantity of {item_type}: {item.quantity}")
                return item.quantity
        print(f"DEBUG: Item {item_type} is absent; quantity: 0")
        return 0

    def get_display_name(self):
        """Return the full display name, including terrain and item information."""

        # Tables always display the T symbol.
        if self.terrain_type == 'table':
            result = 'T'
            # print(f"DEBUG: Display name (table special case): {result}")
            return result

        # Prefer item information: display the initial character of the item type when items exist.
        if self.can_store_items and self.items:
            # Use the first item's type initial, lowercase where possible to avoid wall W conflicts.
            first_item_type = self.items[0].item_type
            # Resource symbols: wood->w, sand->s, stone->S; avoid conflicts with wall W.
            display_mapping = {
                'wood': 'w',
                'sand': 's',
                'stone': 'S'  # Stone uses uppercase S to distinguish it from sand.
            }
            display_char = display_mapping.get(first_item_type, first_item_type[0].lower() if first_item_type else self.display_symbol)
            result = display_char
            # print(f"DEBUG: Display name (item priority): {result} (item: {first_item_type})")
            return result

        # Display when no items remain:
        # - Empty resource placeholders display '.' for consistency with empty ground.
        if self.terrain_type == 'resource':
            result = '.'
            return result
        # - Other terrain uses its base symbol.
        result = self.display_symbol
        # print(f"DEBUG: Display name (base terrain): {result}")
        return result

    def is_depleted(self):
        """Override depletion checks to account for item storage."""
        if self.terrain_type.endswith('_source'):
            # Check depletion of resource sources.
            depleted = super().is_depleted()
            print(f"DEBUG: Resource source depletion check: {depleted}")
            return depleted
        else:
            # Check depletion of storage terrain.
            total_items = self.get_items_total_quantity()
            depleted = total_items >= self.capacity
            print(f"DEBUG: Storage depletion check: items={total_items}, capacity={self.capacity}, depleted={depleted}")
            return depleted

    def get_terrain_info(self):
        """Return detailed terrain information."""
        info = {
            'terrain_type': self.terrain_type,
            'position': self.pos,
            'traversable': self.traversable,
            'can_store_items': self.can_store_items,
            'capacity': self.capacity,
            'current_items': len(self.items),
            'total_quantity': self.get_items_total_quantity(),
            'items_detail': [{'type': item.item_type, 'quantity': item.quantity} for item in self.items],
            'display_name': self.get_display_name()
        }
        print(f"DEBUG: Terrain information: {info}")
        return info

    def __repr__(self):
        """Override the object representation."""
        items_info = f" items={self.items}" if self.items else ""
        return f"EnhancedCapacityMesh('{self.terrain_type}'@{self.pos}, capacity={self.capacity}{items_info})"


# Terrain factory.
def create_terrain(terrain_type, position, **kwargs):
    """Create terrain using the factory."""
    print(f"DEBUG: Creating terrain: type={terrain_type}, position={position}, parameters={kwargs}")

    # Handle special shapes.
    shape = kwargs.get('shape')
    if shape is not None and not isinstance(shape, np.ndarray):
        shape = np.array(shape, dtype=str)
        kwargs['shape'] = shape

    terrain = EnhancedCapacityMesh(terrain_type, position, **kwargs)
    print(f"DEBUG: Terrain created: {terrain}")
    return terrain


# Test code.
if __name__ == "__main__":
    print("=== Testing EnhancedCapacityMesh ===")

    # Test empty ground.
    ground = create_terrain('empty', (5, 5))
    print(f"Created empty ground: {ground}")

    # Test adding items.
    ground.add_item('wood', 2)
    ground.add_item('stone', 1)
    print(f"After adding items: {ground}")

    # Test the display name.
    print(f"Display name: {ground.get_display_name()}")

    # Test impassable mountains.
    mountain = create_terrain('mountain', (3, 3))
    print(f"Created mountain: {mountain}")
    print(f"Mountain passable: {mountain.traversable}")

    # Test the nest.
    nest = create_terrain('nest', (7, 7), capacity=3)
    print(f"Created nest: {nest}")
    nest.add_item('food', 2)
    print(f"Nest after adding food: {nest.get_display_name()}")

    print("=== Tests complete ===")
