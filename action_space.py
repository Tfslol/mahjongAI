"""Action space definitions for MuZero-compatible Mahjong environment."""

import numpy as np
from typing import NamedTuple

# Action space size
NUM_TILE_TYPES = 34
NUM_CLAIM_ACTIONS = 7
TOTAL_ACTIONS = NUM_TILE_TYPES + NUM_CLAIM_ACTIONS  # 41 total

# Discard actions: 0-33 (one per tile type)
DISCARD_START = 0
DISCARD_END = 34

# Claim actions: 34-40
ACTION_SKIP = 34
ACTION_WIN = 35
ACTION_PONG = 36
ACTION_KONG = 37
ACTION_CHOW_LEFT = 38
ACTION_CHOW_MID = 39
ACTION_CHOW_RIGHT = 40


class ActionType(NamedTuple):
    """Represents a decoded action."""

    type: str  # 'discard', 'win', 'pong', 'kong', 'chow', 'skip'
    tile_type: int | None = None  # For discard actions (0-33)
    chow_pos: int | None = None  # For chow: 0=left, 1=mid, 2=right


def decode_action(action: int) -> ActionType:
    """Decode action integer into structured format.

    Args:
        action: Integer in range [0, 40]

    Returns:
        ActionType with decoded information
    """
    if 0 <= action < 34:
        return ActionType(type="discard", tile_type=action)
    elif action == ACTION_SKIP:
        return ActionType(type="skip")
    elif action == ACTION_WIN:
        return ActionType(type="win")
    elif action == ACTION_PONG:
        return ActionType(type="pong")
    elif action == ACTION_KONG:
        return ActionType(type="kong")
    elif action == ACTION_CHOW_LEFT:
        return ActionType(type="chow", chow_pos=0)
    elif action == ACTION_CHOW_MID:
        return ActionType(type="chow", chow_pos=1)
    elif action == ACTION_CHOW_RIGHT:
        return ActionType(type="chow", chow_pos=2)
    else:
        raise ValueError(f"Invalid action: {action}")


def tile_to_type(tile: int) -> int:
    """Convert tile value (1-34) to tile type (0-33)."""
    return tile - 1


def type_to_tile(tile_type: int) -> int:
    """Convert tile type (0-33) to tile value (1-34)."""
    return tile_type + 1
