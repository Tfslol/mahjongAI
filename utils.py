"""Utility functions for Mahjong game display and conversion."""

import numpy as np

# Tile constants
DOTS_START, DOTS_END = 1, 10
BAMBOO_START, BAMBOO_END = 10, 19
CHAR_START, CHAR_END = 19, 28
WIND_START, WIND_END = 28, 32
DRAGON_START, DRAGON_END = 32, 35
FLOWER_START, FLOWER_END = 35, 43
ANIMAL_START, ANIMAL_END = 43, 47

# Wind/Dragon mappings
EAST, SOUTH, WEST, NORTH = 28, 29, 30, 31
RED_DRAGON, GREEN_DRAGON, WHITE_DRAGON = 32, 33, 34

# Game constants
NUM_PLAYERS = 4
MAX_HAND_SIZE = 14
MAX_TAI = 5

BOOL_OPTIONS = ("y", "n")


def tile_to_str(tile: np.uint8) -> str:
    """Convert tile to English notation"""
    if tile == 0:
        return "  "
    elif 1 <= tile <= 9:
        return f"{tile}t"
    elif 10 <= tile <= 18:
        return f"{tile-9}s"
    elif 19 <= tile <= 27:
        return f"{tile-18}w"
    elif 28 <= tile <= 34:
        return f"{tile-27}z"
    return "??"


def meld_to_str(meld: np.ndarray, meld_type: int) -> str:
    """Convert meld to compact string"""
    tiles = [t for t in meld if t > 0]
    if not tiles:
        return ""

    # Add indicator for hidden kong
    prefix = "H:" if meld_type == 4 else ""

    first_tile = tiles[0]
    if first_tile <= 9:
        suit = "t"
        values = [t for t in tiles]
    elif first_tile <= 18:
        suit = "s"
        values = [t - 9 for t in tiles]
    elif first_tile <= 27:
        suit = "w"
        values = [t - 18 for t in tiles]
    else:
        suit = "z"
        values = [t - 27 for t in tiles]

    return prefix + "".join(str(v) for v in values) + suit


def wind_to_str(wind: int) -> str:
    """Convert wind value to string"""
    wind_names = {EAST: "East", SOUTH: "South", WEST: "West", NORTH: "North"}
    return wind_names.get(wind, "?")


def get_input(message: str, options: list[str] | tuple[str, ...]) -> str:
    """Get validated input from user"""
    response = (
        input(f"{message} {{ {' '.join(str(a) for a in options)} }}: ").strip().lower()
    )

    while response not in options:
        response = (
            input(
                f"Please input from the following options {{ {' '.join(str(a) for a in options)} }}: "
            )
            .strip()
            .lower()
        )
    return response


def get_sorted_hand_with_mapping(hand: np.ndarray) -> tuple[list[int], list[int]]:
    """Get sorted hand tiles with mapping back to original indices

    Returns:
        sorted_tiles: List of tiles in sorted order
        index_mapping: Mapping from sorted index to original index
    """
    indexed_tiles = [(hand[i], i) for i in range(len(hand)) if hand[i] > 0]
    indexed_tiles.sort(key=lambda x: x[0])

    sorted_tiles = [tile for tile, _ in indexed_tiles]
    index_mapping = [orig_idx for _, orig_idx in indexed_tiles]

    return sorted_tiles, index_mapping
