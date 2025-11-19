"""Pure Mahjong game logic without I/O."""

import numpy as np
from utils import (
    DOTS_START,
    BAMBOO_START,
    CHAR_START,
    CHAR_END,
    WIND_START,
    WIND_END,
    DRAGON_START,
    DRAGON_END,
    FLOWER_START,
    FLOWER_END,
    ANIMAL_START,
    ANIMAL_END,
    EAST,
    SOUTH,
    WEST,
    NORTH,
    NUM_PLAYERS,
    MAX_HAND_SIZE,
    MAX_TAI,
)

# Game constants
DEBUG = False
MAX_WALL_SIZE = 148 - 40 if DEBUG else 148
DEAD_WALL_SIZE = 15


class MahjongGame:
    """Singapore Mahjong game engine with pure game logic."""

    def __init__(self):
        # Wall (shuffled on reset)
        self.wall = np.zeros(MAX_WALL_SIZE, dtype=np.uint8)
        self._create_base_tiles()
        self.wall_ptr = 0

        # Player hands: 4 players x 14 tiles
        self.hands = np.zeros((NUM_PLAYERS, MAX_HAND_SIZE), dtype=np.uint8)
        self.hand_sizes = np.zeros(NUM_PLAYERS, dtype=np.uint8)

        # Player melds: 4 players x 4 melds x 4 tiles (0 = empty)
        self.melds = np.zeros((NUM_PLAYERS, 4, 4), dtype=np.uint8)
        self.meld_types = np.zeros((NUM_PLAYERS, 4), dtype=np.uint8)

        # Player bonus tiles
        self.flowers = np.zeros((NUM_PLAYERS, 8), dtype=np.uint8)
        self.animals = np.zeros((NUM_PLAYERS, 4), dtype=np.uint8)
        self.flower_counts = np.zeros(NUM_PLAYERS, dtype=np.uint8)
        self.animal_counts = np.zeros(NUM_PLAYERS, dtype=np.uint8)

        # Discard pools
        self.discards = np.zeros((NUM_PLAYERS, 30), dtype=np.uint8)
        self.discard_counts = np.zeros(NUM_PLAYERS, dtype=np.uint8)

        # Payout table
        self.payouts = np.array([[4, 2], [8, 4], [16, 8], [32, 16], [64, 32]], dtype=np.uint8)

        # Initialize game
        self.reset()

    def _create_base_tiles(self):
        """Create base tile set"""
        ptr = 0
        # Suited tiles: 4 of each (1-9)
        for tile in range(1, CHAR_END):
            for _ in range(4):
                self.wall[ptr] = tile
                ptr += 1

        if not DEBUG:
            # Winds && Dragons: 4 of each
            for tile in range(WIND_START, DRAGON_END):
                for _ in range(4):
                    self.wall[ptr] = tile
                    ptr += 1

            # Flowers && Animals: 1 of each
            for tile in range(FLOWER_START, ANIMAL_END):
                self.wall[ptr] = tile
                ptr += 1
        assert ptr == MAX_WALL_SIZE

    def draw_tile(self) -> np.uint8:
        """Draw tile from wall"""
        if self.wall_ptr >= MAX_WALL_SIZE:
            return np.uint8(0)
        tile = self.wall[self.wall_ptr]
        self.wall_ptr += 1
        return tile

    def deal_initial_tiles(self):
        """Deal 13 tiles to each player, 14 to dealer"""
        self.hands.fill(0)
        self.hand_sizes.fill(0)

        # Deal 13 to each
        for _ in range(13):
            for p in range(NUM_PLAYERS):
                tile = self.draw_tile()
                self.add_tile_to_hand(p, tile)

        # Extra tile to dealer
        tile = self.draw_tile()
        self.add_tile_to_hand(self.dealer, tile)

        # Replace bonus tiles
        for p in range(NUM_PLAYERS):
            self.replace_bonus_tiles(p)

    def add_tile_to_hand(self, player: int, tile: np.uint8):
        """Add tile to player's hand"""
        if self.hand_sizes[player] < MAX_HAND_SIZE:
            self.hands[player, self.hand_sizes[player]] = tile
            self.hand_sizes[player] += 1

    def remove_tile_from_hand(self, player: int, tile_idx: int) -> np.uint8:
        """Remove tile at index from hand"""
        if tile_idx >= self.hand_sizes[player]:
            return np.uint8(0)

        tile = self.hands[player, tile_idx]
        for i in range(tile_idx, self.hand_sizes[player] - 1):
            self.hands[player, i] = self.hands[player, i + 1]

        self.hands[player, self.hand_sizes[player] - 1] = np.uint8(0)
        self.hand_sizes[player] -= 1
        return tile

    def remove_tiles_from_hand(self, player: int, tiles: list[np.uint8]):
        """Remove specific tiles from hand"""
        hand = list(self.hands[player, : self.hand_sizes[player]])
        for tile in tiles:
            if tile in hand:
                hand.remove(tile)

        self.hands[player].fill(0)
        for i, tile in enumerate(hand):
            self.hands[player, i] = tile
        self.hand_sizes[player] = len(hand)

    def add_meld(self, player: int, tiles: list[np.uint8], meld_type: int):
        """Add meld to player's melds"""
        for meld_idx in range(4):
            if self.melds[player, meld_idx, 0] == 0:
                for i, tile in enumerate(tiles):
                    if i < 4:
                        self.melds[player, meld_idx, i] = tile
                self.meld_types[player, meld_idx] = meld_type
                break

    def upgrade_pong_to_kong(self, player: int, tile: np.uint8) -> bool:
        """Upgrade existing pong to kong"""
        for meld_idx in range(4):
            if self.meld_types[player, meld_idx] == 2:
                if self.melds[player, meld_idx, 0] == tile:
                    self.melds[player, meld_idx, 3] = tile
                    self.meld_types[player, meld_idx] = 3
                    return True
        return False

    def can_hidden_kong(self, player: int) -> list[np.uint8]:
        """Check if player can make hidden kong"""
        hand = self.hands[player, : self.hand_sizes[player]]
        counts = np.bincount(hand, minlength=47)
        return [np.uint8(tile) for tile in range(1, 47) if counts[tile] == 4]

    def can_upgrade_kong(self, player: int) -> list[np.uint8]:
        """Check if player can upgrade pong to kong"""
        upgradeable = []
        for meld_idx in range(4):
            if self.meld_types[player, meld_idx] == 2:
                tile = self.melds[player, meld_idx, 0]
                hand = self.hands[player, : self.hand_sizes[player]]
                if np.sum(hand == tile) > 0:
                    upgradeable.append(tile)
        return upgradeable

    def replace_bonus_tiles(self, player: int):
        """Replace flowers/animals with tiles from wall"""
        i = 0
        while i < self.hand_sizes[player]:
            tile = self.hands[player, i]

            if FLOWER_START <= tile < FLOWER_END:
                if self.flower_counts[player] < 8:
                    self.flowers[player, self.flower_counts[player]] = tile
                    self.flower_counts[player] += 1

                replacement = self.draw_tile()
                self.hands[player, i] = replacement

                if not (FLOWER_START <= replacement < ANIMAL_END):
                    i += 1

            elif ANIMAL_START <= tile < ANIMAL_END:
                if self.animal_counts[player] < 4:
                    self.animals[player, self.animal_counts[player]] = tile
                    self.animal_counts[player] += 1

                replacement = self.draw_tile()
                self.hands[player, i] = replacement

                if not (FLOWER_START <= replacement < ANIMAL_END):
                    i += 1
            else:
                i += 1

    def calculate_tai(self, player: int) -> tuple[int, bool]:
        """Calculate tai (bonus points) from flowers and animals"""
        tai = 0

        # Player's flower bonus
        player_flowers = [player + 35, player + 39]
        for flower in self.flowers[player, : self.flower_counts[player]]:
            if flower in player_flowers:
                tai += 1

        # Complete flower set bonus
        red_flowers = sum(1 for f in self.flowers[player, : self.flower_counts[player]] if 35 <= f <= 38)
        blue_flowers = sum(1 for f in self.flowers[player, : self.flower_counts[player]] if 39 <= f <= 42)

        if red_flowers == 4:
            tai += 1
        if blue_flowers == 4:
            tai += 1

        # Animal bonus
        if self.animal_counts[player] == 4:
            tai += 5

        is_flowerless = self.flower_counts[player] == 0 and self.animal_counts[player] == 0

        return tai, is_flowerless

    def check_flush(self, hand: np.ndarray) -> int:
        """Check flush type: 0=none, 2=half, 4=full"""
        suit = -1
        has_honors = False

        for tile in hand:
            if tile == 0:
                continue
            if tile < 28:
                tile_suit = (tile - 1) // 9
                if suit == -1:
                    suit = tile_suit
                elif suit != tile_suit:
                    return 0
            else:
                has_honors = True

        return 2 if has_honors else 4

    def check_nine_gates(self, counts: np.ndarray) -> bool:
        """Check for Nine Gates pattern"""
        if np.sum(counts) != 14:
            return False
        for head in (DOTS_START, BAMBOO_START, CHAR_START):
            for tile in range(head, head + 9):
                if tile < head or tile > head + 8:
                    break
                elif tile in (head, head + 8):
                    if counts[tile] < 3:
                        break
                else:
                    if counts[tile] < 1:
                        break
            else:
                return True
        return False

    def find_best_winning_pattern(self, hand: np.ndarray, player: int) -> tuple[str, int]:
        """Find best possible winning pattern and its points"""
        counts = np.bincount(hand, minlength=47)

        # Check special hands (only for concealed hands)
        has_melds = any(self.melds[player, i, 0] > 0 for i in range(4))

        if not has_melds:
            if self.is_thirteen_wonders(counts):
                return ("Thirteen Wonders", 13)
            elif self.check_nine_gates(counts):
                return ("Nine Gates", 13)

        # Check flush for hand
        hand_flush = self.check_flush(hand)

        # Check flush for melds
        meld_flush = 4  # Start with full flush assumption
        for meld_idx in range(4):
            if self.melds[player, meld_idx, 0] > 0:
                meld_tiles = self.melds[player, meld_idx]
                meld_tiles = meld_tiles[meld_tiles > 0]
                meld_flush_check = self.check_flush(meld_tiles)

                # Take more restrictive flush
                if meld_flush_check < meld_flush:
                    meld_flush = meld_flush_check

        # Combined flush is the minimum (most restrictive)
        if not has_melds:
            combined_flush = hand_flush
        else:
            combined_flush = min(hand_flush, meld_flush) if hand_flush > 0 and meld_flush > 0 else 0

        flushness = {0: "", 2: "Half Flush", 4: "Full Flush"}[combined_flush]

        # Check structure hierarchy for hand (from highest to lowest)
        hand_structure = 0
        _, is_flowerless = self.calculate_tai(player)

        if self.is_sequence_hand_valid(hand, player) and is_flowerless:
            hand_structure = 4  # Pure sequence
        elif self.is_seven_pairs(counts) and not has_melds:
            hand_structure = 4  # Seven pairs
        elif self.all_triplets(counts):
            hand_structure = 2  # Pure triplets
        elif self.is_lesser_sequence(hand):
            hand_structure = 1  # Lesser sequence (any valid sets)

        # Check structure for melds (what constraints do melds impose?)
        meld_has_sequence = False
        meld_has_triplet = False

        for meld_idx in range(4):
            if self.melds[player, meld_idx, 0] > 0:
                meld_type = self.meld_types[player, meld_idx]

                # 1=chow, 2=pong, 3=kong, 4=hidden_kong
                if meld_type == 1:
                    meld_has_sequence = True
                elif meld_type in [2, 3, 4]:
                    meld_has_triplet = True

        # Determine what structures are compatible
        if not has_melds:
            combined_structure = hand_structure
        else:
            # Sequence hand: needs ALL sequences (hand + melds)
            if hand_structure >= 1 and not meld_has_triplet and meld_has_sequence:
                # Check if hand forms sequences too
                if self.is_sequence_hand_valid(hand, player) and is_flowerless:
                    combined_structure = 4
                elif self.is_lesser_sequence(hand):
                    combined_structure = 1
                else:
                    combined_structure = 0

            # Triplets hand: needs ALL triplets (hand + melds)
            elif hand_structure >= 1 and not meld_has_sequence and meld_has_triplet:
                if self.all_triplets(counts):
                    combined_structure = 2
                elif self.is_lesser_sequence(hand):
                    combined_structure = 1
                else:
                    combined_structure = 0

            # Mixed melds (both sequences and triplets) or just valid sets
            elif hand_structure >= 1:
                # Can only be lesser sequence (allows mixing)
                combined_structure = 1
            else:
                combined_structure = 0

        # Map structure value to name
        structure_names = {
            4: "Sequence Hand",
            2: "Triplets Hand",
            1: "Lesser Sequence Hand",
            0: "",
        }
        structure_name = structure_names.get(combined_structure, "")

        if not combined_structure and not combined_flush:
            return ("Chicken Hand", 0)
        else:
            final_name = (flushness + " " + structure_name).strip()
            if not final_name:
                final_name = flushness if flushness else "Chicken Hand"

            return (
                final_name,
                min(combined_flush + combined_structure, MAX_TAI),
            )

    def is_sequence_hand_valid(self, hand: np.ndarray, player: int) -> bool:
        """Check sequence hand with proper restrictions"""
        counts = np.bincount(hand, minlength=47)

        for tile in range(1, 47):
            if counts[tile] >= 2:
                if tile >= 28 and tile <= 31:
                    if tile == self.prevailing_wind:
                        continue
                    seat_wind = EAST + player
                    if tile == seat_wind:
                        continue
                elif tile >= 32 and tile <= 34:
                    continue

                test_counts = counts.copy()
                test_counts[tile] -= 2

                if self.can_form_sequences_only(test_counts):
                    return True

        return False

    def is_lesser_sequence(self, hand: np.ndarray) -> bool:
        """Check lesser sequence"""
        counts = np.bincount(hand, minlength=47)
        return self.can_form_sets(counts)

    def can_form_sequences_only(self, counts: np.ndarray) -> bool:
        """Check if can form only sequences"""
        total = np.sum(counts)
        if total % 3 != 0:
            return False

        for tile in range(1, 28):
            while counts[tile] > 0:
                suit = (tile - 1) // 9
                tile_in_suit = (tile - 1) % 9

                if tile_in_suit > 6:
                    return False

                next1, next2 = tile + 1, tile + 2
                if next2 >= (suit + 1) * 9 + 1 or counts[next1] == 0 or counts[next2] == 0:
                    return False

                counts[tile] -= 1
                counts[next1] -= 1
                counts[next2] -= 1

        return np.sum(counts[28:35]) == 0

    def calculate_points(self, player: int) -> tuple[str, int]:
        """Calculate best winning hand and points"""
        hand = self.hands[player, : self.hand_sizes[player]]

        pattern_name, base_points = self.find_best_winning_pattern(hand, player)

        tai, _ = self.calculate_tai(player)

        # Add wind/dragon bonuses from melds
        for meld_idx in range(4):
            if self.melds[player, meld_idx, 0] > 0:
                meld_tile = self.melds[player, meld_idx, 0]
                if meld_tile >= 32 and meld_tile <= 34:
                    tai += 1
                elif meld_tile >= 28 and meld_tile <= 31:
                    if meld_tile == self.prevailing_wind:
                        tai += 1
                    seat_wind = EAST + player
                    if meld_tile == seat_wind:
                        tai += 1

        total_points = base_points + tai
        total_points = min(total_points, 5)

        return pattern_name, total_points

    def can_win(self, player: int) -> bool:
        """Check if player can win"""
        hand = self.hands[player, : self.hand_sizes[player]]
        return self.is_winning_hand(hand)

    def is_winning_hand(self, hand: np.ndarray) -> bool:
        """Fast winning hand check"""
        if len(hand) % 3 != 2:
            return False

        counts = np.bincount(hand, minlength=47)

        if self.is_thirteen_wonders(counts):
            return True
        if self.is_seven_pairs(counts):
            return True

        return self.can_form_sets(counts)

    def is_thirteen_wonders(self, counts: np.ndarray) -> bool:
        """Check thirteen wonders"""
        required = [1, 9, 10, 18, 19, 27, 28, 29, 30, 31, 32, 33, 34]
        non_zero = np.nonzero(counts)[0]
        return np.sum(counts) == 14 and np.array_equal(sorted(non_zero), required)

    def is_seven_pairs(self, counts: np.ndarray) -> bool:
        """Check seven pairs"""
        return np.sum(counts == 2) == 7 and np.sum(counts) == 14

    def can_form_sets(self, counts: np.ndarray) -> bool:
        """Check if tiles can form 4 sets + 1 pair"""
        counts = counts.copy()

        for tile in range(1, 47):
            if counts[tile] >= 2:
                counts[tile] -= 2
                if self.form_sets_recursive(counts):
                    return True
                counts[tile] += 2

        return False

    def form_sets_recursive(self, counts: np.ndarray) -> bool:
        """Recursively form sets"""
        total = np.sum(counts)
        if total == 0:
            return True
        if total % 3 != 0:
            return False

        tile = np.argmax(counts > 0)

        # Try triplet
        if counts[tile] >= 3:
            counts[tile] -= 3
            if self.form_sets_recursive(counts):
                counts[tile] += 3
                return True
            counts[tile] += 3

        # Try sequence
        if tile < 28:
            suit = (tile - 1) // 9
            tile_in_suit = (tile - 1) % 9

            if tile_in_suit <= 6:
                next1, next2 = tile + 1, tile + 2
                suit_end = suit * 9 + 10
                if next2 < suit_end and counts[next1] > 0 and counts[next2] > 0:
                    counts[tile] -= 1
                    counts[next1] -= 1
                    counts[next2] -= 1
                    if self.form_sets_recursive(counts):
                        counts[tile] += 1
                        counts[next1] += 1
                        counts[next2] += 1
                        return True
                    counts[tile] += 1
                    counts[next1] += 1
                    counts[next2] += 1

        return False

    def all_triplets(self, counts: np.ndarray) -> bool:
        """Check if all triplets"""
        pairs = np.sum(counts == 2)
        triplets = np.sum(counts == 3)
        return pairs == 1 and triplets == 4

    def can_pong_kong(self, player: int, tile: np.uint8) -> tuple[bool, bool]:
        """Check if can pong or kong"""
        hand = self.hands[player, : self.hand_sizes[player]]
        tile_count = np.sum(hand == tile)
        return tile_count >= 2, tile_count >= 3

    def can_chow(self, player: int, tile: np.uint8) -> list[list[np.uint8]]:
        """Check possible chows (only from previous player)"""
        if (player - 1) % NUM_PLAYERS != self.last_discard_player:
            return []

        if tile >= 28:
            return []

        hand = self.hands[player, : self.hand_sizes[player]]
        possible_chows = []

        suit = (tile - 1) // 9
        tile_in_suit = (tile - 1) % 9
        suit_start = suit * 9 + 1

        for pos in range(3):
            if pos > tile_in_suit:
                continue
            seq_start = tile_in_suit - pos
            if 0 <= seq_start <= 6:
                sequence = [suit_start + seq_start + i for i in range(3)]
                needed = [t for t in sequence if t != tile]
                if all(np.sum(hand == need_tile) > 0 for need_tile in needed):
                    possible_chows.append(sequence)

        return possible_chows

    def advance_dealer(self, dealer_won: bool):
        """Advance dealer and wind based on game rules"""
        if not dealer_won:
            self.dealer = (self.dealer + 1) % NUM_PLAYERS

            if self.dealer == 0:
                wind_order = [EAST, SOUTH, WEST, NORTH]
                current_wind_idx = wind_order.index(self.prevailing_wind)
                if current_wind_idx < 3:
                    self.prevailing_wind = wind_order[current_wind_idx + 1]
                else:
                    self.game_complete = True

    def get_tiles_remaining(self) -> int:
        """Get number of tiles remaining in wall"""
        return MAX_WALL_SIZE - self.wall_ptr

    def reset(self):
        """Reset game for new round"""
        self.hands.fill(0)
        self.hand_sizes.fill(0)
        self.melds.fill(0)
        self.meld_types.fill(0)
        self.flowers.fill(0)
        self.animals.fill(0)
        self.flower_counts.fill(0)
        self.animal_counts.fill(0)
        self.discards.fill(0)
        self.discard_counts.fill(0)

        self.current_player = 0
        self.dealer = 0
        self.prevailing_wind = EAST
        self.game_over = False
        self.winner = -1
        self.points_scored = 0
        self.last_discard = np.uint8(0)
        self.last_discard_player = -1
        self.game_complete = False
        self.skip_draw = False

        np.random.shuffle(self.wall)
        self.wall_ptr = 0

        self.deal_initial_tiles()
