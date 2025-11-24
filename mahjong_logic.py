"""Pure Mahjong game logic without I/O - optimized for RL environment."""

import numpy as np

# Tile constants
DOTS_START, DOTS_END = 1, 10
BAMBOO_START, BAMBOO_END = 10, 19
CHAR_START, CHAR_END = 19, 28
WIND_START, WIND_END = 28, 32
DRAGON_START, DRAGON_END = 32, 35
FLOWER_START, FLOWER_END = 35, 43
ANIMAL_START = 43

EAST, SOUTH, WEST, NORTH = 28, 29, 30, 31

NUM_PLAYERS = 4
MAX_HAND_SIZE = 14
MAX_TAI = 5
MIN_TAI = 1

# Game constants
DEBUG = False
MAX_WALL_SIZE = 148 - 40 if DEBUG else 148


class MahjongGame:
    """Singapore Mahjong game engine with pure game logic."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Wall (shuffled on reset)
        self.wall = np.zeros(MAX_WALL_SIZE, dtype=np.uint8)
        self._create_base_tiles()

        # Player hands as counts (optimized)
        self.hand_counts = np.zeros((NUM_PLAYERS, 34), dtype=np.uint8)

        # Player melds (simplified storage)
        self.meld_tiles = np.zeros((NUM_PLAYERS, 4), dtype=np.uint8)  # tile type for each meld
        self.meld_counts = np.zeros((NUM_PLAYERS, 4), dtype=np.uint8)  # count (3 or 4)
        self.meld_types = np.zeros((NUM_PLAYERS, 4), dtype=np.uint8)  # 1=chow, 2=pong, 3=kong, 4=hidden_kong

        # Player bonus tiles
        self.flowers = np.zeros((NUM_PLAYERS, 8), dtype=np.uint8)  # Individual flowers
        self.animals = np.zeros((NUM_PLAYERS, 4), dtype=np.uint8)  # Individual animals
        self.flower_counts = np.zeros(NUM_PLAYERS, dtype=np.uint8)
        self.animal_counts = np.zeros(NUM_PLAYERS, dtype=np.uint8)

        # Discard pools (sequential)
        self.discards = np.zeros((NUM_PLAYERS, 40), dtype=np.uint8)
        self.discard_counts = np.zeros(NUM_PLAYERS, dtype=np.uint8)

        # Call reset to initialize state
        self.reset()

    def _create_base_tiles(self):
        """Create base tile set"""
        ptr = 0
        # Suited tiles: 4 of each (1-9 for each suit)
        for tile in range(1, CHAR_END):
            for _ in range(4):
                self.wall[ptr] = tile
                ptr += 1

        if not DEBUG:
            # Winds & Dragons: 4 of each
            for tile in range(WIND_START, DRAGON_END):
                for _ in range(4):
                    self.wall[ptr] = tile
                    ptr += 1

            # Flowers: 1 of each (8 unique flowers)
            for tile in range(FLOWER_START, FLOWER_END):
                self.wall[ptr] = tile
                ptr += 1

            # Animals: 4 of the same tile (simplified)
            for _ in range(4):
                self.wall[ptr] = ANIMAL_START
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
        self.hand_counts.fill(0)

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
        if tile > 0 and tile <= 34:
            self.hand_counts[player, tile - 1] += 1

    def remove_tile_from_hand(self, player: int, tile_type: int) -> bool:
        """Remove one tile of given type from hand"""
        if self.hand_counts[player, tile_type] > 0:
            self.hand_counts[player, tile_type] -= 1
            return True
        return False

    def remove_tiles_from_hand(self, player: int, tile_type: int, count: int):
        """Remove multiple tiles of same type from hand"""
        self.hand_counts[player, tile_type] = max(0, self.hand_counts[player, tile_type] - count)

    def add_meld(self, player: int, tile_type: int, count: int, meld_type: int):
        """Add meld to player's melds"""
        for meld_idx in range(4):
            if self.meld_counts[player, meld_idx] == 0:
                self.meld_tiles[player, meld_idx] = tile_type
                self.meld_counts[player, meld_idx] = count
                self.meld_types[player, meld_idx] = meld_type
                break

    def upgrade_pong_to_kong(self, player: int, tile_type: int) -> bool:
        """Upgrade existing pong to kong"""
        for meld_idx in range(4):
            if self.meld_types[player, meld_idx] == 2:
                if self.meld_tiles[player, meld_idx] == tile_type:
                    self.meld_counts[player, meld_idx] = 4
                    self.meld_types[player, meld_idx] = 3
                    return True
        return False

    def can_hidden_kong(self, player: int) -> list[int]:
        """Check if player can make hidden kong"""
        return [tile_type for tile_type in range(34) if self.hand_counts[player, tile_type] == 4]

    def can_upgrade_kong(self, player: int) -> list[int]:
        """Check if player can upgrade pong to kong"""
        upgradeable = []
        for meld_idx in range(4):
            if self.meld_types[player, meld_idx] == 2:
                tile_type = self.meld_tiles[player, meld_idx]
                if self.hand_counts[player, tile_type] > 0:
                    upgradeable.append(tile_type)
        return upgradeable

    def replace_bonus_tiles(self, player: int):
        """Replace flowers/animals with tiles from wall"""
        # Check all tiles in hand
        for tile_type in range(34):
            while self.hand_counts[player, tile_type] > 0:
                tile = tile_type + 1

                if FLOWER_START <= tile < FLOWER_END:
                    # Flower
                    self.hand_counts[player, tile_type] -= 1
                    if self.flower_counts[player] < 8:
                        self.flowers[player, self.flower_counts[player]] = tile
                        self.flower_counts[player] += 1

                    replacement = self.draw_tile()
                    if replacement > 0:
                        self.add_tile_to_hand(player, replacement)
                    break  # Process one at a time

                elif tile >= ANIMAL_START:
                    # Animal
                    self.hand_counts[player, tile_type] -= 1
                    if self.animal_counts[player] < 4:
                        self.animals[player, self.animal_counts[player]] = tile
                        self.animal_counts[player] += 1

                    replacement = self.draw_tile()
                    if replacement > 0:
                        self.add_tile_to_hand(player, replacement)
                    break
                else:
                    break

    def calculate_tai(self, player: int) -> tuple[int, bool]:
        """Calculate tai (bonus points) from flowers and animals

        Flower scoring:
        - Each player's own flower (matching seat): +1 tai
        - Complete red flower set (35-38): +1 tai
        - Complete blue flower set (39-42): +1 tai

        Animal scoring:
        - Complete set of 4 animals: +5 tai
        """
        tai = 0

        # Player's own flowers (matching seat)
        # Player 0 (East): 35, 39
        # Player 1 (South): 36, 40
        # Player 2 (West): 37, 41
        # Player 3 (North): 38, 42
        player_flowers = [player + 35, player + 39]
        for flower in self.flowers[player, : self.flower_counts[player]]:
            if flower in player_flowers:
                tai += 1

        # Complete flower sets
        red_flowers = sum(1 for f in self.flowers[player, : self.flower_counts[player]] if 35 <= f <= 38)
        blue_flowers = sum(1 for f in self.flowers[player, : self.flower_counts[player]] if 39 <= f <= 42)

        if red_flowers == 4:
            tai += 1
        if blue_flowers == 4:
            tai += 1

        # Animal bonus - complete set
        if self.animal_counts[player] == 4:
            tai += 5

        # Flowerless status
        is_flowerless = self.flower_counts[player] == 0 and self.animal_counts[player] == 0

        return tai, is_flowerless

    def check_flush(self, counts: np.ndarray) -> int:
        """Check flush type: 0=none, 2=half, 4=full"""
        suit_mask = np.zeros(3, dtype=bool)
        has_honors = False

        # Check suited tiles
        for suit in range(3):
            if np.any(counts[suit * 9 : (suit + 1) * 9] > 0):
                suit_mask[suit] = True

        # Check honors
        if np.any(counts[27:34] > 0):
            has_honors = True

        num_suits = np.sum(suit_mask)

        if num_suits > 1:
            return 0  # Multiple suits
        elif num_suits == 1:
            return 2 if has_honors else 4
        else:
            return 0

    def find_best_winning_pattern(self, counts: np.ndarray, player: int) -> tuple[str, int]:
        """Find best possible winning pattern and its points"""
        total_tiles = np.sum(counts)

        # Check special hands (only for concealed hands)
        has_melds = np.any(self.meld_counts[player] > 0)

        if not has_melds:
            if self.is_thirteen_wonders(counts):
                return ("Thirteen Wonders", 13)

            # Check nine gates
            if total_tiles == 14:
                for suit in range(3):
                    suit_counts = counts[suit * 9 : (suit + 1) * 9]
                    if np.sum(suit_counts) == 14:
                        if suit_counts[0] >= 3 and suit_counts[8] >= 3:
                            if np.all(suit_counts[1:8] >= 1):
                                return ("Nine Gates", 13)

        # Check flush
        flush_score = self.check_flush(counts)

        # Check flush for melds
        meld_flush = 4
        for meld_idx in range(4):
            if self.meld_counts[player, meld_idx] > 0:
                tile_type = self.meld_tiles[player, meld_idx]
                tile_counts = np.zeros(34, dtype=np.uint8)
                tile_counts[tile_type] = self.meld_counts[player, meld_idx]
                meld_flush = min(meld_flush, self.check_flush(tile_counts))

        combined_flush = min(flush_score, meld_flush) if has_melds else flush_score

        # Check structure
        _, is_flowerless = self.calculate_tai(player)
        hand_structure = 0

        if self.is_sequence_hand_valid(counts, player) and is_flowerless:
            hand_structure = 4
        elif self.is_seven_pairs(counts) and not has_melds:
            hand_structure = 4
        elif self.all_triplets(counts):
            hand_structure = 2
        elif self.can_form_sets(counts):
            hand_structure = 1

        # Check meld constraints
        meld_has_sequence = False
        meld_has_triplet = False
        for meld_idx in range(4):
            if self.meld_counts[player, meld_idx] > 0:
                if self.meld_types[player, meld_idx] == 1:
                    meld_has_sequence = True
                elif self.meld_types[player, meld_idx] in [2, 3, 4]:
                    meld_has_triplet = True

        if has_melds:
            if hand_structure >= 1 and not meld_has_triplet and meld_has_sequence:
                if self.is_sequence_hand_valid(counts, player) and is_flowerless:
                    combined_structure = 4
                elif self.can_form_sets(counts):
                    combined_structure = 1
                else:
                    combined_structure = 0
            elif hand_structure >= 1 and not meld_has_sequence and meld_has_triplet:
                if self.all_triplets(counts):
                    combined_structure = 2
                elif self.can_form_sets(counts):
                    combined_structure = 1
                else:
                    combined_structure = 0
            else:
                combined_structure = 1 if hand_structure >= 1 else 0
        else:
            combined_structure = hand_structure

        # Build pattern name
        flushness = {0: "", 2: "Half Flush", 4: "Full Flush"}[combined_flush]
        structure_names = {4: "Sequence Hand", 2: "Triplets Hand", 1: "Lesser Sequence Hand", 0: ""}
        structure_name = structure_names[combined_structure]

        if not combined_structure and not combined_flush:
            return ("Chicken Hand", 0)
        else:
            final_name = (flushness + " " + structure_name).strip()
            if not final_name:
                final_name = flushness if flushness else "Chicken Hand"

            return (final_name, min(combined_flush + combined_structure, MAX_TAI))

    def is_sequence_hand_valid(self, counts: np.ndarray, player: int) -> bool:
        """Check sequence hand with proper restrictions"""
        for tile_type in range(34):
            if counts[tile_type] >= 2:
                tile = tile_type + 1

                # Skip restricted tiles
                if 28 <= tile <= 31:
                    if tile == self.prevailing_wind:
                        continue
                    seat_wind = EAST + player
                    if tile == seat_wind:
                        continue
                elif 32 <= tile <= 34:
                    continue

                test_counts = counts.copy()
                test_counts[tile_type] -= 2

                if self.can_form_sequences_only(test_counts):
                    return True

        return False

    def can_form_sequences_only(self, counts: np.ndarray) -> bool:
        """Check if can form only sequences"""
        counts = counts.copy()
        total = np.sum(counts)
        if total % 3 != 0:
            return False

        for tile_type in range(27):
            while counts[tile_type] > 0:
                suit = tile_type // 9
                tile_in_suit = tile_type % 9

                if tile_in_suit > 6:
                    return False

                next1, next2 = tile_type + 1, tile_type + 2
                if counts[next1] == 0 or counts[next2] == 0:
                    return False

                counts[tile_type] -= 1
                counts[next1] -= 1
                counts[next2] -= 1

        return np.sum(counts[27:34]) == 0

    def calculate_points(self, player: int) -> tuple[str, int]:
        """Calculate best winning hand and points"""
        counts = self.hand_counts[player].copy()

        pattern_name, base_points = self.find_best_winning_pattern(counts, player)
        tai, _ = self.calculate_tai(player)

        # Add wind/dragon bonuses from melds
        for meld_idx in range(4):
            if self.meld_counts[player, meld_idx] > 0:
                tile_type = self.meld_tiles[player, meld_idx]
                tile = tile_type + 1

                if 32 <= tile <= 34:
                    tai += 1
                elif 28 <= tile <= 31:
                    if tile == self.prevailing_wind:
                        tai += 1
                    seat_wind = EAST + player
                    if tile == seat_wind:
                        tai += 1

        total_points = base_points + tai
        total_points = min(total_points, 5)

        return pattern_name, total_points

    def can_win(self, player: int) -> bool:
        """Check if player can win (with MIN_TAI requirement)"""
        counts = self.hand_counts[player].copy()

        if not self.is_winning_hand(counts):
            return False

        # Check minimum tai requirement
        pattern_name, points = self.calculate_points(player)
        return points >= MIN_TAI

    def is_winning_hand(self, counts: np.ndarray) -> bool:
        """Fast winning hand check"""
        total = np.sum(counts)
        if total % 3 != 2:
            return False

        if self.is_thirteen_wonders(counts):
            return True
        if self.is_seven_pairs(counts):
            return True

        return self.can_form_sets(counts)

    def is_thirteen_wonders(self, counts: np.ndarray) -> bool:
        """Check thirteen wonders"""
        required = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        for tile_type in required:
            if counts[tile_type] == 0:
                return False

        return np.sum(counts) == 14 and np.sum(counts > 0) == 13

    def is_seven_pairs(self, counts: np.ndarray) -> bool:
        """Check seven pairs"""
        return np.sum(counts == 2) == 7 and np.sum(counts) == 14

    def can_form_sets(self, counts: np.ndarray) -> bool:
        """Check if tiles can form 4 sets + 1 pair"""
        counts = counts.copy()

        for tile_type in range(34):
            if counts[tile_type] >= 2:
                counts[tile_type] -= 2
                if self.form_sets_recursive(counts):
                    return True
                counts[tile_type] += 2

        return False

    def form_sets_recursive(self, counts: np.ndarray) -> bool:
        """Recursively form sets"""
        total = np.sum(counts)
        if total == 0:
            return True
        if total % 3 != 0:
            return False

        tile_type = np.argmax(counts > 0)

        # Try triplet
        if counts[tile_type] >= 3:
            counts[tile_type] -= 3
            if self.form_sets_recursive(counts):
                counts[tile_type] += 3
                return True
            counts[tile_type] += 3

        # Try sequence
        if tile_type < 27:
            tile_in_suit = tile_type % 9
            if tile_in_suit <= 6:
                next1, next2 = tile_type + 1, tile_type + 2
                if counts[next1] > 0 and counts[next2] > 0:
                    counts[tile_type] -= 1
                    counts[next1] -= 1
                    counts[next2] -= 1
                    if self.form_sets_recursive(counts):
                        counts[tile_type] += 1
                        counts[next1] += 1
                        counts[next2] += 1
                        return True
                    counts[tile_type] += 1
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
        if tile == 0 or tile > 34:
            return False, False

        tile_type = tile - 1
        count = self.hand_counts[player, tile_type]
        return count >= 2, count >= 3

    def can_chow(self, player: int, tile: np.uint8) -> list[list[int]]:
        """Check possible chows (only from previous player)"""
        if (player - 1) % NUM_PLAYERS != self.last_discard_player:
            return []

        if tile == 0 or tile >= 28:
            return []

        tile_type = tile - 1
        suit = tile_type // 9
        tile_in_suit = tile_type % 9

        possible_chows = []

        for pos in range(3):
            if pos > tile_in_suit:
                continue

            seq_start = tile_in_suit - pos
            if seq_start <= 6:
                # Check if we have the other two tiles
                chow_types = [suit * 9 + seq_start + i for i in range(3)]
                needed = [t for t in chow_types if t != tile_type]

                if all(self.hand_counts[player, t] > 0 for t in needed):
                    possible_chows.append(chow_types)

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
        self.hand_counts.fill(0)
        self.meld_tiles.fill(0)
        self.meld_counts.fill(0)
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
