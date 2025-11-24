"""Observation encoding for MuZero-compatible Mahjong environment."""

import numpy as np
from typing import Any


class ObservationEncoder:
    """Encodes game state into tensor observation for neural networks."""

    def __init__(self, num_players: int = 4, max_discards: int = 40):
        self.num_players = num_players
        self.max_discards = max_discards
        self.obs_size = self._calculate_obs_size()

    def _calculate_obs_size(self) -> int:
        """Calculate total observation size.

        Observation components:
        - Own hand tile counts: 34
        - Own melds (simplified): 4 melds × 3 values (tile_type, count, meld_type) = 12
        - Own tai: 1
        - Own flowerless status: 1
        - Last discard tile type (raw index): 1
        - Sequential discards for each player: 4 × 40 = 160
        - Other players' visible melds: 3 × 12 = 36
        - Other players' hand sizes: 3
        - Other players' tai: 3
        - Other players' flowerless: 3
        - Current player (one-hot): 4
        - Dealer (one-hot): 4
        - Prevailing wind (one-hot): 4
        - Wall tiles remaining: 1
        - Game phase indicators: 2 (draw_phase, claim_phase)

        Total: 34 + 12 + 1 + 1 + 1 + 160 + 36 + 3 + 3 + 3 + 4 + 4 + 4 + 1 + 2 = 269
        """
        own_hand = 34
        own_melds = 12
        own_tai = 1
        own_flowerless = 1
        last_discard = 1
        discards = self.num_players * self.max_discards
        other_melds = (self.num_players - 1) * 12
        other_hand_sizes = self.num_players - 1
        other_tai = self.num_players - 1
        other_flowerless = self.num_players - 1
        current_player = self.num_players
        dealer = self.num_players
        wind = 4
        wall = 1
        phase = 2

        return (
            own_hand
            + own_melds
            + own_tai
            + own_flowerless
            + last_discard
            + discards
            + other_melds
            + other_hand_sizes
            + other_tai
            + other_flowerless
            + current_player
            + dealer
            + wind
            + wall
            + phase
        )

    def encode(self, game: Any, perspective_player: int) -> np.ndarray:
        """Encode game state from perspective of given player.

        Args:
            game: MahjongGame instance
            perspective_player: Player index to encode from

        Returns:
            Flattened observation vector
        """
        obs_parts = []

        # Own hand tile counts (34)
        hand_counts = game.hand_counts[perspective_player]
        obs_parts.append(hand_counts.astype(np.float32))

        # Own melds (12: 4 melds × 3 values each)
        own_melds_flat = np.zeros(12, dtype=np.float32)
        for i in range(4):
            if game.meld_counts[perspective_player, i] > 0:
                own_melds_flat[i * 3] = game.meld_tiles[perspective_player, i]  # tile type
                own_melds_flat[i * 3 + 1] = game.meld_counts[perspective_player, i]  # count
                own_melds_flat[i * 3 + 2] = game.meld_types[perspective_player, i]  # type
        obs_parts.append(own_melds_flat)

        # Own tai and flowerless (2)
        tai, is_flowerless = game.calculate_tai(perspective_player)
        obs_parts.append(np.array([tai], dtype=np.float32))
        obs_parts.append(np.array([1.0 if is_flowerless else 0.0], dtype=np.float32))

        # Last discard (1: raw tile type, 34 = no discard)
        if game.last_discard > 0:
            last_discard_idx = game.last_discard - 1
        else:
            last_discard_idx = 34
        obs_parts.append(np.array([last_discard_idx], dtype=np.float32))

        # Sequential discards for all players (4 × 40 = 160)
        for p in range(self.num_players):
            player_discards = np.zeros(self.max_discards, dtype=np.float32)
            for i in range(min(game.discard_counts[p], self.max_discards)):
                tile = game.discards[p, i]
                if tile > 0:
                    player_discards[i] = tile - 1  # Convert to 0-33
                else:
                    player_discards[i] = 34  # Empty slot
            obs_parts.append(player_discards)

        # Other players' melds and info
        for p in range(self.num_players):
            if p == perspective_player:
                continue

            # Melds (12)
            melds_flat = np.zeros(12, dtype=np.float32)
            for i in range(4):
                if game.meld_counts[p, i] > 0:
                    melds_flat[i * 3] = game.meld_tiles[p, i]
                    melds_flat[i * 3 + 1] = game.meld_counts[p, i]
                    melds_flat[i * 3 + 2] = game.meld_types[p, i]
            obs_parts.append(melds_flat)

            # Hand size (1)
            obs_parts.append(np.array([np.sum(game.hand_counts[p])], dtype=np.float32))

            # Tai and flowerless (2)
            tai, is_flowerless = game.calculate_tai(p)
            obs_parts.append(np.array([tai], dtype=np.float32))
            obs_parts.append(np.array([1.0 if is_flowerless else 0.0], dtype=np.float32))

        # Current player (one-hot, 4)
        current_player_onehot = np.zeros(self.num_players, dtype=np.float32)
        current_player_onehot[game.current_player] = 1.0
        obs_parts.append(current_player_onehot)

        # Dealer (one-hot, 4)
        dealer_onehot = np.zeros(self.num_players, dtype=np.float32)
        dealer_onehot[game.dealer] = 1.0
        obs_parts.append(dealer_onehot)

        # Prevailing wind (one-hot, 4)
        wind_onehot = np.zeros(4, dtype=np.float32)
        wind_idx = game.prevailing_wind - 28  # 28=EAST, 29=SOUTH, 30=WEST, 31=NORTH
        if 0 <= wind_idx < 4:
            wind_onehot[wind_idx] = 1.0
        obs_parts.append(wind_onehot)

        # Wall tiles remaining (1)
        tiles_remaining = np.array([game.get_tiles_remaining()], dtype=np.float32)
        obs_parts.append(tiles_remaining)

        # Game phase indicators (2)
        draw_phase = 1.0 if game.last_discard == 0 else 0.0
        claim_phase = 1.0 if game.last_discard > 0 else 0.0
        obs_parts.append(np.array([draw_phase, claim_phase], dtype=np.float32))

        # Concatenate all parts
        observation = np.concatenate(obs_parts)

        # Verify size
        assert (
            observation.shape[0] == self.obs_size
        ), f"Observation size mismatch: {observation.shape[0]} != {self.obs_size}"

        return observation

    def get_observation_space_shape(self) -> tuple:
        """Get shape for gym observation space."""
        return (self.obs_size,)
