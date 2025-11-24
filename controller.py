"""Environment-safe controller for Mahjong game - no I/O, action-driven."""

import numpy as np
from mahjong_logic import MahjongGame, NUM_PLAYERS
from action_space import decode_action, ACTION_SKIP


class MahjongController:
    """Orchestrates game flow based on integer actions."""

    def __init__(self, game: MahjongGame):
        self.game = game
        # Cache for legal actions to avoid recomputation
        self._legal_cache = {}
        self._cache_valid = False

    def invalidate_cache(self):
        """Invalidate legal action cache after state change"""
        self._cache_valid = False
        self._legal_cache = {}

    def apply_action(self, action: int, player: int) -> dict:
        """Apply action to game state.

        Args:
            action: Integer action in range [0, 40]
            player: Player making the action

        Returns:
            Result dict with keys:
            - success: bool
            - game_over: bool
            - winner: int (-1 if no winner)
            - points: int
            - error: str (if any)
        """
        self.invalidate_cache()
        action_obj = decode_action(action)

        if action_obj.type == "discard":
            return self._apply_discard(player, action_obj.tile_type)
        elif action_obj.type == "win":
            return self._apply_win(player)
        elif action_obj.type == "pong":
            return self._apply_pong(player)
        elif action_obj.type == "kong":
            return self._apply_kong(player)
        elif action_obj.type == "chow":
            return self._apply_chow(player, action_obj.chow_pos)
        elif action_obj.type == "skip":
            return self._apply_skip()
        else:
            return {
                "success": False,
                "game_over": False,
                "winner": -1,
                "points": 0,
                "error": f"Unknown action type: {action_obj.type}",
            }

    def _apply_discard(self, player: int, tile_type: int) -> dict:
        """Apply discard action."""
        if self.game.hand_counts[player, tile_type] == 0:
            return {
                "success": False,
                "game_over": False,
                "winner": -1,
                "points": 0,
                "error": f"Tile type {tile_type} not in hand",
            }

        # Remove tile and add to discards
        self.game.hand_counts[player, tile_type] -= 1
        tile = tile_type + 1

        if self.game.discard_counts[player] < 40:
            self.game.discards[player, self.game.discard_counts[player]] = tile
            self.game.discard_counts[player] += 1

        self.game.last_discard = tile
        self.game.last_discard_player = player
        self.game.current_player = (player + 1) % NUM_PLAYERS

        return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

    def _apply_win(self, player: int) -> dict:
        """Apply win action."""
        if not self.game.can_win(player):
            return {
                "success": False,
                "game_over": False,
                "winner": -1,
                "points": 0,
                "error": "Cannot win with current hand",
            }

        self.game.game_over = True
        self.game.winner = player
        pattern_name, points = self.game.calculate_points(player)
        self.game.points_scored = points

        return {
            "success": True,
            "game_over": True,
            "winner": player,
            "points": points,
            "error": "",
            "pattern": pattern_name,
        }

    def _apply_pong(self, player: int) -> dict:
        """Apply pong action on last discard."""
        if self.game.last_discard == 0:
            return {"success": False, "game_over": False, "winner": -1, "points": 0, "error": "No tile to pong"}

        can_pong, _ = self.game.can_pong_kong(player, self.game.last_discard)

        if not can_pong:
            return {"success": False, "game_over": False, "winner": -1, "points": 0, "error": "Cannot pong this tile"}

        tile_type = self.game.last_discard - 1
        self.game.remove_tiles_from_hand(player, tile_type, 2)
        self.game.add_meld(player, tile_type, 3, 2)
        self.game.last_discard = np.uint8(0)
        self.game.skip_draw = True
        self.game.current_player = player

        return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

    def _apply_kong(self, player: int) -> dict:
        """Apply kong action (either hidden or upgrade or claim)."""
        # Check if claiming kong from discard
        if self.game.last_discard > 0:
            _, can_kong = self.game.can_pong_kong(player, self.game.last_discard)
            if can_kong:
                tile_type = self.game.last_discard - 1
                self.game.remove_tiles_from_hand(player, tile_type, 3)
                self.game.add_meld(player, tile_type, 4, 3)
                self.game.last_discard = np.uint8(0)
                self.game.current_player = player

                # Draw replacement
                if self.game.wall_ptr < len(self.game.wall):
                    tile = self.game.draw_tile()
                    self.game.add_tile_to_hand(player, tile)
                    self.game.replace_bonus_tiles(player)

                return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

        # Check hidden kong
        hidden_kongs = self.game.can_hidden_kong(player)
        if hidden_kongs:
            tile_type = hidden_kongs[0]
            self.game.remove_tiles_from_hand(player, tile_type, 4)
            self.game.add_meld(player, tile_type, 4, 4)

            if self.game.wall_ptr < len(self.game.wall):
                tile = self.game.draw_tile()
                self.game.add_tile_to_hand(player, tile)
                self.game.replace_bonus_tiles(player)

            return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

        # Check upgrade kong
        upgrade_kongs = self.game.can_upgrade_kong(player)
        if upgrade_kongs:
            tile_type = upgrade_kongs[0]
            self.game.hand_counts[player, tile_type] -= 1
            self.game.upgrade_pong_to_kong(player, tile_type)

            if self.game.wall_ptr < len(self.game.wall):
                tile = self.game.draw_tile()
                self.game.add_tile_to_hand(player, tile)
                self.game.replace_bonus_tiles(player)

            return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

        return {"success": False, "game_over": False, "winner": -1, "points": 0, "error": "No kong available"}

    def _apply_chow(self, player: int, chow_pos: int) -> dict:
        """Apply chow action."""
        if self.game.last_discard == 0:
            return {"success": False, "game_over": False, "winner": -1, "points": 0, "error": "No tile to chow"}

        chow_options = self.game.can_chow(player, self.game.last_discard)

        if not chow_options or chow_pos >= len(chow_options):
            return {
                "success": False,
                "game_over": False,
                "winner": -1,
                "points": 0,
                "error": f"Invalid chow position {chow_pos}",
            }

        chow = chow_options[chow_pos]
        tile_type = self.game.last_discard - 1

        # Remove the two tiles we need from hand
        for t in chow:
            if t != tile_type:
                self.game.hand_counts[player, t] -= 1

        # Add chow meld (use first tile of sequence)
        self.game.add_meld(player, chow[0], 3, 1)
        self.game.last_discard = np.uint8(0)
        self.game.skip_draw = True
        self.game.current_player = player

        return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

    def _apply_skip(self) -> dict:
        """Skip claim (no action)."""
        # Clear last discard if in claim phase
        if self.game.last_discard > 0:
            self.game.last_discard = np.uint8(0)

        return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

    def handle_discard_claims(self, actions: dict[int, int]) -> dict:
        """Handle claims on discarded tile from multiple players.

        Args:
            actions: Dict mapping player_id -> action

        Returns:
            Result dict from highest priority claim
        """
        if self.game.last_discard == 0 or self.game.last_discard_player < 0:
            return {"success": False, "game_over": False, "winner": -1, "points": 0, "error": "No discard to claim"}

        # Priority 1: Check for wins
        for offset in range(1, NUM_PLAYERS):
            check_player = (self.game.last_discard_player + offset) % NUM_PLAYERS

            if check_player in actions:
                action_obj = decode_action(actions[check_player])
                if action_obj.type == "win":
                    # Temporarily add tile to check win
                    tile_type = self.game.last_discard - 1
                    self.game.hand_counts[check_player, tile_type] += 1
                    can_win = self.game.can_win(check_player)

                    if can_win:
                        return self._apply_win(check_player)

                    # Remove tile if can't win
                    self.game.hand_counts[check_player, tile_type] -= 1

        # Priority 2: Pong/Kong
        for offset in range(1, NUM_PLAYERS):
            check_player = (self.game.last_discard_player + offset) % NUM_PLAYERS

            if check_player in actions:
                action_obj = decode_action(actions[check_player])

                if action_obj.type == "kong":
                    _, can_kong = self.game.can_pong_kong(check_player, self.game.last_discard)
                    if can_kong:
                        return self._apply_kong(check_player)

                elif action_obj.type == "pong":
                    can_pong, _ = self.game.can_pong_kong(check_player, self.game.last_discard)
                    if can_pong:
                        return self._apply_pong(check_player)

        # Priority 3: Chow (only next player)
        next_player = (self.game.last_discard_player + 1) % NUM_PLAYERS
        if next_player in actions:
            action_obj = decode_action(actions[next_player])
            if action_obj.type == "chow":
                chow_options = self.game.can_chow(next_player, self.game.last_discard)
                if chow_options and action_obj.chow_pos is not None:
                    return self._apply_chow(next_player, action_obj.chow_pos)

        # No valid claims - clear discard
        self.game.last_discard = np.uint8(0)
        return {"success": True, "game_over": False, "winner": -1, "points": 0, "error": ""}

    def get_legal_actions(self, player: int) -> np.ndarray:
        """Get legal action mask for player.

        Returns:
            Boolean array of length 41
        """
        # Check cache
        if self._cache_valid and player in self._legal_cache:
            return self._legal_cache[player].copy()

        mask = np.zeros(41, dtype=bool)

        # Check if in claim phase or discard phase
        if self.game.last_discard > 0:
            # Claim phase
            mask[ACTION_SKIP] = True  # Can always skip

            # Can we claim this discard?
            if player != self.game.last_discard_player:
                # Check win
                tile_type = self.game.last_discard - 1
                self.game.hand_counts[player, tile_type] += 1
                if self.game.can_win(player):
                    mask[35] = True  # Win
                self.game.hand_counts[player, tile_type] -= 1

                # Check pong/kong
                can_pong, can_kong = self.game.can_pong_kong(player, self.game.last_discard)
                if can_pong:
                    mask[36] = True  # Pong
                if can_kong:
                    mask[37] = True  # Kong

                # Check chow (only next player)
                if (self.game.last_discard_player + 1) % NUM_PLAYERS == player:
                    chow_options = self.game.can_chow(player, self.game.last_discard)
                    if chow_options:
                        for i in range(min(len(chow_options), 3)):
                            mask[38 + i] = True

        elif player == self.game.current_player:
            # Discard phase (current player's turn)

            # Mark available discards
            for tile_type in range(34):
                if self.game.hand_counts[player, tile_type] > 0:
                    mask[tile_type] = True

            # Check win
            if self.game.can_win(player):
                mask[35] = True

            # Check hidden kong
            hidden_kongs = self.game.can_hidden_kong(player)
            if hidden_kongs:
                mask[37] = True

            # Check upgrade kong
            upgrade_kongs = self.game.can_upgrade_kong(player)
            if upgrade_kongs:
                mask[37] = True

        # Ensure at least one legal action
        if not np.any(mask):
            # Emergency: if no legal actions, allow skip
            mask[ACTION_SKIP] = True

        # Cache result
        self._legal_cache[player] = mask.copy()
        self._cache_valid = True

        return mask
