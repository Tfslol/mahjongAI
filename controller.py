"""Game controller that orchestrates game flow."""

import os
import numpy as np
from game import MahjongGame
from player import Player
from utils import (
    tile_to_str,
    meld_to_str,
    get_sorted_hand_with_mapping,
    get_input,
    NUM_PLAYERS,
)


class GameController:
    """Orchestrates game flow between game logic and players."""

    def __init__(self, players: list[Player]):
        if len(players) != NUM_PLAYERS:
            raise ValueError(f"Need exactly {NUM_PLAYERS} players")

        self.game = MahjongGame()
        self.players = players

    def handle_discard_claims(self) -> bool:
        """Handle claiming of discarded tile.

        Returns:
            True if game ended (win), False otherwise
        """
        if self.game.last_discard == 0 or self.game.last_discard_player < 0:
            return False

        print(
            f"\nLast discard: {tile_to_str(self.game.last_discard)} from Player {self.game.last_discard_player + 1}"
        )

        # Priority 1: Check for wins
        for offset in range(1, NUM_PLAYERS):
            check_player = (self.game.last_discard_player + offset) % NUM_PLAYERS

            # Temporarily add tile to check win
            self.game.add_tile_to_hand(check_player, self.game.last_discard)
            can_win = self.game.can_win(check_player)

            if can_win:
                if self.players[check_player].should_claim_win(
                    self.game, self.game.last_discard
                ):
                    self.game.current_player = check_player
                    self.game.game_over = True
                    self.game.winner = check_player
                    pattern_name, points = self.game.calculate_points(check_player)
                    self.game.points_scored = points
                    return True

            # Remove tile
            self.game.hand_sizes[check_player] -= 1
            self.game.hands[check_player, self.game.hand_sizes[check_player]] = (
                np.uint8(0)
            )

        # Priority 2: Pong/Kong
        for offset in range(1, NUM_PLAYERS):
            check_player = (self.game.last_discard_player + offset) % NUM_PLAYERS
            can_pong, can_kong = self.game.can_pong_kong(
                check_player, self.game.last_discard
            )

            if can_kong:
                if self.players[check_player].should_claim_kong(
                    self.game, self.game.last_discard
                ):
                    self.game.current_player = check_player
                    self.game.remove_tiles_from_hand(
                        check_player, [self.game.last_discard] * 3
                    )
                    self.game.add_meld(check_player, [self.game.last_discard] * 4, 3)
                    self.game.last_discard = np.uint8(0)

                    # Draw replacement
                    if self.game.wall_ptr < len(self.game.wall):
                        tile = self.game.draw_tile()
                        self.game.add_tile_to_hand(check_player, tile)
                        self.game.replace_bonus_tiles(check_player)

                    return False

            elif can_pong:
                if self.players[check_player].should_claim_pong(
                    self.game, self.game.last_discard
                ):
                    self.game.current_player = check_player
                    self.game.remove_tiles_from_hand(
                        check_player, [self.game.last_discard] * 2
                    )
                    self.game.add_meld(check_player, [self.game.last_discard] * 3, 2)
                    self.game.last_discard = np.uint8(0)
                    self.game.skip_draw = True
                    return False

        # Priority 3: Chow (only next player)
        next_player = (self.game.last_discard_player + 1) % NUM_PLAYERS
        chow_options = self.game.can_chow(next_player, self.game.last_discard)

        if chow_options and self.game.current_player == next_player:
            choice = self.players[next_player].choose_chow(
                self.game, self.game.last_discard, chow_options
            )

            if choice is not None:
                chow = chow_options[choice]
                tiles_to_remove = [t for t in chow if t != self.game.last_discard]
                self.game.remove_tiles_from_hand(next_player, tiles_to_remove)
                self.game.add_meld(next_player, chow, 1)
                self.game.last_discard = np.uint8(0)
                self.game.skip_draw = True
                return False

        # No claims
        self.game.last_discard = np.uint8(0)
        return False

    def execute_discard_action(self, action_type: str, value: int | None):
        """Execute the action chosen by current player."""
        player = self.game.current_player

        if action_type == "win":
            self.game.game_over = True
            self.game.winner = player
            pattern_name, points = self.game.calculate_points(player)
            self.game.points_scored = points

        elif action_type == "hidden_kong":
            self.game.remove_tiles_from_hand(player, [value] * 4)
            self.game.add_meld(player, [value] * 4, 4)

            if self.game.wall_ptr < len(self.game.wall):
                tile = self.game.draw_tile()
                self.game.add_tile_to_hand(player, tile)
                self.game.replace_bonus_tiles(player)

        elif action_type == "upgrade_kong":
            # Remove 1 tile from hand
            hand = self.game.hands[player, : self.game.hand_sizes[player]]
            for i in range(len(hand)):
                if hand[i] == value:
                    self.game.remove_tile_from_hand(player, i)
                    break

            self.game.upgrade_pong_to_kong(player, value)

            if self.game.wall_ptr < len(self.game.wall):
                tile = self.game.draw_tile()
                self.game.add_tile_to_hand(player, tile)
                self.game.replace_bonus_tiles(player)

        elif action_type == "discard":
            tile = self.game.remove_tile_from_hand(player, value)
            if self.game.discard_counts[player] < 30:
                self.game.discards[player, self.game.discard_counts[player]] = tile
                self.game.discard_counts[player] += 1

            self.game.last_discard = tile
            self.game.last_discard_player = player
            self.game.current_player = (self.game.current_player + 1) % NUM_PLAYERS

    def display_results(self):
        """Display end-of-game results."""
        os.system("cls" if os.name == "nt" else "clear")

        if self.game.winner >= 0:
            pattern_name, points = self.game.calculate_points(self.game.winner)
            print(f"\n🎉 Player {self.game.winner + 1} wins!")
            print(f"Winning hand: {pattern_name}")
            print(f"Total points: {points}")

            winner = self.game.winner
            hand = self.game.hands[winner, : self.game.hand_sizes[winner]]
            sorted_tiles, _ = get_sorted_hand_with_mapping(hand)
            tiles_str = "  ".join([f"{tile_to_str(t):<2s}" for t in sorted_tiles])
            print(f"Hand: {tiles_str}")

            melds_str = " ".join(
                [
                    meld_to_str(
                        self.game.melds[winner, i], self.game.meld_types[winner, i]
                    )
                    for i in range(4)
                    if self.game.melds[winner, i, 0] > 0
                ]
            )
            if melds_str:
                print(f"Melds: {melds_str}")

            tai, is_flowerless = self.game.calculate_tai(winner)
            flowerless_str = " flowerless" if is_flowerless else ""
            print(f"Tai: {tai}{flowerless_str}")
        else:
            print("\n🤝 Game ended in draw!")

    def play_round(self):
        """Play one complete round."""
        self.game.reset()

        while not self.game.game_over:
            # Handle discard claims first
            if self.game.last_discard > 0:
                if self.handle_discard_claims():
                    break

            # Draw tile if needed
            if self.game.last_discard == 0 and not self.game.skip_draw:
                if self.game.wall_ptr < len(self.game.wall):
                    tile = self.game.draw_tile()
                    self.game.add_tile_to_hand(self.game.current_player, tile)
                    self.game.replace_bonus_tiles(self.game.current_player)
                else:
                    self.game.game_over = True
                    break
            else:
                self.game.skip_draw = False

            # Get current player's action
            current_player = self.game.current_player
            player = self.players[current_player]

            action_type, value = player.choose_discard_action(self.game)
            self.execute_discard_action(action_type, value)

            # Break if game ended
            if self.game.game_over:
                break

        # Display results
        self.display_results()

        # Advance dealer
        self.game.advance_dealer(
            self.game.winner == self.game.dealer if self.game.winner >= 0 else False
        )

    def play_game(self):
        """Play complete game (multiple rounds)."""
        while not self.game.game_complete:
            self.play_round()

            if self.game.game_complete:
                print("\n🏁 Game complete! All rounds finished.")
                break

            get_input("\nPress Enter to continue to next round", [""])
