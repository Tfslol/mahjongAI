"""Player classes for Mahjong game."""

from abc import ABC, abstractmethod
import os
import numpy as np
from game import MahjongGame
from utils import (
    tile_to_str,
    meld_to_str,
    wind_to_str,
    get_input,
    get_sorted_hand_with_mapping,
    BOOL_OPTIONS,
)


class Player(ABC):
    """Abstract base class for players."""

    def __init__(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def choose_discard_action(self, game: MahjongGame) -> tuple[str, int | None]:
        """Choose action during player's turn.

        Returns:
            (action_type, value) where:
            - ('discard', tile_index)
            - ('hidden_kong', tile_value)
            - ('upgrade_kong', tile_value)
            - ('win', None)
        """
        pass

    @abstractmethod
    def should_claim_win(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Decide whether to claim win on discarded tile."""
        pass

    @abstractmethod
    def should_claim_kong(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Decide whether to kong discarded tile."""
        pass

    @abstractmethod
    def should_claim_pong(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Decide whether to pong discarded tile."""
        pass

    @abstractmethod
    def choose_chow(self, game: MahjongGame, tile: np.uint8, options: list) -> int | None:
        """Choose which chow to make, or None to skip.

        Returns:
            Index of chow option, or None to skip
        """
        pass


class HumanPlayer(Player):
    """Human player with console I/O."""

    def display_game_state(self, game: MahjongGame):
        """Display current game state from this player's perspective."""
        os.system("cls" if os.name == "nt" else "clear")

        print(
            f"\n=== {wind_to_str(game.prevailing_wind)} Round, Dealer: Player {game.dealer + 1}, Tiles Left: {game.get_tiles_remaining()} ==="
        )

        # Show other players (hidden hands)
        for p in range(4):
            if p == self.player_id:
                continue

            melds_str = " ".join(
                [meld_to_str(game.melds[p, i], game.meld_types[p, i]) for i in range(4) if game.melds[p, i, 0] > 0]
            )
            tai, is_flowerless = game.calculate_tai(p)
            flowerless_str = " flowerless" if is_flowerless else ""
            discards_str = " ".join([tile_to_str(game.discards[p, i]) for i in range(game.discard_counts[p])])

            print(f"\nPlayer {p+1}: Hand ({game.hand_sizes[p]}) Melds: [{melds_str}] Tai:{tai}{flowerless_str}")
            print(f"  Discards: [{discards_str}]")

        # Show own hand
        p = self.player_id
        hand = game.hands[p, : game.hand_sizes[p]]
        sorted_tiles, index_map = get_sorted_hand_with_mapping(hand)

        melds_str = " ".join(
            [meld_to_str(game.melds[p, i], game.meld_types[p, i]) for i in range(4) if game.melds[p, i, 0] > 0]
        )
        tai, is_flowerless = game.calculate_tai(p)
        flowerless_str = " flowerless" if is_flowerless else ""
        discards_str = " ".join([tile_to_str(game.discards[p, i]) for i in range(game.discard_counts[p])])

        print(f"\nPlayer {p+1}:")

        # Print indices
        indices = "  ".join([f"{i:<2d}" for i in range(len(sorted_tiles))])
        print(f"             {indices}")

        # Print tiles
        tiles_str = "  ".join([f"{tile_to_str(t):<2s}" for t in sorted_tiles])
        print(f"  Hand ({game.hand_sizes[p]}): {tiles_str}")

        print(f"  Melds: [{melds_str}] Tai:{tai}{flowerless_str}")
        print(f"  Discards: [{discards_str}]")

        return sorted_tiles, index_map

    def choose_discard_action(self, game: MahjongGame) -> tuple[str, int | None]:
        """Get action from human player."""
        sorted_tiles, index_map = self.display_game_state(game)

        # Check available actions
        hidden_kongs = game.can_hidden_kong(self.player_id)
        upgrade_kongs = game.can_upgrade_kong(self.player_id)
        can_win = game.can_win(self.player_id)

        print(f"\nPlayer {self.player_id + 1}'s turn:")

        action_str = "Discard {index}"
        if hidden_kongs:
            action_str += f", 'h' to hidden kong {' or '.join([tile_to_str(t) for t in hidden_kongs])}"
        if upgrade_kongs:
            action_str += f", 'u' to upgrade kong {' or '.join([tile_to_str(t) for t in upgrade_kongs])}"
        if can_win:
            action_str += ", 'w' to win"

        print(f"Valid actions: {action_str}")

        while True:
            try:
                user_input = input("Choose action: ").strip().lower()

                if user_input == "w" and can_win:
                    return ("win", None)

                elif user_input == "h" and hidden_kongs:
                    if len(hidden_kongs) == 1:
                        return ("hidden_kong", hidden_kongs[0])
                    else:
                        idx = int(get_input("Choose", [str(i) for i in range(len(hidden_kongs))]))
                        return ("hidden_kong", hidden_kongs[idx])

                elif user_input == "u" and upgrade_kongs:
                    if len(upgrade_kongs) == 1:
                        return ("upgrade_kong", upgrade_kongs[0])
                    else:
                        idx = int(get_input("Choose", [str(i) for i in range(len(upgrade_kongs))]))
                        return ("upgrade_kong", upgrade_kongs[idx])

                else:
                    # Discard action
                    action_num = int(user_input)
                    if 0 <= action_num < len(index_map):
                        return ("discard", index_map[action_num])
                    else:
                        print("Invalid index!")

            except (ValueError, IndexError):
                print("Invalid input!")

    def should_claim_win(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Ask if player wants to claim win."""
        self.display_hand(game)
        response = get_input(f"Player {self.player_id + 1}, win with {tile_to_str(tile)}?", BOOL_OPTIONS)
        return response == "y"

    def should_claim_kong(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Ask if player wants to kong."""
        self.display_hand(game)
        response = get_input(f"Player {self.player_id + 1}, kong {tile_to_str(tile)}?", BOOL_OPTIONS)
        return response == "y"

    def should_claim_pong(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Ask if player wants to pong."""
        self.display_hand(game)
        response = get_input(f"Player {self.player_id + 1}, pong {tile_to_str(tile)}?", BOOL_OPTIONS)
        return response == "y"

    def choose_chow(self, game: MahjongGame, tile: np.uint8, options: list) -> int | None:
        """Ask which chow to make."""
        self.display_hand(game)

        if len(options) == 1:
            response = get_input(f"Player {self.player_id + 1}, chow {tile_to_str(tile)}?", BOOL_OPTIONS)
            return 0 if response == "y" else None
        else:
            print(f"Multiple chow options available:")
            for i, chow in enumerate(options):
                print(f"  {i}: {' '.join([tile_to_str(t) for t in chow])}")

            choice = get_input(
                f"Player {self.player_id + 1}, choose chow option",
                [str(i) for i in range(len(options))] + ["n"],
            )

            return None if choice == "n" else int(choice)

    def display_hand(self, game: MahjongGame):
        # Display current hand
        hand = game.hands[self.player_id, :game.hand_sizes[self.player_id]]
        sorted_tiles, _ = get_sorted_hand_with_mapping(hand)
        tiles_str = "  ".join([tile_to_str(t) for t in sorted_tiles])
        print(f"Your hand: {tiles_str}")


class MCTSPlayer(Player):
    """AI player using MCTS (placeholder for now)."""
    def __init__(self, player_id: int):
        super().__init__(player_id)
        # TODO: Add MCTS/NN model

    def choose_discard_action(self, game: MahjongGame) -> tuple[str, int | None]:
        """Choose action using MCTS."""
        # TODO: Implement MCTS decision
        # For now, just discard first tile
        return ("discard", 0)

    def should_claim_win(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Always claim win."""
        return True

    def should_claim_kong(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Simple heuristic for kong."""
        return True

    def should_claim_pong(self, game: MahjongGame, tile: np.uint8) -> bool:
        """Simple heuristic for pong."""
        return False  # Conservative

    def choose_chow(self, game: MahjongGame, tile: np.uint8, options: list) -> int | None:
        """Simple heuristic for chow."""
        return None  # Don't chow for now
