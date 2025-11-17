"""Main entry point for Singapore Mahjong game."""

from controller import GameController
from player import HumanPlayer, MCTSPlayer


def main():
    """Run the game."""
    print("=== Singapore Mahjong ===")
    print("\nGame Modes:")
    print("1. All Human Players (4 humans)")
    print("2. Human vs AI (1 human, 3 AI)")
    print("3. All AI (4 AI)")

    while True:
        try:
            choice = int(input("\nChoose mode (1-3): ").strip())
            if choice in [1, 2, 3]:
                break
            print("Invalid choice!")
        except ValueError:
            print("Invalid input!")

    # Create players based on choice
    if choice == 1:
        # All humans
        players = [HumanPlayer(i) for i in range(4)]
    elif choice == 2:
        # 1 human, 3 AI
        players = [HumanPlayer(0), MCTSPlayer(1), MCTSPlayer(2), MCTSPlayer(3)]
    else:
        # All AI
        players = [MCTSPlayer(i) for i in range(4)]

    # Create controller and start game
    controller = GameController(players)
    controller.play_game()

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
