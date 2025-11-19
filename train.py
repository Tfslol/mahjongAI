"""MCTS with Neural Network training for Mahjong using LightZero framework."""

import numpy as np
import torch
import torch.nn as nn
import os
from typing import List, Tuple, Dict, Optional
from collections import deque
import random

# LightZero imports
try:
    from lzero.policy import MuZeroPolicy
    from lzero.entry import train_muzero
    from ding.config import compile_config
    from ding.envs import BaseEnvManager
    from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator

    LIGHTZERO_AVAILABLE = True
except ImportError:
    print("LightZero not available, using custom implementation")
    LIGHTZERO_AVAILABLE = False

from game import MahjongGame
from controller import GameController
from player import Player
from utils import NUM_PLAYERS, MAX_HAND_SIZE


# =============================================================================
# Dynamic Action Space Handler
# =============================================================================


class ActionSpace:
    """Dynamic action space that adapts based on game state."""

    @staticmethod
    def get_available_actions(game: MahjongGame, player: int) -> Dict[str, List]:
        """Get all available actions with their parameters.

        Returns:
            Dict with action types and their options
        """
        actions = {
            "discard": [],
            "chow": [],
            "pong": False,
            "kong": False,
            "win": False,
            "hidden_kong": [],
            "upgrade_kong": [],
        }

        # Discard actions - all tiles in hand
        for i in range(game.hand_sizes[player]):
            actions["discard"].append(i)

        # Claim actions if there's a last discard
        if game.last_discard > 0 and game.last_discard_player != player:
            # Chow options (can be multiple sequences)
            chow_options = game.can_chow(player, game.last_discard)
            if chow_options:
                actions["chow"] = chow_options

            # Pong/Kong
            can_pong, can_kong = game.can_pong_kong(player, game.last_discard)
            actions["pong"] = can_pong
            actions["kong"] = can_kong

        # Win
        actions["win"] = game.can_win(player)

        # Kong actions
        actions["hidden_kong"] = game.can_hidden_kong(player)
        actions["upgrade_kong"] = game.can_upgrade_kong(player)

        return actions

    @staticmethod
    def actions_to_indices(actions: Dict) -> List[Tuple[str, int, Optional[List]]]:
        """Convert action dict to list of (action_type, action_id, params).

        Returns:
            List of tuples where:
            - action_type: 'discard', 'chow', 'pong', etc.
            - action_id: unique ID for this specific action
            - params: additional parameters (e.g., which chow sequence)
        """
        action_list = []
        action_id = 0

        # Discard actions
        for tile_idx in actions["discard"]:
            action_list.append(("discard", action_id, tile_idx))
            action_id += 1

        # Chow actions (one per option)
        for chow_idx, chow_seq in enumerate(actions["chow"]):
            action_list.append(("chow", action_id, (chow_idx, chow_seq)))
            action_id += 1

        # Pong
        if actions["pong"]:
            action_list.append(("pong", action_id, None))
            action_id += 1

        # Kong
        if actions["kong"]:
            action_list.append(("kong", action_id, None))
            action_id += 1

        # Win
        if actions["win"]:
            action_list.append(("win", action_id, None))
            action_id += 1

        # Hidden kongs
        for kong_tile in actions["hidden_kong"]:
            action_list.append(("hidden_kong", action_id, kong_tile))
            action_id += 1

        # Upgrade kongs
        for kong_tile in actions["upgrade_kong"]:
            action_list.append(("upgrade_kong", action_id, kong_tile))
            action_id += 1

        return action_list


# =============================================================================
# State Encoding (reuse from existing code)
# =============================================================================


def encode_state(game: MahjongGame, player: int) -> np.ndarray:
    """Encode game state for neural network (9316 features)."""
    state_parts = []

    # 1. Own hand (14 * 47 = 658)
    hand_encoding = np.zeros((14, 47), dtype=np.float32)
    for i in range(game.hand_sizes[player]):
        tile = game.hands[player, i]
        if tile > 0:
            hand_encoding[i, tile] = 1.0
    state_parts.append(hand_encoding.flatten())

    # 2. Own melds (4 * 4 * 47 = 752)
    meld_encoding = np.zeros((4, 4, 47), dtype=np.float32)
    for m in range(4):
        for t in range(4):
            tile = game.melds[player, m, t]
            if tile > 0:
                meld_encoding[m, t, tile] = 1.0
    state_parts.append(meld_encoding.flatten())

    # 3. Visible discards (4 * 30 * 47 = 5640)
    discard_encoding = np.zeros((4, 30, 47), dtype=np.float32)
    for p in range(4):
        for i in range(game.discard_counts[p]):
            tile = game.discards[p, i]
            if tile > 0:
                discard_encoding[p, i, tile] = 1.0
    state_parts.append(discard_encoding.flatten())

    # 4. Other melds (3 * 4 * 4 * 47 = 2256)
    other_meld_encoding = np.zeros((3, 4, 4, 47), dtype=np.float32)
    idx = 0
    for p in range(4):
        if p == player:
            continue
        for m in range(4):
            for t in range(4):
                tile = game.melds[p, m, t]
                if tile > 0:
                    other_meld_encoding[idx, m, t, tile] = 1.0
        idx += 1
    state_parts.append(other_meld_encoding.flatten())

    # 5. Metadata (10)
    metadata = np.array(
        [
            player / 3.0,
            game.current_player / 3.0,
            game.dealer / 3.0,
            (game.prevailing_wind - 28) / 3.0 if game.prevailing_wind >= 28 else 0.0,
            game.get_tiles_remaining() / 148.0,
            game.hand_sizes[player] / 14.0,
            1.0 if game.last_discard > 0 else 0.0,
            game.last_discard_player / 3.0 if game.last_discard_player >= 0 else 0.0,
            1.0 if game.can_win(player) else 0.0,
            len(game.can_hidden_kong(player)) / 4.0,
        ],
        dtype=np.float32,
    )
    state_parts.append(metadata)

    return np.concatenate(state_parts)


# =============================================================================
# Neural Network
# =============================================================================


class MahjongNet(nn.Module):
    """Neural network with dynamic action support."""

    def __init__(self, state_size: int = 9316, hidden_size: int = 512, max_actions: int = 50):
        super().__init__()

        self.max_actions = max_actions

        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Policy head outputs logits for up to max_actions
        self.policy_head = nn.Sequential(nn.Linear(hidden_size // 2, 256), nn.ReLU(), nn.Linear(256, max_actions))

        # Value head
        self.value_head = nn.Sequential(nn.Linear(hidden_size // 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, state, action_mask=None):
        """Forward pass with optional action masking.

        Args:
            state: (batch, state_size) tensor
            action_mask: (batch, max_actions) boolean tensor
        """
        shared = self.shared(state)
        policy_logits = self.policy_head(shared)
        value = self.value_head(shared)

        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to -inf so softmax gives 0 probability
            policy_logits = policy_logits.masked_fill(~action_mask, float("-inf"))

        return policy_logits, value


# =============================================================================
# MCTS AI Player (using dynamic actions)
# =============================================================================


class MCTSPlayerDynamic(Player):
    """AI player using MCTS with dynamic action space."""

    def __init__(self, player_id: int, model: MahjongNet, device: torch.device, num_simulations: int = 100):
        super().__init__(player_id)
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.model.eval()

    def get_action_probs(self, game: MahjongGame) -> Tuple[List, np.ndarray]:
        """Get action probabilities using neural network.

        Returns:
            action_list: List of available actions
            probs: Probability distribution over actions
        """
        # Get available actions
        actions_dict = ActionSpace.get_available_actions(game, self.player_id)
        action_list = ActionSpace.actions_to_indices(actions_dict)

        if not action_list:
            return [], np.array([])

        # Encode state
        state = encode_state(game, self.player_id)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Create action mask
        action_mask = torch.zeros(1, self.model.max_actions, dtype=torch.bool).to(self.device)
        for i, (_, action_id, _) in enumerate(action_list):
            if action_id < self.model.max_actions:
                action_mask[0, action_id] = True

        # Get policy from network
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor, action_mask)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Extract probabilities for valid actions
        probs = np.array([policy[action_id] for _, action_id, _ in action_list])
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(action_list)) / len(action_list)

        return action_list, probs

    def choose_discard_action(self, game: MahjongGame) -> Tuple[str, int | None]:
        """Choose action using neural network."""
        action_list, probs = self.get_action_probs(game)

        if not action_list:
            return ("discard", 0)

        # Sample action
        idx = np.random.choice(len(action_list), p=probs)
        action_type, _, params = action_list[idx]

        if action_type == "discard":
            return ("discard", params)
        elif action_type == "win":
            return ("win", None)
        elif action_type == "hidden_kong":
            return ("hidden_kong", params)
        elif action_type == "upgrade_kong":
            return ("upgrade_kong", params)
        else:
            return ("discard", 0)

    def should_claim_win(self, game: MahjongGame, tile: np.uint8) -> bool:
        action_list, probs = self.get_action_probs(game)
        for (action_type, _, _), prob in zip(action_list, probs):
            if action_type == "win" and prob > 0.5:
                return True
        return False

    def should_claim_kong(self, game: MahjongGame, tile: np.uint8) -> bool:
        action_list, probs = self.get_action_probs(game)
        for (action_type, _, _), prob in zip(action_list, probs):
            if action_type == "kong" and prob > 0.3:
                return True
        return False

    def should_claim_pong(self, game: MahjongGame, tile: np.uint8) -> bool:
        action_list, probs = self.get_action_probs(game)
        for (action_type, _, _), prob in zip(action_list, probs):
            if action_type == "pong" and prob > 0.3:
                return True
        return False

    def choose_chow(self, game: MahjongGame, tile: np.uint8, options: list) -> int | None:
        action_list, probs = self.get_action_probs(game)

        chow_actions = [
            (i, (action_type, params, prob))
            for i, ((action_type, _, params), prob) in enumerate(zip(action_list, probs))
            if action_type == "chow"
        ]

        if not chow_actions:
            return None

        # Find best chow option
        best_idx = max(chow_actions, key=lambda x: x[1][2])
        chow_idx, _ = best_idx[1][1]

        return chow_idx if best_idx[1][2] > 0.3 else None


# =============================================================================
# Self-Play
# =============================================================================


def self_play_game(
    model: MahjongNet, device: torch.device, temperature: float = 1.0
) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
    """Play one game collecting training data.

    Returns:
        List of (state, action_probs, num_actions, outcome)
    """
    players = [MCTSPlayerDynamic(i, model, device, num_simulations=50) for i in range(4)]
    controller = GameController(players)

    # Track states and actions
    history = []  # (state, action_probs, num_actions, player)

    game = controller.game

    while not game.game_over:
        player_id = game.current_player
        player = players[player_id]

        # Get state and action probabilities
        state = encode_state(game, player_id)
        action_list, probs = player.get_action_probs(game)

        if not action_list:
            break

        # Create full action probability vector
        action_vector = np.zeros(model.max_actions, dtype=np.float32)
        for (_, action_id, _), prob in zip(action_list, probs):
            if action_id < model.max_actions:
                action_vector[action_id] = prob

        # Apply temperature
        if temperature != 1.0:
            action_vector = action_vector ** (1.0 / temperature)
            if action_vector.sum() > 0:
                action_vector = action_vector / action_vector.sum()

        # Store for training
        history.append((state, action_vector, len(action_list), player_id))

        # Execute action through controller
        try:
            action_type, value = player.choose_discard_action(game)
            controller.execute_discard_action(action_type, value)

            # Handle claims if needed
            if game.last_discard > 0:
                controller.handle_discard_claims()

            # Draw if needed
            if game.last_discard == 0 and not game.skip_draw:
                if game.wall_ptr < len(game.wall):
                    tile = game.draw_tile()
                    game.add_tile_to_hand(game.current_player, tile)
                    game.replace_bonus_tiles(game.current_player)
                else:
                    game.game_over = True
            else:
                game.skip_draw = False

        except Exception as e:
            print(f"Error during self-play: {e}")
            break

    # Assign outcomes
    winner = game.winner
    training_data = []

    for state, action_probs, num_actions, player_id in history:
        if winner == player_id:
            outcome = 1.0
        elif winner >= 0:
            outcome = -1.0
        else:
            outcome = 0.0

        training_data.append((state, action_probs, num_actions, outcome))

    return training_data


# =============================================================================
# Training Loop
# =============================================================================


def train_model(num_iterations: int = 100, games_per_iter: int = 10, batch_size: int = 64, lr: float = 0.001):
    """Train Mahjong AI with dynamic action space."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = MahjongNet(state_size=9316, hidden_size=512, max_actions=50)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    replay_buffer = deque(maxlen=10000)

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Self-play
        temperature = 1.0 if iteration < 20 else 0.5

        for game_idx in range(games_per_iter):
            training_data = self_play_game(model, device, temperature)
            replay_buffer.extend(training_data)

            if (game_idx + 1) % 5 == 0:
                print(f"  Played {game_idx + 1}/{games_per_iter} games")

        # Train
        if len(replay_buffer) >= batch_size:
            print(f"Training on {len(replay_buffer)} samples...")
            model.train()

            losses = []
            for epoch in range(5):
                batch_data = random.sample(list(replay_buffer), batch_size)

                states = torch.FloatTensor(np.array([x[0] for x in batch_data])).to(device)
                target_policies = torch.FloatTensor(np.array([x[1] for x in batch_data])).to(device)
                target_values = torch.FloatTensor(np.array([[x[3]] for x in batch_data])).to(device)

                policy_logits, values = model(states)

                policy_loss = -(target_policies * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
                value_loss = ((values - target_values) ** 2).mean()
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())

            scheduler.step()
            print(f"  Loss: {np.mean(losses):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            model.eval()

        # Save checkpoint
        if (iteration + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_iter_{iteration+1}.pth")
            print(f"  Saved checkpoint")

    torch.save(model.state_dict(), "mahjong_model_final.pth")
    print("\nTraining complete!")


if __name__ == "__main__":
    train_model(num_iterations=200, games_per_iter=20, batch_size=128, lr=0.002)
