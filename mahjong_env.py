"""Multi-agent Mahjong environment for MuZero."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict
from mahjong_logic import MahjongGame, NUM_PLAYERS
from mahjongAI.controller import MahjongController
from observation_encoder import ObservationEncoder
from action_space import TOTAL_ACTIONS


class MahjongEnv(gym.Env):
    """
    Multi-agent Singapore Mahjong environment compatible with MuZero.

    Each agent gets their own observation and reward.

    Action space: Discrete(41)
        - 0-33: Discard tile type
        - 34: Skip/no claim
        - 35: Win
        - 36: Pong
        - 37: Kong
        - 38: Chow left
        - 39: Chow middle
        - 40: Chow right

    Observation: Encoded game state vector (269 dimensions)
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        """Initialize environment.

        Args:
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.game = MahjongGame(seed=seed)
        self.controller = MahjongController(self.game)
        self.encoder = ObservationEncoder(num_players=NUM_PLAYERS)
        self.num_agents = NUM_PLAYERS

        # Define spaces
        self.action_space = gym.spaces.Discrete(TOTAL_ACTIONS)
        obs_shape = self.encoder.get_observation_space_shape()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # State tracking
        self.current_round = 0
        self.max_rounds = 16  # 4 winds × 4 dealers
        self.seed_value = seed

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
        """Reset and return observations for all agents.

        Returns:
            observations: Dict mapping agent_id -> observation
            infos: Dict mapping agent_id -> info
        """
        if seed is not None:
            self.seed_value = seed
            self.game = MahjongGame(seed=seed)
            self.controller = MahjongController(self.game)

        self.game.reset()
        self.current_round = 0

        observations = {}
        infos = {}

        for agent_id in range(self.num_agents):
            obs = self.encoder.encode(self.game, agent_id)
            observations[agent_id] = obs
            infos[agent_id] = self._get_info()

        return observations, infos

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, dict]]:
        """Execute actions from agents.

        Args:
            actions: Dict mapping agent_id -> action (only current player or claimers)

        Returns:
            observations: Dict of next states for all agents
            rewards: Dict of rewards for all agents
            dones: Dict of done flags for all agents
            infos: Dict of info dicts for all agents
        """
        # Determine who should act
        if self.game.last_discard > 0:
            # Claim phase - handle claims from all players
            result = self.controller.handle_discard_claims(actions)
        else:
            # Current player's turn
            current_player = self.game.current_player
            if current_player not in actions:
                # No action provided - default to first legal action
                legal_actions = self.controller.get_legal_actions(current_player)
                action = np.where(legal_actions)[0][0]
            else:
                action = actions[current_player]

            result = self.controller.apply_action(action, current_player)

        terminated = result["game_over"]

        # Draw tile if needed
        if not terminated and self.game.last_discard == 0 and not self.game.skip_draw:
            if self.game.wall_ptr < len(self.game.wall):
                tile = self.game.draw_tile()
                self.game.add_tile_to_hand(self.game.current_player, tile)
                self.game.replace_bonus_tiles(self.game.current_player)
            else:
                terminated = True
                self.game.game_over = True
        else:
            self.game.skip_draw = False

        # Build outputs for all agents
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in range(self.num_agents):
            observations[agent_id] = self.encoder.encode(self.game, agent_id)

            # Calculate individual rewards
            if terminated and result["game_over"]:
                winner = result["winner"]
                points = result["points"]

                if winner == agent_id:
                    # Won the game
                    rewards[agent_id] = 100.0 + points * 20.0
                else:
                    # Lost the game
                    rewards[agent_id] = -50.0
            else:
                # No reward for non-terminal actions
                rewards[agent_id] = 0.0

            dones[agent_id] = terminated
            infos[agent_id] = self._get_info()
            infos[agent_id]["result"] = result

        return observations, rewards, dones, infos

    def _get_info(self) -> dict:
        """Get additional information about current state.

        Returns:
            Info dictionary
        """
        return {
            "current_player": self.game.current_player,
            "dealer": self.game.dealer,
            "wall_remaining": self.game.get_tiles_remaining(),
            "game_over": self.game.game_over,
            "winner": self.game.winner,
            "round": self.current_round,
            "prevailing_wind": self.game.prevailing_wind,
        }

    def get_legal_actions(self, player: int | None = None) -> np.ndarray:
        """Get legal action mask.

        Args:
            player: Player to get actions for (default: current player)

        Returns:
            Boolean mask of legal actions
        """
        if player is None:
            player = self.game.current_player
        return self.controller.get_legal_actions(player)

    def render(self, mode: str = "human") -> None:
        """Render environment (not implemented for MuZero)."""
        pass

    def close(self) -> None:
        """Close environment."""
        pass
