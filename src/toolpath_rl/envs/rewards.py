import numpy as np

from .observation import Observation

OCC, COV, AGENT = Observation.OCCUPANCY, Observation.COVERAGE, Observation.AGENT


class Rewards:
    """Reward of one step, computed from the observation before and after it.

    r = w_cover   * cover_reward        (+1 if a new free cell was filled)
      - w_outside * penalty_outside     (1 if the agent is on an occupied cell)
      - w_revisit * penalty_revisiting  (1 if the free cell was already filled)
      - w_time    * time_action         (1 every step)

    Each step is exactly one of: new cell, outside, revisit. Bumping into
    the grid edge (agent stays put) counts as a revisit.

    Usage:
        rewards = Rewards(w_cover=1.0, w_outside=0.5, w_revisit=0.2, w_time=0.05)
        r = rewards.get_reward(prev_obs, obs)
    """

    def __init__(
        self,
        w_cover: float = 1.0,
        w_outside: float = 0.5,
        w_revisit: float = 0.2,
        w_time: float = 0.05,
    ) -> None:
        """
        Args:
            w_cover: reward for filling a new free cell.
            w_outside: penalty for stepping on an occupied cell (hole / outside).
            w_revisit: penalty for stepping on an already filled cell.
            w_time: penalty paid every step, pushes for shorter paths.
        """
        self.w_cover = w_cover
        self.w_outside = w_outside
        self.w_revisit = w_revisit
        self.w_time = w_time

    def get_reward(self, prev_obs: np.ndarray, obs: np.ndarray) -> float:
        """Total reward of the step prev_obs -> obs.

        Args:
            prev_obs: np.ndarray[float32], shape (3, H, W), before the step.
            obs: np.ndarray[float32], shape (3, H, W), after the step.

        Returns:
            float, weighted sum of the terms.
        """
        t = self.terms(prev_obs, obs)
        return (self.w_cover * t["cover"]
                - self.w_outside * t["outside"]
                - self.w_revisit * t["revisit"]
                - self.w_time * t["time"])

    def terms(self, prev_obs: np.ndarray, obs: np.ndarray) -> dict[str, float]:
        """Unweighted terms, useful for logging.

        Returns:
            dict[str, float], keys cover, outside, revisit, time; each 0 or 1.
        """
        pos = agent_pos(obs)
        return {
            "cover": self.cover_reward(prev_obs, pos),
            "outside": self.penalty_outside(obs, pos),
            "revisit": self.penalty_revisiting(prev_obs, pos),
            "time": self.time_action(),
        }

    def cover_reward(self, prev_obs: np.ndarray, pos: tuple[int, int]) -> float:
        """1.0 if the agent is on a free cell that was not filled before."""
        free = prev_obs[OCC][pos] == 0
        was_filled = prev_obs[COV][pos] == 1
        return float(free and not was_filled)

    def penalty_outside(self, obs: np.ndarray, pos: tuple[int, int]) -> float:
        """1.0 if the agent is on an occupied cell (outside the part or a hole)."""
        return float(obs[OCC][pos] == 1)

    def penalty_revisiting(self, prev_obs: np.ndarray, pos: tuple[int, int]) -> float:
        """1.0 if the agent is on a free cell that was already filled."""
        return float(prev_obs[COV][pos] == 1)

    def time_action(self) -> float:
        """1.0 every step (constant cost of moving)."""
        return 1.0


def agent_pos(obs: np.ndarray) -> tuple[int, int]:
    """(row, col) of the agent, read from the agent channel.

    Returns:
        tuple[int, int], (row, col).
    """
    row, col = np.argwhere(obs[AGENT] == 1)[0]
    return int(row), int(col)
