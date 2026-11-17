import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..dataset import LoadDataset
from .observation import Observation
from .actions import ActionSpace, Cell
from .rewards import Rewards


class Environment:
    """Coverage of one grid: the agent moves one cell per step and fills the
    free cells it visits. The episode ends when every free cell is filled
    (terminated) or after max_steps (truncated).

    Usage:
        env = Environment(cfg)              # samples from the train split
        obs = env.reset()                   # random grid of the split
        obs = env.reset(grid=my_grid)       # a given grid, e.g. for evaluation
        obs, reward, terminated, truncated = env.step(action)
        env.loader.data["val"]              # the loaded splits

    Config read from cfg:
        cfg.dataset            LoadDataset             (required)
        cfg.rewards            Rewards(**cfg.rewards)  (default Rewards())
        cfg.env.pad_to         Observation padding     (default None)
        cfg.env.max_steps      step limit              (default None)
        cfg.env.sample_seed    shape sampler seed      (default None)
    """

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        """
        Args:
            cfg: the full run config (configs/config.yaml).
            split: split reset() samples from, "train", "val" or "test".

        Missing optional keys use the defaults:
            max_steps None -> 4 x the number of free cells of the grid.
            sample_seed None -> new seed every run (logged by LoadDataset).
        """

        rewards_cfg = cfg.rewards
        self.split = split
        self.loader = LoadDataset(cfg.dataset, sample_seed=cfg.env.sample_seed)
        self.loader.load_dataset()
        self.rewards = Rewards(**rewards_cfg) if rewards_cfg else Rewards()
        self.obs_fn = Observation(pad_to=cfg.env.pad_to)
        self.actions = ActionSpace()
        self.max_steps = cfg.env.max_steps

        # Episode state, set by reset()
        self.grid: np.ndarray  # (H, W) bool, 1 = occupied
        self.covered: np.ndarray  # (H, W) bool, 1 = free cell filled
        self.pos: Cell  # (row, col) of the agent
        self.obs: np.ndarray  # (3, H, W) float32, last observation
        self.n_steps = 0
        self.step_limit = 0

    def reset(self, grid: np.ndarray | None = None) -> np.ndarray:
        """Start a new episode, agent on the first free cell (already filled).

        Args:
            grid: grid to cover; None samples one from self.split.

        Returns:
            np.ndarray[float32], shape (3, H, W), first observation.
        """
        self.grid = self.loader.sample(self.split) if grid is None else grid
        free_cells = np.argwhere(~self.grid)  # (n_free, 2), (row, col)
        if len(free_cells) == 0:
            raise ValueError("grid has no free cells")

        self.pos = (int(free_cells[0][0]), int(free_cells[0][1]))
        self.covered = np.zeros_like(self.grid, dtype=bool)
        self.covered[self.pos] = True
        self.n_steps = 0
        self.step_limit = self.max_steps or 4 * len(free_cells)

        self.obs = self.obs_fn(self.grid, self.covered, self.pos)
        return self.obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """Move the agent one cell.

        Args:
            action: Action or its int value (0 to 3).

        Returns:
            observation: np.ndarray[float32], shape (3, H, W).
            reward: float, see Rewards.
            terminated: bool, True if every free cell is filled.
            truncated: bool, True if the step limit was reached first.
        """
        self.pos, _ = self.actions.move(self.pos, action, self.grid.shape)
        if not self.grid[self.pos]:
            self.covered[self.pos] = True
        self.n_steps += 1

        prev_obs = self.obs
        self.obs = self.obs_fn(self.grid, self.covered, self.pos)
        reward = self.rewards.get_reward(prev_obs, self.obs)

        terminated = self.coverage() == 1.0
        truncated = not terminated and self.n_steps >= self.step_limit
        return self.obs, reward, terminated, truncated

    def coverage(self) -> float:
        """Fraction of free cells filled so far, in [0, 1]."""
        return float(self.covered.sum() / (~self.grid).sum())

    def action_mask(self) -> np.ndarray:
        """Which actions keep the agent on the grid.

        Returns:
            np.ndarray[bool], shape (4,), True = allowed, indexed by Action.
        """
        return self.actions.mask(self.pos, self.grid.shape)
