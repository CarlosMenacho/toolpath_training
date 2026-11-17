from enum import IntEnum

import numpy as np

Cell = tuple[int, int]  # (row, col)


class Action(IntEnum):
    """Discrete moves of the agent: one cell per step, 4-connected.

    Row 0 is the top of the grid, so UP decreases the row.

    Usage:
        a = Action.UP
        int(a)          # 0, what the policy outputs
        Action(3)       # Action.RIGHT, from a policy output
        a.delta         # (-1, 0), change in (row, col)
        len(Action)     # 4, size of the action space
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def delta(self) -> Cell:
        """Change in (row, col) for this move.

        Returns:
            tuple[int, int], (d_row, d_col).
        """
        return _DELTAS[self]


_DELTAS: dict[Action, Cell] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


class ActionSpace:
    """Applies actions on a grid of a given shape.

    Moving into an occupied cell is allowed (the agent may cross holes or
    empty space, as Zigzag does; it just fills nothing). Moving off the grid
    is not: the agent stays where it is, and the step is wasted.

    Usage:
        actions = ActionSpace()
        new_pos, moved = actions.move(pos, Action.RIGHT, grid.shape)
        mask = actions.mask(pos, grid.shape)  # which moves stay on the grid
        a = actions.sample(rng)               # random action
    """

    n = len(Action)  # 4

    def move(self, pos: Cell, action: int | Action,
             shape: tuple[int, int]) -> tuple[Cell, bool]:
        """Position after taking an action.

        Args:
            pos: (row, col) of the agent.
            action: Action or its int value (0 to 3).
            shape: (H, W) of the grid.

        Returns:
            tuple[tuple[int, int], bool], new (row, col) and whether the agent
            moved (False if the move would leave the grid).
        """
        dr, dc = Action(action).delta
        row, col = pos[0] + dr, pos[1] + dc
        if not self.inside((row, col), shape):
            return pos, False
        return (row, col), True

    def mask(self, pos: Cell, shape: tuple[int, int]) -> np.ndarray:
        """Which actions keep the agent on the grid (for action masking).

        Returns:
            np.ndarray[bool], shape (4,), True = allowed, indexed by Action.
        """
        return np.array([
            self.inside((pos[0] + a.delta[0], pos[1] + a.delta[1]), shape)
            for a in Action
        ])

    def sample(self, rng: np.random.Generator) -> Action:
        """Random action, e.g. for exploration or a random baseline.

        Returns:
            Action, uniform over the 4 moves.
        """
        return Action(int(rng.integers(self.n)))

    @staticmethod
    def inside(pos: Cell, shape: tuple[int, int]) -> bool:
        """True if (row, col) is on a grid of shape (H, W)."""
        return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]
