import numpy as np

Cell = tuple[int, int]  # (row, col)


class Observation:
    """Builds the agent's observation from the environment state.

    channel 0  occupancy  1 = outside the part / hole   (fixed per episode)
    channel 1  coverage   1 = free cell already filled   (changes every step)
    channel 2  agent      1 = agent's cell, else 0       (changes every step)

    Usage:
        obs_fn = Observation(pad_to=64)
        obs = obs_fn(grid, covered, pos)   # np.ndarray[float32], (3, 64, 64)

    Padding: grids have different shapes (H, W). With pad_to, every
    observation is (3, pad_to, pad_to) so episodes can be batched. The grid
    sits in the top-left corner; padded cells are occupied (channel 0 = 1),
    so to the agent they look like walls.
    """

    n_channels = 3
    OCCUPANCY, COVERAGE, AGENT = 0, 1, 2  # channel indices

    def __init__(self, pad_to: int | None = None):
        """
        Args:
            pad_to: side of the square output. None keeps the grid's own
                (H, W). Must be >= max(H, W) of every grid.
        """
        self.pad_to = pad_to

    def __call__(
        self, grid: np.ndarray, covered: np.ndarray, pos: Cell
    ) -> np.ndarray:
        """Observation for one step.

        Args:
            grid: np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
            covered: np.ndarray[bool], shape (H, W), 1 = free cell filled.
            pos: (row, col) of the agent.

        Returns:
            np.ndarray[float32], shape (3, H, W), or (3, pad_to, pad_to)
            with padding. Values are 0 or 1.
        """
        if grid.shape != covered.shape:
            raise ValueError(
                f"grid {grid.shape} and covered {covered.shape} differ"
            )
        h, w = grid.shape
        out_h, out_w = self.shape(grid)[1:]
        if out_h < h or out_w < w:
            raise ValueError(
                f"grid {grid.shape} is larger than pad_to={self.pad_to}"
            )

        obs = np.zeros((self.n_channels, out_h, out_w), dtype=np.float32)
        obs[self.OCCUPANCY] = 1.0  # padding = occupied
        obs[self.OCCUPANCY, :h, :w] = grid
        obs[self.COVERAGE, :h, :w] = covered
        obs[self.AGENT, pos[0], pos[1]] = 1.0
        return obs

    def shape(self, grid: np.ndarray) -> tuple[int, int, int]:
        """Shape of the observation for this grid.

        Returns:
            tuple[int, int, int], (3, H, W) or (3, pad_to, pad_to).
        """
        if self.pad_to is None:
            return (self.n_channels, *grid.shape)
        return (self.n_channels, self.pad_to, self.pad_to)
