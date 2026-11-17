"""Live / recorded visualization of coverage episodes with Rerun.

Needs rerun-sdk >= 0.23 (pip install rerun-sdk). Each step is logged on a
"step" timeline, so the Rerun viewer can scrub through an episode.

Logged entities (under the episode name):
    map          segmentation image: free / occupied / covered cells
    agent        point at the agent's cell
    path         path so far, blue = filled a new cell, orange = wasted
    coverage     scalar plot, % of free cells filled
    reward       scalar plot (only if given)
"""
from pathlib import Path

import numpy as np
import rerun as rr

Cell = tuple[int, int]  # (row, col)

# Class ids of the map image, same colors as plot_data.py
FREE, OCCUPIED, COVERED = 0, 1, 2
CLASS_COLORS = [
    (FREE, "free", (255, 255, 255)),
    (OCCUPIED, "occupied", (217, 217, 217)),
    (COVERED, "covered", (0, 114, 178)),
]
FILL_COLOR = (0, 114, 178)  # step that fills a new cell
WASTE_COLOR = (213, 94, 0)  # step that fills nothing
AGENT_COLOR = (58, 58, 58)


class RerunVisualizer:
    """Logs grids, agent and path of coverage episodes to Rerun.

    Usage, step by step (e.g. inside the env loop):
        viz = RerunVisualizer(spawn=True)
        viz.start_episode("episode_0", grid)
        viz.log_step(t, grid, covered, pos, reward=r)

    Usage, replay a finished path (e.g. Zigzag):
        viz.log_trajectory("zigzag/ring", grid, traj)

    Usage, record to a file and open it later with `rerun file.rrd`:
        viz = RerunVisualizer(spawn=False, save_path="outputs/run.rrd")
    """

    def __init__(
        self,
        app_id: str = "toolpath_rl",
        spawn: bool = True,
        save_path: str | Path | None = None,
    ):
        """
        Args:
            app_id: name shown in the Rerun viewer.
            spawn: open the Rerun viewer and stream to it live.
            save_path: also save everything to this .rrd file. None does not save.
        """
        rr.init(app_id)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(save_path)  # Rerun records to one sink: the file
        elif spawn:
            rr.spawn()
        self._episode = "episode"
        self._segments: list[np.ndarray] = []  # (2, 2) xy per step
        self._wasted: list[bool] = []

    def start_episode(self, name: str, grid: np.ndarray) -> None:
        """Begin a new episode: resets the path and logs the color legend.

        Args:
            name: entity path of the episode, e.g. "train/episode_12".
            grid: np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
        """
        self._episode = name
        self._segments = []
        self._wasted = []
        rr.log(name, rr.AnnotationContext(CLASS_COLORS), static=True)
        rr.log(f"{name}/info", rr.TextDocument(
            f"grid {grid.shape[1]}×{grid.shape[0]}, "
            f"{int((~grid).sum())} free cells"
        ), static=True)

    def log_step(
        self,
        t: int,
        grid: np.ndarray,
        covered: np.ndarray,
        pos: Cell,
        prev_pos: Cell | None = None,
        new_cell: bool = True,
        reward: float | None = None,
    ) -> None:
        """Log the state after step t.

        Args:
            t: step index (0 = start of episode).
            grid: np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
            covered: np.ndarray[bool], shape (H, W), 1 = free cell filled.
            pos: (row, col) of the agent now.
            prev_pos: (row, col) before this step. None at t = 0 (no path).
            new_cell: True if this step filled a new cell (path color).
            reward: reward of this step. None does not log it.
        """
        name = self._episode
        rr.set_time("step", sequence=t)

        # Map: 0 free, 1 occupied, 2 covered
        cell_map = grid.astype(np.uint8)
        cell_map[covered] = COVERED
        rr.log(f"{name}/map", rr.SegmentationImage(cell_map))

        # Agent at the centre of its cell (image x = col, y = row)
        rr.log(f"{name}/agent", rr.Points2D(
            [_center(pos)], colors=[AGENT_COLOR], radii=0.35
        ))

        # Path so far, one 2-point strip per step
        if prev_pos is not None:
            self._segments.append(np.array([_center(prev_pos), _center(pos)]))
            self._wasted.append(not new_cell)
            colors = [WASTE_COLOR if w else FILL_COLOR for w in self._wasted]
            rr.log(f"{name}/path", rr.LineStrips2D(
                self._segments, colors=colors, radii=0.12
            ))

        n_free = int((~grid).sum())
        coverage = 100.0 * int(covered.sum()) / max(n_free, 1)
        rr.log(f"{name}/coverage", rr.Scalars(coverage))
        if reward is not None:
            rr.log(f"{name}/reward", rr.Scalars(reward))

    def log_trajectory(
        self, name: str, grid: np.ndarray, traj: np.ndarray, every: int = 1
    ) -> None:
        """Replay a finished path step by step (e.g. from Zigzag).

        Args:
            name: entity path of the episode, e.g. "zigzag/ring".
            grid: np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
            traj: np.ndarray[int], shape (T+1, 2), (row, col) at every step.
            every: log only every k-th step (faster for long paths); the
                last step is always logged.
        """
        self.start_episode(name, grid)
        covered = np.zeros_like(grid, dtype=bool)
        prev = None
        for t, (r, c) in enumerate(np.asarray(traj, dtype=int)):
            pos = (int(r), int(c))
            new_cell = not grid[pos] and not covered[pos]
            if new_cell:
                covered[pos] = True
            if t % every == 0 or t == len(traj) - 1:
                self.log_step(t, grid, covered, pos, prev, new_cell)
            elif prev is not None:  # keep the path complete between logs
                self._segments.append(np.array([_center(prev), _center(pos)]))
                self._wasted.append(not new_cell)
            prev = pos

    def log_observation(self, t: int, obs: np.ndarray) -> None:
        """Log the 3 channels the agent sees, as separate images.

        Args:
            t: step index.
            obs: np.ndarray[float32], shape (3, S, S), from Observation.
        """
        rr.set_time("step", sequence=t)
        for i, channel in enumerate(("occupancy", "coverage", "agent")):
            image = (obs[i] * 255).astype(np.uint8)
            rr.log(f"{self._episode}/obs/{channel}", rr.Image(image))


def _center(pos: Cell) -> tuple[float, float]:
    """(row, col) -> (x, y) of the cell centre in image coordinates."""
    return (pos[1] + 0.5, pos[0] + 0.5)
