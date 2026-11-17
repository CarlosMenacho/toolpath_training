from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

FREE_COLOR = "#FFFFFF"  # free cell (0)
OCCUPIED_COLOR = "#D9D9D9"  # occupied cell (1), light so the path stands out
BORDER_COLOR = "#BFBFBF"  # cell borders
FILL_COLOR = "#0072B2"  # step that fills a new cell
WASTE_COLOR = "#D55E00"  # step that fills nothing (revisit or occupied cell)
END_COLOR = "#3A3A3A"  # end marker


def plot_grid(
    grid: np.ndarray,
    traj: np.ndarray | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    legend: bool = True,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Axes:
    """Draw a coverage grid and, optionally, the agent's path on it.

    Args:
        grid: np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
        traj: np.ndarray[int], shape (T+1, 2), (row, col) at every step.
            None draws the grid only.
        ax: axes to draw on. None creates a new figure.
        title: axes title. None writes the grid size and free cell count.
        legend: show the legend (only when traj is given).
        show: open the figure in a window.
        save_path: file to save the whole figure to (.svg, .pdf, .png).
            Missing folders are created. None does not save.

    Returns:
        matplotlib Axes with the plot.

    Usage:
        plot_grid(grid, show=True)                         # look at it
        plot_grid(grid, traj, save_path="outputs/grid.svg")  # save it
        plot_grid(grid, traj, show=True, save_path="outputs/grid.svg")
    """
    grid = np.asarray(grid, dtype=bool)
    h, w = grid.shape
    if ax is None:
        size = 3.0  # inches for the longest side
        _, ax = plt.subplots(
            figsize=(size * w / max(h, w), size * h / max(h, w))
        )

    # Cells: 0 = free (white), 1 = occupied (grey)
    cmap = ListedColormap([FREE_COLOR, OCCUPIED_COLOR])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    # Cell borders, thinner on large grids
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(
        which="minor", color=BORDER_COLOR, lw=0.15 if max(h, w) > 32 else 0.4
    )
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    n_free = int((~grid).sum())
    if traj is not None:
        traj = np.asarray(traj, dtype=int)
        wasted = _wasted_steps(grid, traj)
        axes_pt = ax.bbox.width * 72 / ax.figure.dpi  # axes width in points
        _draw_path(ax, traj, wasted, cell_pt=axes_pt / w)
        if title is None:
            title = f"{len(traj) - 1} steps, {int(wasted.sum())} wasted"
        if legend:
            _add_legend(ax)
    if title is None:
        title = f"{w}x{h}, {n_free} free cells"
    ax.set_title(title)

    fig = ax.get_figure(root=True)
    if fig is None:
        raise ValueError("ax is not attached to a figure")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    return ax


def _wasted_steps(grid: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """Which steps filled nothing.

    Returns:
        np.ndarray[bool], shape (T,), True = step t -> t+1 entered an occupied
        cell or a cell that was already filled.
    """
    filled = np.zeros_like(grid, dtype=bool)
    r, c = traj[0]
    filled[r, c] = not grid[r, c]
    wasted = np.zeros(len(traj) - 1, dtype=bool)
    for t, (r, c) in enumerate(traj[1:]):
        if grid[r, c] or filled[r, c]:
            wasted[t] = True
        else:
            filled[r, c] = True
    return wasted


def _draw_path(
    ax: Axes, traj: np.ndarray, wasted: np.ndarray, cell_pt: float
) -> None:
    """Path segments (blue = fills, orange = wasted) plus start/end markers."""
    lw = float(np.clip(0.3 * cell_pt, 0.6, 3.0))  # follows the cell size
    xy = traj[:, ::-1].astype(float)  # imshow: x = col, y = row
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    for mask, color in [(~wasted, FILL_COLOR), (wasted, WASTE_COLOR)]:
        ax.add_collection(
            LineCollection(
                list(segments[mask]), colors=color, lw=lw, capstyle="round"
            )
        )
    ms = 2.5 * lw + 2
    ax.plot(*xy[0], "o", ms=ms, color=FILL_COLOR, mec="white", mew=0.8)
    ax.plot(*xy[-1], "s", ms=ms, color=END_COLOR, mec="white", mew=0.8)


def _add_legend(ax: Axes) -> None:
    handles = [
        Line2D([], [], color=FILL_COLOR, label="fills a new cell"),
        Line2D([], [], color=WASTE_COLOR, label="wasted step"),
        Line2D([], [],
               marker="o",
               ls="",
               color=FILL_COLOR,
               mec="white",
               label="start"),
        Line2D([], [],
               marker="s",
               ls="",
               color=END_COLOR,
               mec="white",
               label="end"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize="small"
    )
