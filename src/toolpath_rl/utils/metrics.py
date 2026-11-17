import numpy as np


def filled_curve(free, traj):
    """f_t = number of distinct free cells visited after t steps, t = 0..T.

    free: bool [H, W], True = cell to fill.
    traj: int [T+1, 2], (row, col) of the agent at every step.

    Raises ValueError if the path leaves the grid or makes a move that is not
    one 4-neighbour step, since the metrics assume those rules.
    """
    free = np.asarray(free, dtype=bool)
    traj = np.asarray(traj, dtype=int)
    if traj.ndim != 2 or traj.shape[1] != 2 or len(traj) == 0:
        raise ValueError(f"traj must have shape [T+1, 2], got {traj.shape}")
    h, w = free.shape
    r, c = traj.T
    if r.min() < 0 or c.min() < 0 or r.max() >= h or c.max() >= w:
        raise ValueError(f"traj leaves the {h}x{w} grid")
    bad = np.abs(np.diff(traj, axis=0)).sum(axis=1) != 1
    if bad.any():
        t = int(np.argmax(bad))
        raise ValueError(
            f"step {t}->{t + 1} is not a 4-neighbour move: "
            f"{tuple(traj[t].tolist())} -> {tuple(traj[t + 1].tolist())}"
        )
    # A step fills a cell only on the first visit of a free cell
    _, first_visit = np.unique(r * w + c, return_index=True)
    new = np.zeros(len(traj), dtype=bool)
    new[first_visit] = True
    new &= free[r, c]
    return np.cumsum(new)


def path_metrics(free, traj):
    """Coverage metrics of one path on one grid.

    n_cells:            free cells n
    steps:              T* (first step with full coverage), or T if incomplete
    wasted_steps:       R, steps up to `steps` that filled nothing
    complete:           every free cell was filled
    coverage_pct:       % of free cells filled after `steps`
    speed_pct_per_step: 100 / T*         (NaN if incomplete)
    cells_per_step:     (n - 1) / T*     (NaN if incomplete), 1.0 = no waste

    Steps after full coverage are ignored.
    """
    filled = filled_curve(free, traj)
    n = int(np.count_nonzero(free))
    if n == 0:
        raise ValueError("grid has no free cells")
    done = np.flatnonzero(filled == n)
    complete = done.size > 0
    steps = int(done[0]) if complete else len(filled) - 1
    gained = int(filled[steps] - filled[0])
    ok = complete and steps > 0
    return {
        "n_cells": n,
        "steps": steps,
        "wasted_steps": steps - gained,
        "complete": complete,
        "coverage_pct": 100.0 * int(filled[steps]) / n,
        "speed_pct_per_step": 100.0 / steps if ok else float("nan"),
        "cells_per_step": (n - 1) / steps if ok else float("nan"),
    }


def evaluate(grids, planners):
    """Run every planner on every grid and score the paths.

    grids:    {geometry name: bool [H, W]}
    planners: {method name: callable(free) -> traj [T+1, 2]}

    Returns
        rows:  list of dicts, one per (geometry, method), metrics included
        trajs: {(geometry, method): traj}, for plotting

    Example:
        planners = {
            "Zigzag": lambda free: zigzag(free)[1],
            "RL": lambda free: rollout(agent, free),
        }
        rows, trajs = evaluate(grids, planners)
    """
    rows, trajs = [], {}
    for geometry, free in grids.items():
        for method, plan in planners.items():
            traj = np.asarray(plan(free))
            trajs[geometry, method] = traj
            rows.append({
                "geometry": geometry,
                "method": method,
                **path_metrics(free, traj),
            })
    return rows, trajs
