from collections import deque

import numpy as np

Cell = tuple[int, int]  # (row, col)

# One step up, down, left or right
MOVES: list[Cell] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Zigzag:
    """Boustrophedon (zigzag) coverage of a grid.

    The agent visits the free cells row by row: left to right, then right to
    left, and so on. If the next cell is not a neighbour, it walks the
    shortest path to it. Walking over cells that are already filled counts as
    wasted steps.

    Usage:
        planner = Zigzag()
        traj = planner(free)   # free: bool grid [H, W], True = cell to fill
                               # traj: [T+1, 2], (row, col) at every step
    """

    name = "Zigzag"

    def __call__(self, free: np.ndarray) -> np.ndarray:
        return self.plan(free)

    def plan(self, free: np.ndarray) -> np.ndarray:
        """Return the agent's path that fills every free cell."""
        order = self.snake_order(free)
        if len(order) == 0:
            raise ValueError("grid has no free cells")

        covered = np.zeros_like(free, dtype=bool)
        pos = order[0]
        covered[pos] = True
        traj = [pos]

        for target in order[1:]:
            if covered[target]:
                continue  # already filled while passing through

            path = self.shortest_path(free, pos, target)
            if path is None:
                # Target is on another island: walk across empty cells too
                everywhere = np.ones_like(free, dtype=bool)
                path = self.shortest_path(everywhere, pos, target) or []

            for pos in path:
                traj.append(pos)
                if free[pos]:
                    covered[pos] = True

        return np.array(traj)

    def snake_order(self, free: np.ndarray) -> list[Cell]:
        """Free cells row by row: even rows left->right, odd rows right->left."""
        order = []
        for row in range(free.shape[0]):
            cols = list(np.flatnonzero(free[row]))
            if row % 2 == 1:
                cols.reverse()
            for col in cols:
                order.append((row, int(col)))
        return order

    def shortest_path(self, passable: np.ndarray, start: Cell,
                      goal: Cell) -> list[Cell] | None:
        """Shortest path from start to goal (start not included).

        Breadth-first search over cells where passable is True.
        Returns None if goal cannot be reached.
        """
        height, width = passable.shape
        came_from = {start: start}
        queue = deque([start])

        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for dr, dc in MOVES:
                nxt = (cur[0] + dr, cur[1] + dc)
                inside = 0 <= nxt[0] < height and 0 <= nxt[1] < width
                if inside and passable[nxt] and nxt not in came_from:
                    came_from[nxt] = cur
                    queue.append(nxt)

        if goal not in came_from:
            return None

        # Walk back from goal to start
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path
