"""Build the coverage grid from a part's geometry.

Pipeline:
    3D mesh --slice at height z--> 2D polygons (mm) --rasterize--> bool grid

The grid is the problem that every planner (Zigzag, RL agent) solves and that
toolpath_rl.utils.metrics scores. Details: docs/zigzag_metric.md, section 1.
"""
import numpy as np
import shapely
import trimesh
import logging
from shapely.geometry import Polygon
from shapely.ops import unary_union

log = logging.getLogger(__name__)


class GridBuilder:
    """Turns a 3D mesh or 2D polygons into a bool grid [H, W].

    Grid convention:
        free[row, col] == True   cell inside the part, must be filled
        free[row, col] == False  outside the part or inside a hole
        row 0 is the top of the part (largest y), col 0 the left (smallest x)

    Scaling: the longest side of the shape's bounding box gets n_cells cells,
    so every geometry gives a grid of about the same size. The cell size in mm
    therefore differs from part to part.

    Usage:
        builder = GridBuilder(n_cells=64)
        free = builder.from_mesh(mesh)         # slice at mid height
        free = builder.from_mesh(mesh, z=0.1)  # slice at z
        free = builder.from_polygons(polys)    # shapes that are already 2D

    Attributes:
        n_cells: number of cells on the longest side of every grid.
    """

    def __init__(self, n_cells: int = 64):
        """
        Args:
            n_cells: grid resolution, cells on the longest side (default 64).
        """
        self.n_cells = n_cells
        log.debug(f"Generating an environment using {n_cells} cells")

    def _slice(self, mesh: trimesh.Trimesh, z: float) -> list[Polygon]:
        """Cut the mesh with a horizontal plane at height z (helper).

        The cut is flattened by shifting it down by z only, so the polygons
        keep the part's own x/y coordinates in mm (no rotation).

        Args:
            mesh: the part, in mm.
            z: height of the cutting plane, in mm.

        Returns:
            list[Polygon], length K (one per region, holes included),
            float coords in mm.

        Raises:
            ValueError: if the plane at z does not cross the mesh.
        """
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            log.warning(f"Got an error getting the section")
            z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
            log.error(
                "z = {z} is outside the mesh ({z_min:.2f} to {z_max:.2f})"
            )
            raise ValueError(
                f"z = {z} is outside the mesh ({z_min:.2f} to {z_max:.2f})"
            )

        # Only shift down by z, so x/y stay the part's own coordinates
        shift_down = trimesh.transformations.translation_matrix([0, 0, -z])
        planar, _ = section.to_2D(to_2D=shift_down)
        return list(planar.polygons_full)

    def from_mesh(
        self, mesh: trimesh.Trimesh, z: float | None = None
    ) -> np.ndarray:
        """Slice the mesh at height z and rasterize the cross-section.
        Args:
            mesh: the part, in mm.
            z: height of the cut in mm. None (default) cuts at mid height,
                (z_min + z_max) / 2.
        Returns:
            np.ndarray[bool], shape (H, W), True = cell to fill.
        Raises:
            ValueError: if z is outside the mesh.
        """
        mid_height = float(mesh.bounds[:, 2].mean())
        z_cut = mid_height if z is None else z
        return self.from_polygons(self._slice(mesh, z_cut))

    def from_polygons(self, polys: list[Polygon]) -> np.ndarray:
        """Rasterize 2D polygons: a cell is free if its centre is inside.
        Steps:
            1. merge the polygons into one region
            2. cell size = longest side of the bounding box / n_cells
            3. compute the (x, y) centre of every cell
            4. True where the centre lies inside the region
        Example: a 10 x 4 mm rectangle with n_cells = 5 has cell size 2 mm,
        column centres x = 1, 3, 5, 7, 9 and row centres y = 3, 1 (top
        first), giving a 2 x 5 grid of all True.

        Note: features thinner than one cell can disappear or break apart,
        because only the cell centre is tested.
        Args:
            polys: shapely Polygons, any units (usually mm).
        Returns:
            np.ndarray[bool], shape (H, W), True = cell to fill.
            max(H, W) = n_cells.
        """
        region = unary_union(polys)
        x0, y0, x1, y1 = region.bounds
        cell = max(x1 - x0, y1 - y0) / self.n_cells
        n_cols = int(np.ceil((x1 - x0) / cell))
        n_rows = int(np.ceil((y1 - y0) / cell))

        # Centre of every column and row
        xs = x0 + (np.arange(n_cols) + 0.5) * cell
        ys = y0 + (np.arange(n_rows) + 0.5) * cell

        gx, gy = np.meshgrid(xs, ys[::-1])  # row 0 = top of the part
        return shapely.contains_xy(region, gx, gy)
