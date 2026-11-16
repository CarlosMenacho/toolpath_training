import logging
from pathlib import Path
import numpy as np
import trimesh

from ..geometry import GridBuilder

log = logging.getLogger(__name__)


class GridDataset:
    """Coverage grids of one split (train, val or test).

    Every access loads the mesh and builds its grid from scratch (no cache),
    so a change in GridBuilder is always applied.

    Usage (normally created by LoadDataset):
        train = GridDataset(ids, raw_dir, GridBuilder(n_cells=64))
        len(train)                  # number of meshes
        grid = train[0]             # grid of the first mesh
        grid = train.sample(rng)    # random grid, e.g. for env.reset()
        grids = dict(train.items()) # {id: grid}, e.g. for metrics.evaluate()
    """

    def __init__(
        self,
        ids: list[str],
        raw_dir: Path,
        builder: GridBuilder,
    ):
        """
        Args:
            ids: mesh ids of this split.
            raw_dir: folder with the mesh files.
            builder: turns a mesh into a grid.
        """
        self.ids = ids
        self.raw_dir = raw_dir
        self.builder = builder

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> np.ndarray:
        """Grid of the i-th mesh.

        Returns:
            np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
        """
        mesh_id = self.ids[i]
        mesh_path = next(self.raw_dir.glob(f"{mesh_id}.*"))  # .stl, .obj, ...
        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"{mesh_path} did not load as a mesh")
        return self.builder.from_mesh(mesh)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Grid of a random mesh of this split.

        Returns:
            np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
        """
        return self[int(rng.integers(len(self)))]

    def items(self):
        """All grids of this split, one by one.

        Yields:
            tuple[str, np.ndarray[bool]], mesh id and grid of shape (H, W),
            0 = free, 1 = occupied.
        """
        for i, mesh_id in enumerate(self.ids):
            yield mesh_id, self[i]
