import csv
import random
import logging
from pathlib import Path
from omegaconf import DictConfig

from .dataset import GridDataset
from ..geometry import GridBuilder

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


class LoadDataset:
    """Reads meta.csv, splits the mesh ids in code and builds one GridDataset
    per split. No split files: the same seed always gives the same split.

    Usage:
        data = LoadDataset(cfg.dataset).load_dataset()
        train, val, test = data["train"], data["val"], data["test"]
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: the dataset config (cfg.dataset, configs/dataset/thingi10k.yaml).
        """
        self.cfg = cfg

    def load_dataset(self) -> dict[str, GridDataset]:
        """Read, split and wrap the data.

        Returns:
            dict[str, GridDataset], {"train": ..., "val": ..., "test": ...}.
        """
        ids = self.read_ids()
        splits = self.split_ids(ids)
        builder = GridBuilder(n_cells=self.cfg.n_cells)
        return {
            name: GridDataset(
                split_ids,
                Path(self.cfg.raw_dir),
                builder,
            )
            for name, split_ids in splits.items()
        }

    def read_ids(self) -> list[str]:
        """Mesh ids from meta.csv that loaded and are not too large.

        Returns:
            list[str], mesh ids, sorted.
        """
        ids = []
        with open(self.cfg.meta_csv) as f:
            for row in csv.DictReader(f):
                if row["load_ok"] != "True":
                    continue
                if int(row["n_faces"]) > self.cfg.max_faces:
                    continue
                ids.append(row["id"])
        log.info("%d usable meshes in %s", len(ids), self.cfg.meta_csv)
        return sorted(ids)

    def split_ids(self, ids: list[str]) -> dict[str, list[str]]:
        """Shuffle ids with the config seed and cut them into three splits.

        Meshes in split.force_test always go to test.

        Returns:
            dict[str, list[str]], {"train": ids, "val": ids, "test": ids}.
        """
        s = self.cfg.split
        force_test = [str(i) for i in s.force_test]
        rest = [i for i in ids if i not in force_test]
        random.Random(s.seed).shuffle(rest)

        n_train = int(s.fractions[0] * len(ids))
        n_val = int(s.fractions[1] * len(ids))
        splits = {
            "train": rest[:n_train],
            "val": rest[n_train:n_train + n_val],
            "test": force_test + rest[n_train + n_val:],
        }
        for name in SPLITS:
            log.info("%s: %d meshes", name, len(splits[name]))
        return splits
