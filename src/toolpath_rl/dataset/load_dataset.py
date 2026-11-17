import csv
import random
import logging
from pathlib import Path
import numpy as np
from omegaconf import DictConfig

from .dataset import GridDataset
from ..geometry import GridBuilder

log = logging.getLogger(__name__)


class LoadDataset:
    """Reads meta.csv, splits the mesh ids in code and builds one GridDataset
    per split. No split files: the same seed always gives the same split.

    Usage:
        loader = LoadDataset(cfg.dataset)
        data = loader.load_dataset()
        train, val, test = data["train"], data["val"], data["test"]
        grid = loader.sample("train")   # random shape of the train split
    """

    def __init__(self, cfg: DictConfig, sample_seed: int | None = None):
        """
        Args:
            cfg: the dataset config (cfg.dataset, configs/dataset/thingi10k.yaml).
            sample_seed: seed of the shape sampler. None draws a new one from
                the OS every run; it is logged so a run can be replayed.
        """
        self.cfg = cfg
        if sample_seed is None:
            sample_seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
        log.info("Sampling seed: %d", sample_seed)
        self.sample_seed = sample_seed
        self.rng = np.random.default_rng(seed=sample_seed)
        self.data: dict[str, GridDataset] = {}

    def load_dataset(self) -> dict[str, GridDataset]:
        """Read, split and wrap the data.

        The split itself is fixed by cfg.split.seed; only the order of the
        meshes inside each split is shuffled with the sampling rng, so it
        changes every run (and is replayed with the same sample_seed).

        Returns:
            dict[str, GridDataset], {"train": ..., "val": ..., "test": ...}.
        """
        ids = self.read_ids()
        splits = self.split_ids(ids)
        raw_dir = Path(self.cfg.raw_dir)
        builder = GridBuilder(n_cells=self.cfg.n_cells)

        self.data = {}
        for name, split_ids in splits.items():
            shuffled = self.rng.permutation(split_ids).tolist()
            self.data[name] = GridDataset(shuffled, raw_dir, builder)
        return self.data

    def sample(self, split: str = "train") -> np.ndarray:
        """Grid of a random shape of one split, e.g. for env.reset().

        Args:
            split: "train", "val" or "test".

        Returns:
            np.ndarray[bool], shape (H, W), 0 = free, 1 = occupied.
        """
        if not self.data:
            raise RuntimeError("call load_dataset() before sample()")
        dataset = self.data[split]
        i = int(self.rng.integers(len(dataset)))
        log.info("Sampled %s mesh %s", split, dataset.ids[i])
        return dataset[i]

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

        train_end = int(s.fractions[0] * len(ids))
        val_end = train_end + int(s.fractions[1] * len(ids))
        train = rest[:train_end]
        val = rest[train_end:val_end]
        test = force_test + rest[val_end:]

        log.info("split: train %d, val %d, test %d meshes",
                 len(train), len(val), len(test))
        return {"train": train, "val": val, "test": test}
