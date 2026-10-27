import logging
import random

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tour_length(points: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Closed tour length. points: [B, N, 2], order: [B, N] -> [B]."""
    tour = points.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
    return (tour - tour.roll(-1, dims=1)).norm(dim=-1).sum(dim=1)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Output dir: %s", HydraConfig.get().runtime.output_dir)
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        log.warning("CUDA not available, falling back to %s", device)

    # Random planar instances in the unit square
    b, n = cfg.problem.batch_size, cfg.problem.n_nodes
    points = torch.rand(b, n, 2, device=device)

    # Random-order baseline: the number the RL agent has to beat
    order = torch.argsort(torch.rand(b, n, device=device), dim=1)
    cost = tour_length(points, order)
    log.info(
        "[%s] random tour length: %.3f ± %.3f", cfg.algo.name, cost.mean(),
        cost.std()
    )


if __name__ == "__main__":
    main()
