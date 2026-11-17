import logging
import random
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.toolpath_rl.envs import Environment
from src.toolpath_rl.utils import RerunVisualizer, plot_grid

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Output dir: %s", HydraConfig.get().runtime.output_dir)
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        log.warning("CUDA not available, falling back to %s", device)

    log.info(f"Loading the dataset")

    env = Environment(cfg=cfg, split="train")

    # Random policy, a few episodes
    rng = np.random.default_rng(cfg.seed)
    rrd_path = f"{HydraConfig.get().runtime.output_dir}/random_policy.rrd"
    viz = RerunVisualizer(save_path=rrd_path)
    log.info("Rerun recording: %s", rrd_path)

    obs = env.reset()
    episode, episode_return = 0, 0.0
    viz.start_episode(f"episode_{episode}", env.grid)
    viz.log_step(0, env.grid, env.covered, env.pos)
    viz.log_observation(0, obs)
    for _ in range(1000):
        action = env.actions.sample(rng)
        log.warning(f"{action} has type of {type(action)}")
        prev_pos, n_covered = env.pos, int(env.covered.sum())
        obs, reward, terminated, truncated = env.step(action)
        episode_return += reward

        new_cell = int(env.covered.sum()) > n_covered
        viz.log_step(
            env.n_steps,
            env.grid,
            env.covered,
            env.pos,
            prev_pos=prev_pos,
            new_cell=new_cell,
            reward=reward
        )
        viz.log_observation(env.n_steps, obs)

        if terminated or truncated:
            log.info(
                f"episode {episode}: steps {env.n_steps}, "
                f"coverage {env.coverage():.1%}, return {episode_return:.2f}, "
                f"terminated {terminated}"
            )
            obs = env.reset()
            episode, episode_return = episode + 1, 0.0
            viz.start_episode(f"episode_{episode}", env.grid)
            viz.log_step(0, env.grid, env.covered, env.pos)
            viz.log_observation(0, obs)

        log.debug(
            f"episode {episode}: steps {env.n_steps}, "
            f"coverage {env.coverage():.1%}, return {episode_return:.2f}, "
            f"terminated {terminated}"
        )

    plot_grid(grid=obs[0], show=True)
    plot_grid(grid=obs[1], show=True)
    plot_grid(grid=obs[2], show=True)


if __name__ == "__main__":
    main()
