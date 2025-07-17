import logging
import os
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf


from common import make_env

# logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./cfgs", config_name="main_cfg")
def train(cfg: DictConfig) -> None:

    OmegaConf.resolve(cfg)
    log.info(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, f=os.path.join(cfg.experiment.output_dir, "cfg.yaml"))

    env_func = partial(make_env, task=cfg.task.name, task_config=cfg.task.config)

    if cfg.algo.name == "safe_explorer_ppo":
        from safe_control_gym.controllers.safe_explorer.safe_ppo import SafeExplorerPPO

        agent_cls = SafeExplorerPPO

    elif cfg.algo.name == "ppo":
        from safe_control_gym.controllers.ppo.ppo import PPO

        agent_cls = PPO
    
    else:
        raise ValueError("unknown agent name")
    
    agent = agent_cls(
            env_func=env_func,
            training=True,
            checkpoint_path=os.path.join(cfg.experiment.checkpoint_dir, "model_latest.pt"),
            output_dir=cfg.experiment.output_dir,
            seed=cfg.experiment.seed,
            device=cfg.device,
            **cfg.algo.config,
        )

    agent.reset()
    agent.learn()


if __name__ == "__main__":
    # pretraining : python train_safe_exp.py +algo=safe_exp_ppo_pretrain +task=simple_pendulum experiment.tag=pretrain
    # train : python train_safe_exp.py +algo=safe_exp_ppo_train +task=twolinkarm experiment.tag=train algo.config.pretrained=
    train()
