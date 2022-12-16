import logging
import os
import sys
sys.path.append(os.getcwd())
# os.environ["PYTHONNOUSERSITE"] = "1"
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from training_engine.logger import build_logger
from training_engine.utils import update_config_for_training, build_training_experiment_folder

from training_engine.training_engine import TrainingEngine, build_training_engine
import multiprocessing as mp


logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# torch.multiprocessing.set_sharing_strategy('file_system')
# mp.set_start_method('spawn')

# If set, use the env. variable to overwrite the default dataset and experiment paths
CONFIG_PATH = 'config'
CONFIG_NAME = 'default_main'

torch.set_printoptions(sci_mode=False)

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.1")
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # print (cfg)

    if cfg.py_func == 'train':
        # Build training engine
        engine = build_training_engine(cfg)

        # Run training
        logger.info('Starting training...')
        engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine

    elif cfg.py_func == 'validate':
        engine = build_training_engine(cfg)

        # Run training
        logger.info('Starting evaluation...')
        engine.trainer.validate(model=engine.model, datamodule=engine.datamodule)
        return engine
    
    elif cfg.py_func == 'test':
        # Build training engine
        engine = build_training_engine(cfg)

        # Test model
        logger.info('Starting testing...')
        engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


if __name__ == '__main__':
    main()
