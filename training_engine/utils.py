import logging
import math
import os
from pathlib import Path
from shutil import rmtree
from hydra._internal.utils import _locate
from typing import cast, Any, Callable, Dict, Type, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def is_target_type(cfg: DictConfig, target_type: Union[Type[Any], Callable[..., Any]]) -> bool:
    """
    Check whether the config's resolved type matches the target type or callable.
    :param cfg: config
    :param target_type: Type or callable to check against.
    :return: Whether cfg._target_ matches the target_type.
    """
    return bool(_locate(cfg._target_) == target_type)

def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)
    # print (cfg)
    if cfg.trainer.overfitting.enable:
        cfg.data_module.num_workers = 0
    
    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(f'Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config...')
        logger.info('\n' + OmegaConf.to_yaml(cfg))



def get_num_gpus_used(cfg: DictConfig) -> int:
    """
    Gets the number of gpus used in ddp by searching through the environment variable WORLD_SIZE, PytorchLightning Trainer specified number of GPUs, and torch.cuda.device_count() in that order.
    :param cfg: Config with experiment parameters.
    :return num_gpus: Number of gpus used in ddp.
    """
    num_gpus = os.getenv('WORLD_SIZE', -1)

    if num_gpus == -1:  # if environment variable WORLD_SIZE is not set, find from trainer
        logger.info('WORLD_SIZE was not set.')
        trainer_num_gpus = cfg.lightning.trainer.params.gpus

        if isinstance(num_gpus, str):
            raise RuntimeError('Error, please specify gpus as integer. Received string.')
        trainer_num_gpus = cast(int, trainer_num_gpus)

        if trainer_num_gpus == -1:  # if trainer gpus = -1, all gpus are used, so find all available devices
            logger.info(
                'PytorchLightning Trainer gpus was set to -1, finding number of GPUs used from torch.cuda.device_count().'
            )
            cuda_num_gpus = torch.cuda.device_count() * int(os.getenv('NUM_NODES', 1))
            num_gpus = cuda_num_gpus

        else:  # if trainer gpus is not -1
            logger.info(f'Trainer gpus was set to {trainer_num_gpus}, using this as the number of gpus.')
            num_gpus = trainer_num_gpus

    num_gpus = int(num_gpus)
    logger.info(f'Number of gpus found to be in use: {num_gpus}')
    return num_gpus



def build_training_experiment_folder(cfg: DictConfig) -> None:
    """
    Builds the main experiment folder for training.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    logger.info('Building experiment folders...')
    main_exp_folder = Path(cfg.output_dir)
    logger.info(f'Experimental folder: {main_exp_folder}')
    main_exp_folder.mkdir(parents=True, exist_ok=True)