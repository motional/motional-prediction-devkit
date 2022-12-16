from dataclasses import dataclass
import datetime
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import cast, List, Optional, Any


import torch
from torchmetrics import Metric
import pytorch_lightning as pl

from callback.profile_callback import ProfileCallback
from training_engine.lightning_module_wrapper import LightningModuleWrapper




logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingEngine:
    """Lightning training engine dataclass wrapping the lightning trainer, model and datamodule."""

    trainer: pl.Trainer  # Trainer for models
    model: pl.LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: pl.LightningDataModule  # Loading data

    def __repr__(self) -> str:
        """
        :return: String representation of class without expanding the fields.
        """
        return f"<{type(self).__module__}.{type(self).__qualname__} object at {hex(id(self))}>"


def build_training_engine(cfg: DictConfig) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    # logger.info('Building training engine...')

    # Construct profiler
    profiler = ProfileCallback(Path(cfg.output_dir)) if cfg.enable_profiling else None

    # Start profiler if enabled
    if profiler:
        profiler.start_profiler("build_training_engine")

    # Build the datamodule
    datamodule = instantiate(cfg.datamodule)
        

    # Build model (lightning module)
    model = build_lightning_module(cfg)
    if cfg.checkpoint_to_validate is not None:
        model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_to_validate)


    # Build trainer
    trainer = build_trainer(cfg)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    # Save profiler output
    if profiler:
        profiler.save_profiler("build_training_engine")

    return engine


def build_lightning_module(cfg: DictConfig) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """

    # Build metrics to evaluate the performance of predictions
    metrics = build_metrics(cfg)
    # Create the complete Module
    model = instantiate(cfg.model, metrics=metrics, test_result_save_dir=cfg.test_result_save_dir)

    return cast(pl.LightningModule, model)


def build_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.trainer.params

    callbacks = build_callbacks(cfg)

    # plugins = [
        # pl.plugins.DDPPlugin(find_unused_parameters=False, num_nodes=params.num_nodes),
    # ]

    loggers = [
        pl.loggers.TensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version='',
            prefix='',
        ),
    ]

    # this feature is unchecked, please adjust for your use accordingly
    if cfg.trainer.overfitting.enable:
        OmegaConf.set_struct(cfg, False)
        params = OmegaConf.merge(params, cfg.trainer.overfitting.params)
        params.check_val_every_n_epoch = params.max_epochs + 1
        OmegaConf.set_struct(cfg, True)

        return pl.Trainer(**params)

    # this feature is unchecked, please adjust for your use accordingly
    if cfg.trainer.checkpoint.resume_training:
        # Resume training from latest checkpoint
        output_dir = Path(cfg.output_dir)
        date_format = cfg.date_format

        OmegaConf.set_struct(cfg, False)
        last_checkpoint = extract_last_checkpoint_from_experiment(output_dir, date_format)
        if not last_checkpoint:
            raise ValueError('Resume Training is enabled but no checkpoint was found!')

        params.resume_from_checkpoint = str(last_checkpoint)
        latest_epoch = torch.load(last_checkpoint)['epoch']
        params.max_epochs += latest_epoch
        logger.info(f'Resuming at epoch {latest_epoch} from checkpoint {last_checkpoint}')

        OmegaConf.set_struct(cfg, True)


    trainer = pl.Trainer(
        callbacks=callbacks,
        # plugins=plugins,
        logger=loggers,
        **params,
    )

    return trainer


def build_metrics(cfg: DictConfig) -> List[Metric]:
    """
    Build metrics based on config
    :param cfg: config
    :return list of metrics.
    """

    instantiated_metrics = []
    for metric_name, cfg_metric in cfg.metrics.items():
        new_metric: Metric = instantiate(cfg_metric)
        instantiated_metrics.append(new_metric)
    return instantiated_metrics


def build_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """
    Build callbacks based on config.
    :param cfg: Dict config.
    :return List of callbacks.
    """
    # logger.info('Building callbacks...')

    instantiated_callbacks = []

    for callback_type in cfg.callbacks.values():
        callback: pl.Callback = instantiate(callback_type)
        instantiated_callbacks.append(callback)

    if cfg.trainer.params.accelerator == 'gpu':
        instantiated_callbacks.append(pl.callbacks.DeviceStatsMonitor(cpu_stats=None))

    # logger.info('Building callbacks...DONE!')

    return instantiated_callbacks



def find_last_checkpoint_in_dir(group_dir: Path, experiment_time: Path) -> Optional[Path]:
    """
    Extract last checkpoint from a experiment
    :param group_dir: defined by ${group}/${experiment} from hydra
    :param experiment_time: date time which will be used as ${group}/${experiment}/${experiment_time}
    return checkpoint dir if existent, otherwise None
    """
    last_checkpoint_dir = group_dir / experiment_time / 'checkpoints'

    if not last_checkpoint_dir.exists():
        return None

    checkpoints = list(last_checkpoint_dir.iterdir())
    last_epoch = max(int(path.stem[6:]) for path in checkpoints if path.stem.startswith('epoch'))
    return last_checkpoint_dir / f'epoch_{last_epoch}.ckpt'


def extract_last_checkpoint_from_experiment(output_dir: Path, date_format: str) -> Optional[Path]:
    """
    Extract last checkpoint from latest experiment
    :param output_dir: of the current experiment, we assume that parent folder has previous experiments of the same type
    :param date_format: format time used for folders
    :return path to latest checkpoint, return None in case no checkpoint was found
    """
    date_times = [datetime.strptime(dir.name, date_format) for dir in output_dir.parent.iterdir() if dir != output_dir]
    date_times.sort(reverse=True)

    for date_time in date_times:
        checkpoint = find_last_checkpoint_in_dir(output_dir.parent, Path(date_time.strftime(date_format)))
        if checkpoint:
            return checkpoint
    return None