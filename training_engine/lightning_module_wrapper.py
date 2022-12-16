import logging
import os
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Metric

logger = logging.getLogger(__name__)


class LightningModuleWrapper(pl.LightningModule):
    """
    Lightning module that wraps the training/validation/testing procedure 
        and handles the metric computation.
    You will still need to implement your own forward() and configure_optimizers() 
        like originally using pl.LightningModule
    """

    def __init__(
        self,
        metrics: List[Metric],
        test_result_save_dir: str = "test_result"
    ) -> None:
        """
        Initializes the class.
        :param metrics: list of planning metrics computed at each step
        :param test_result_save_dir: used to save **test** split results. if you work on 
                train and val splits, you can still give it a value but it will not be used
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.metrics = nn.ModuleList(metrics)
        self.test_result_save_dir = test_result_save_dir
        if not os.path.exists(self.test_result_save_dir):
            os.makedirs(self.test_result_save_dir, exist_ok=True)
        


    def _step(self, batch, prefix: str) -> Dict:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: a result Dict consists keys of {"prediction", "targets", "loss", "metadata"} 
                and optionally more customized keys
        """

        # please make sure that the forward() method of your model (LightningModuleWrapper)
        # takes batch (including inputs and targets)
        # and returns predictions, targets, and loss
        # Note: in this form, you can do everthing you want within your model
        # you won't bother to create many different loss classes in the provided framework
        # just compute your loss in your preferred way and return it
        # predictions: torch.Tensor
        # targets: torch.Tensor
        # loss: a torch.Tensor scalar
        # metadata: a Dict
        results = self.forward(batch)
        predictions = results["prediction"]
        targets = results["target"]
        loss = results["loss"]
        info = results["info"]

        # if test, no need to compute metrics
        # we will compare the files
        if prefix == "test":
            return results
        
        self._compute_metrics(predictions, targets)

        self._log_step(loss, prefix, batch_size=targets.size(0))

        return results

    def _compute_metrics(self, predictions, targets) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        for metric in self.metrics:
            metric.update(predictions, targets)

    def _log_step(
        self,
        loss: torch.Tensor,
        prefix: str,
        batch_size: int = 1,
        loss_name: str = 'loss',
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f'loss/{prefix}_{loss_name}', loss, batch_size=batch_size)

        for metric in self.metrics:
            self.log(f"metrics/{prefix}_{metric.name}", metric, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        results = self._step(batch, 'train')
        return results['loss']

    def validation_step(
        self, batch, batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        with torch.no_grad():
            results = self._step(batch, 'val')
            return results['loss']

    def save_prediction_results(self, sample_name, agents_id, predictions):
        """
        :param: sample_name: name of the sample -- follow the name of the .parquet file in the dataset
        :param: agents_id: agents' id list, List[str], (N)
        :param: predictions: torch.Tensor [N, K, T, 2] 
                N is # agents, K is # modes, T is # timesteps, 2 coord dimension
        """
        result_dict = {id: predictions[i].detach().cpu() for i, id in enumerate(agents_id)} # each prediction is [K, T, 2]
        torch.save(result_dict, os.path.join(self.test_result_save_dir, f"{sample_name}.pt"))

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        with torch.no_grad():
            results = self._step(batch, 'test')
            # the current implementation requires some information from your input batch
            # if you changed the form of input batch, feel free to change below
            # optionally you can get them from results["info"] if you pass through them
            # feel free to change below if you change these information to your own representation
            # the goal is to generate result files: {sample_name}.pt per sample item in the test split
            # see self.save_prediction_results() for the actual format
            sample_names = batch["sample_name"] # filename without extension
            agents_count = batch["agents_count"] # used to check agent count with agent id
            agents_id = batch["agents_id"]

            # torch.Tensor [N, K, T, 2], N is # agents, K is # modes, T is # timesteps, 2 coord dimension
            prediction = results["prediction"] 
            accu_count = 0
            # for each individual sample, save a single result file contains results of all agents in that sample
            for name, count, ids in zip(sample_names, agents_count, agents_id):
                name = name[0]
                count = count[0]
                assert len(ids) == count
                self.save_prediction_results(name, ids, prediction[accu_count: accu_count+len(ids)])
                accu_count += len(ids)

            return results['loss']

