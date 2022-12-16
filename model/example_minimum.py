from typing import List

import torch
import torch.nn as nn

from torchmetrics import Metric
from training_engine.lightning_module_wrapper import LightningModuleWrapper

class ExampleMinimumModel(LightningModuleWrapper):
    def __init__(self, 
                    metrics: List[Metric],
                    test_result_save_dir: str = "test_result" ,
                    **kwargs) -> None:
        super(ExampleMinimumModel, self).__init__(metrics=metrics, test_result_save_dir=test_result_save_dir)

        # define your model
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.Linear(10, 6*80*2)
        )

        # loss function you may use
        self.mse_loss = nn.MSELoss()

        # other parameters in kwargs
        self.lr = kwargs["lr"]

    def forward(self, features):
        """
        features: the (batched) output from your dataloader
        return: 
            prediction: torch.Tensor in [N, K, T, 2], 
                K: modalities, N: agents, T: timesteps (80 if 8s prediction), 2: x,y coords
            target: torch.Tensor in [N, T, 2],
            loss: a scalar loss
        """
        # print (features)
        # print (features["x"].size(), features["y"].size(), features["lane_vectors"].size())
        # print (features["lane_actor_index"].size(), features["lane_actor_vectors"].size())
        target = features["y"]
        # emulate passing some model and make predictions
        prediction = self.model(features["x"][:, -1]).reshape(-1, 6, 80, 2)
        mask = ~target.isnan()
        # emulate calculate some loss
        # we don't modularize loss functions for the best flexibility for the user
        if target.size(1) == 0: # if test, no target
            loss = 0
        else:
            loss = self.mse_loss(prediction[:, 0][mask], target[mask])
        # prediction is in shape of [N, K, T, 2], and target is in shape of [N, T, 2]
        # for calculating metrics
        # K is number of modes, and should be sorted by confidence (if have) descending
        # we will use prediction and target to calculate metrics
        # please keep the returned prediction and target as mentioned meaning and shape
        # and in the world coordinate system (i.e. the system used in orginal dataset files)
        # in order to get correct metrics
        return {"prediction": prediction,   # torch.Tensor [N, K, T, 2]
                "target": target,           # torch.Tensor
                "loss": loss,               # torch.Tensor
                "info": {}                  # Dict, optional, if you want to return something
            }

    def configure_optimizers(self):
        """
        define your optimizers and lr schedulers
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=32, eta_min=0.0)
        return [optimizer], [scheduler]