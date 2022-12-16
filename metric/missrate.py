from typing import List, Dict, Optional, Any, Callable

import torch

from torchmetrics import Metric


class MissRate(Metric):
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False
    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(MissRate, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.name = f"missrate"
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    # TODO:
    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        """
        pred:   torch.Tensor [K, N, T, 2]
        target: torch.Tensor [N, T, 2]
        """
        pred = pred[:self.at, :, -1, :]
        target = target[:, -1, :]
        sums =  torch.norm(pred - target, p=2, dim=-1).sum(dim=-1)
        self.sum += sums[torch.argmin(sums)]
        self.count += target.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count