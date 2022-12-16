from itertools import product
from typing import List, Dict, Optional, Any, Callable

import torch
from torchmetrics import Metric

from metric.utils import filter_and_get_best_mode


class MinFDE(Metric):
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False
    def __init__(self,
                 at: int = 1,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(MinFDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.at = at
        self.name = f"MinFDE_at{self.at}"
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        """
        pred:   torch.Tensor [N, K, T, 2]
        target: torch.Tensor [N, T, 2]
        """
        pred = pred[:, :self.at]

        pred, target, min_dist, best_mode = filter_and_get_best_mode(pred, target)

        self.sum += min_dist.sum()          
        self.count += min_dist.size(0) # number of agents
        # print (self.name, self.sum/self.count)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count