from typing import List, Dict, Optional, Any, Callable

import torch
from torchmetrics import Metric

from metric.utils import filter_and_get_best_mode


class MinADE(Metric):
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
        super(MinADE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.at = at
        self.name = f"MinADE_at{self.at}"
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        """
        pred:   torch.Tensor [N, K, T, 2]
        target: torch.Tensor [N, T, 2]
        """
        pred = pred[:, :self.at] # [N, k, T, 2] # top k for metric@k, k <= K

        # [n, k, T, 2], [n, T, 2], [n]
        # filter invalid agents (i.e. no ground truth) and get the best prediction mode
        pred, target, min_dist, best_mode = filter_and_get_best_mode(pred, target) 

        # make an [n, 1, T, 2] index
        best_mode_index = best_mode[..., None, None, None].repeat((1, 1, pred.size(2), 2))
        # [n, k, T, 2] -> [n, 1, T, 2] -> [n, T, 2]
        best_pred = pred.gather(1, best_mode_index).squeeze(1)
        diff = torch.norm(best_pred - target, p=2, dim=-1) # [n, T]
        
        if diff.size(0) == 0:
            return

        # mean over steps, keep agent dim for accurate averaging. Ignoreing nan values for non-all-none predictions
        metric = torch.cat([d[~d.isnan()].mean().unsqueeze(0) for d in diff], dim=0) # [n]

        self.sum += metric.sum()
        self.count += metric.size(0)
        # print (self.name, self.sum/self.count)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count