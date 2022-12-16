import torch

def filter_and_get_best_mode(pred: torch.Tensor, target: torch.Tensor):
    """
    get the best mode and it's 
    """
    final_timestamp = torch.argmax(target.isnan().any(dim=-1).int(), dim=-1) - 1 # [N]
    valid_agent = final_timestamp >= 0  # [N]
    pred = pred[valid_agent]            # [n, K, T, 2], n means after filter out invalid agents
    target = target[valid_agent]        # [n, T, 2]
    final_timestamp = final_timestamp[valid_agent] # [n]

    pred_terminal_index = final_timestamp[..., None, None, None].repeat((1, pred.size(1), 1, 2)) # make an [n,k,1,2] index
    # [n, K, T, 2] -> [n, k, T, 2] -> [n, k, 1, 2] -> [n, k, 2]
    pred_terminal = pred.gather(2, pred_terminal_index).squeeze(2)

    target_terminal_index = final_timestamp[..., None, None].repeat((1, 1, 2))
    # [n, T, 2] -> [n, 1, 2] -> [n, 2]
    target_terminal = target.gather(1, target_terminal_index).squeeze(1)

    # print (pred_terminal.size(), target_terminal.size())
    dist = torch.norm(pred_terminal.permute((1, 0, 2)) - target_terminal, p=2, dim=-1).permute((1, 0)) # [n, k]
    min_dist, best_mode = dist.min(-1) # min_dist, best_mode: [n]

    return pred, target, min_dist, best_mode