from jaxtyping import Float, Int
from torch import Tensor
import torch


def cross_entropy(logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[Tensor, "1"]:
    logits_shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_sum_exp: Float[Tensor, "batch_size"] = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))
    neg_log_probs: Float[Tensor, "batch_size"] = log_sum_exp - logits_shifted[torch.arange(logits.shape[0], device=logits.device), targets]
    return torch.mean(neg_log_probs, dim=0) 