
import torch


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return torch.where(
        torch.abs(x) < delta,
        torch.pow(x, 2) * 0.5,
        delta * (torch.abs(x) - 0.5 * delta)
    )



