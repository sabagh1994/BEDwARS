import torch
import torch.nn as nn


# MATH:
def invlogit(x, eps):
    """The inverse of the logit function, 1 / (1 + exp(-x))."""
    return (1.0 - 2.0 * eps) / (1.0 + torch.exp(-x)) + eps


def softplus(x):
    return nn.functional.softplus(x)
