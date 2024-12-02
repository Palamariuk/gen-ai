import torch


def mse_loss(x, y):
    diff = x - y
    return (diff ** 2).mean()


def bce_loss(predictions, targets):
    eps = 1e-7  # To avoid log(0)
    predictions = torch.clamp(predictions, eps, 1 - eps)
    return -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions)).sum()


def kld_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()) / mu.size(0)
