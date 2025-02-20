import torch


class DiffusionProcess:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.model = model
        self.T = T
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)  # Linear schedule
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)



