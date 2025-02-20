import pytorch_lightning as pl

from ..models.unet import *


class DiffusionModel(pl.LightningModule):
    def __init__(self, unet, T=1000, beta_start=1e-4, beta_end=0.02, lr=1e-3):
        super().__init__()
        self.unet = unet
        self.T = T
        self.lr = lr

        beta = torch.linspace(beta_start, beta_end, T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

    def forward(self, x, t):
        return self.unet(x, t)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # (B,1,28,28) images in [-1,1]
        B = x.size(0)
        device = x.device

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x)
        x_t = self.add_noise(x, noise, t)

        pred_noise = self(x_t, t)
        loss = F.mse_loss(pred_noise, noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def add_noise(self, x, noise, t):
        B = x.size(0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(B, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

    @torch.no_grad()
    def sample_ddpm(self, batch_size, size):
        """ Generates images using standard DDPM (full reverse process). """
        x_t = torch.randn((batch_size, *size), device=self.device)

        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(x_t) if t > 0 else 0  # No noise at t=0
            t_tensor = torch.tensor([t], device=self.device).float()
            pred_noise = self.unet(x_t, t_tensor)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * z

        return x_t.cpu()

    @torch.no_grad()
    def sample_ddim(self, batch_size, size, ddim_steps=50):
        """ Generates images using DDIM (fewer steps). """
        x_t = torch.randn((batch_size, *size), device=self.device)
        step_size = self.T // ddim_steps  # Skip steps

        for i in range(self.T - 1, 0, -step_size):
            t = torch.tensor([i], device=self.device).float()
            pred_noise = self.unet(x_t, t)

            alpha_bar_t = self.alpha_bar[i]
            alpha_bar_t_1 = self.alpha_bar[max(i - step_size, 0)]

            # DDIM update rule
            x_t = (x_t - pred_noise * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
            x_t = x_t * alpha_bar_t_1.sqrt() + pred_noise * (1 - alpha_bar_t_1).sqrt()

        return x_t.cpu()