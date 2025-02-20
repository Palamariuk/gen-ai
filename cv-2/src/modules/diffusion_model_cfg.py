import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .diffusion_model import DiffusionModel


class DiffusionModelCFG(pl.LightningModule):
    def __init__(self, unet, T=1000, beta_start=1e-4, beta_end=0.02, lr=1e-3, cfg_scale=1.0):
        super().__init__()
        self.unet = unet
        self.T = T
        self.lr = lr
        self.cfg_scale = cfg_scale  # Default guidance scale

        beta = torch.linspace(beta_start, beta_end, T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

    def forward(self, x, t, cond=None, cfg_scale=None):
        """
        Forward pass with optional conditioning and classifier-free guidance.

        Args:
            x (Tensor): Input image tensor.
            t (Tensor): Timestep tensor.
            cond (optional): Conditioning information.
            cfg_scale (optional, float): Guidance scale. If None, defaults to self.cfg_scale.

        Returns:
            Tensor: Predicted noise.
        """
        if cond is not None and not torch.is_tensor(cond):
            # Convert condition to tensor on the same device as x.
            cond = torch.tensor(cond, device=x.device)

        # Use the default guidance scale if not provided.
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # If no condition is provided or guidance scale is 1 (i.e. no guidance), simply forward once.
        if cond is None or cfg_scale == 1.0:
            return self.unet(x, t, cond=cond)
        else:
            # Perform both unconditional and conditional predictions.
            uncond_pred = self.unet(x, t, cond=None)
            cond_pred = self.unet(x, t, cond=cond)
            # Combine predictions using classifier-free guidance.
            return uncond_pred + cfg_scale * (cond_pred - uncond_pred)

    def training_step(self, batch, batch_idx):
        # Assume batch is a tuple (x, cond) where "cond" contains the conditional info.
        x, cond = batch
        B = x.size(0)
        device = x.device

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x)
        x_t = self.add_noise(x, noise, t)

        # Use guidance scale = 1.0 during training to avoid applying CFG.
        pred_noise = self(x_t, t, cond=cond, cfg_scale=1.0)
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
    def sample_ddpm(self, batch_size, size, cond=None, cfg_scale=None):
        """Generates images using the standard DDPM reverse process with optional conditioning."""
        x_t = torch.randn((batch_size, *size), device=self.device)
        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(x_t) if t > 0 else 0  # No noise at t=0
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float)
            pred_noise = self(x_t, t_tensor, cond=cond, cfg_scale=cfg_scale)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * z

        return x_t.cpu()

    @torch.no_grad()
    def sample_ddim(self, batch_size, size, ddim_steps=50, cond=None, cfg_scale=None):
        """Generates images using DDIM with optional conditioning (fewer steps)."""
        x_t = torch.randn((batch_size, *size), device=self.device)
        step_size = self.T // ddim_steps

        for i in range(self.T - 1, 0, -step_size):
            t_tensor = torch.full((batch_size,), i, device=self.device, dtype=torch.float)
            pred_noise = self(x_t, t_tensor, cond=cond, cfg_scale=cfg_scale)
            alpha_bar_t = self.alpha_bar[i]
            alpha_bar_t_1 = self.alpha_bar[max(i - step_size, 0)]

            # DDIM update rule.
            x_t = (x_t - pred_noise * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)
            x_t = x_t * torch.sqrt(alpha_bar_t_1) + pred_noise * torch.sqrt(1 - alpha_bar_t_1)

        return x_t.cpu()