import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import TensorDataset


class LatentRectifiedFlowModule(pl.LightningModule):
    def __init__(self, model, autoencoder, latent_dim, lr=1e-3, mode="flow"):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.latent_dim = latent_dim
        self.lr = lr
        self.mode = mode.lower()
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def encode(self, x):
        mu, logvar = self.autoencoder.vae.encode(x)
        latents = self.autoencoder.vae.reparameterize(mu, logvar)  # (B, 64)
        latent_size = int(self.latent_dim ** 0.5)
        latents = latents.view(latents.size(0), 1, latent_size, latent_size)  # (B, 1, 8, 8)
        return latents

    @torch.no_grad()
    def decode(self, z):
        return self.autoencoder.vae.decode(z)

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        if self.mode == "flow":
            images, _ = batch
            batch_size = images.size(0)
            with torch.no_grad():
                x_target = self.encode(images)
            x0 = torch.randn_like(x_target)
        elif self.mode == "reflow":
            x0, x_target = batch
            batch_size = x0.size(0)
        else:
            raise ValueError("Invalid mode. Use 'flow' or 'reflow'.")

        t = torch.rand(batch_size, device=x_target.device)
        t_expanded = t.view(batch_size, *([1] * (x_target.dim() - 1)))  # e.g., (B, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x_target
        target_velocity = x_target - x0
        pred_velocity = self(x_t, t)
        loss = self.criterion(pred_velocity, target_velocity)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, x0, num_steps=100, image=False):
        dt = 1.0 / num_steps
        x = x0.clone()
        batch_size = x0.size(0)
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=x0.device)
            x = x + self.model(x, t) * dt

        if image:
            x = x.view(x.size(0), 64).to(self.device)
            x = self.decode(x)

        return x

    def generate_fixed_pairs(self, num_fixed=10000, batch_size=64, num_steps=100):
        self.model.eval()
        fixed_pairs = []
        device = next(self.model.parameters()).device
        latent_size = int(self.latent_dim ** 0.5)
        with torch.no_grad():
            for _ in range(num_fixed // batch_size):
                x0 = torch.randn(batch_size, 1, latent_size, latent_size, device=device)
                x_generated = self.sample(x0, num_steps=num_steps)
                fixed_pairs.append((x0.cpu(), x_generated.cpu()))
        x0_all = torch.cat([pair[0] for pair in fixed_pairs], dim=0)
        x_target_all = torch.cat([pair[1] for pair in fixed_pairs], dim=0)
        return TensorDataset(x0_all, x_target_all)