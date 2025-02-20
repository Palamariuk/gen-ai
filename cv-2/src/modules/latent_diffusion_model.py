import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .diffusion_model import DiffusionModel


class LatentDiffusionModel(DiffusionModel):
    def __init__(self, autoencoder, unet, latent_dim, T=1000, beta_start=1e-4, beta_end=0.02, lr=1e-4):
        super().__init__(unet, T, beta_start, beta_end, lr)
        self.autoencoder = autoencoder
        self.latent_dim = latent_dim

    @torch.no_grad()
    def encode(self, x):
        mu, logvar = self.autoencoder.vae.encode(x)
        latents = self.autoencoder.vae.reparameterize(mu, logvar)  # (BATCH, latend_dim=64) - 2d tensor
        latents = latents.view(latents.size(0), 1, 8, 8)  # (BATCH, 1, 8, 8) - 4d tensor
        return latents

    def decode(self, z):
        return self.autoencoder.vae.decode(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        B = x.size(0)
        device = x.device

        z = self.encode(x)

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(z)

        z_t = self.add_noise(z, noise, t)

        pred_noise = self.unet(z_t, t)

        loss = F.mse_loss(pred_noise, noise)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def sample_ddpm(self, batch_size, size=(1,8,8)):
        z_t = super().sample_ddpm(batch_size, size).to(self.device)
        z_t = z_t.view(z_t.size(0), 64).to(self.device)
        images = self.decode(z_t)
        return images.cpu()

    @torch.no_grad()
    def sample_ddim(self, batch_size, size=(1,8,8), ddim_steps=50):
        z_t = super().sample_ddim(batch_size, size, ddim_steps).to(self.device)
        z_t = z_t.view(z_t.size(0), 64).to(self.device)
        images = self.decode(z_t)
        return images.cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
