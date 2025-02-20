import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):
    def __init__(self, vae, lr=1e-3):
        super().__init__()
        self.vae = vae
        self.lr = lr
        self.save_hyperparameters(ignore=["vae_model"])

    def forward(self, x):
        return self.vae(x)

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def training_step(self, batch, batch_idx):
        images, _ = batch
        recon_images, mu, logvar = self.vae(images)
        loss = self.vae_loss(recon_images, images, mu, logvar)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        recon_images, mu, logvar = self.vae(images)
        loss = self.vae_loss(recon_images, images, mu, logvar)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr)
