from ..models.vae import *
from ..pipeline.base_model import BaseModel
from ..metrics.basic import *
from ..metrics.ssim_loss import SSIMLoss

import torchvision


class VAEModule(BaseModel):
    def __init__(self, latent_dim=128, lr=1e-3, beta=0.1):
        super().__init__()
        self.vae_model = VariationalAutoEncoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta

        self.validation_z = torch.randn(40, latent_dim, device=self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vae_model.parameters(), lr=self.lr)
        return {"vae_optimizer": optimizer}

    @staticmethod
    def loss_function(outputs, targets, beta=0.1):
        recon_x, mu, logvar = outputs
        x = targets

        bce = bce_loss(recon_x.view(-1, 3 * 32 * 32), x.view(-1, 3 * 32 * 32))
        kld = kld_loss(mu, logvar)
        total_loss = bce + beta * kld
        return total_loss, bce, kld

    def training_step(self, batch, optimizers):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        optimizer = optimizers["vae_optimizer"]

        # Forward pass
        outputs = self.vae_model(imgs)
        loss, bce, kld = self.loss_function(outputs, imgs, beta=self.beta)

        ssim = SSIMLoss()(imgs, outputs[0])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item(), "bce": bce.item(), "kld": kld.item(), "ssim": ssim.item()}

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        # Forward pass
        outputs = self.vae_model(imgs)
        loss, bce, kld = self.loss_function(outputs, imgs, beta=self.beta)

        ssim = SSIMLoss()(imgs, outputs[0])

        val_output = {"loss": loss.item(), "bce": bce.item(), "kld": kld.item(), "ssim": ssim.item()}

        if batch_idx == 0:
            reconstructed_images = torchvision.utils.make_grid(torch.cat([imgs[:8], outputs[0][:8]], dim=0), nrow=8)
            sampled_imaged = torchvision.utils.make_grid(self.vae_model.decode(self.validation_z))
            val_output.update({
                "reconstruction_img": reconstructed_images,
                "sampling_img": sampled_imaged
            })

        return val_output
