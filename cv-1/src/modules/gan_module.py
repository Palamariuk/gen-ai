from ..models.gan import *
from ..pipeline.base_model import BaseModel
from ..metrics.basic import *
from ..metrics.ssim_loss import SSIMLoss

import torchvision


class GANModule(BaseModel):
    def __init__(self, latent_dim, lr=2e-04, b1=0.5, b2=0.999, device='cpu'):
        super().__init__()
        self.generator = Generator(latent_dim=latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.to(self.device)

        self.validation_z = torch.randn(40, *(latent_dim, 1, 1), device=self.device)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return bce_loss(y_hat, y)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return {"generator": opt_g, "discriminator": opt_d}

    def training_step(self, batch, optimizers):
        # Extract optimizers
        optimizer_g = optimizers["generator"]
        optimizer_d = optimizers["discriminator"]

        # Prepare real data and labels
        real_data = batch[0].to(self.device)
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, device=self.device, dtype=torch.float)
        fake_labels = torch.zeros(batch_size, device=self.device, dtype=torch.float)

        # Discriminator training: Real data
        self.discriminator.zero_grad()
        real_output = self.discriminator(real_data)
        d_loss_real = self.adversarial_loss(real_output, real_labels)
        d_loss_real.backward()

        # Discriminator training: Fake data
        noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device, dtype=torch.float)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data.detach())  # Detach to prevent gradients from affecting the generator
        d_loss_fake = self.adversarial_loss(fake_output, fake_labels)
        d_loss_fake.backward()

        ssim = SSIMLoss()(real_data, fake_data)

        # Update the discriminator
        d_loss_total = d_loss_real + d_loss_fake
        optimizer_d.step()

        # Generator training
        self.generator.zero_grad()
        generator_output = self.discriminator(fake_data)  # No detach; gradients should flow back to the generator
        g_loss = self.adversarial_loss(generator_output, real_labels)  # Use real labels for generator loss
        g_loss.backward()

        # Update the generator
        optimizer_g.step()

        # Return losses and diagnostics
        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss_total.item(),
            "loss": g_loss.item() + d_loss_total.item(),
            "ssim": ssim.item(),
        }

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        z = torch.randn(imgs.shape[0], self.latent_dim, 1, 1, device=self.device)
        generated_imgs = self.generator(z)

        ssim = SSIMLoss()(imgs, generated_imgs)

        # Validation loss: how well discriminator identifies real vs fake
        valid = torch.ones(imgs.size(0), device=self.device)
        fake = torch.zeros(imgs.size(0), device=self.device)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(generated_imgs), fake)

        val_loss = (real_loss + fake_loss) / 2
        val_output = {"loss": val_loss.item(), "ssim": ssim.item()}

        if batch_idx == 0:
            sampled_imaged = torchvision.utils.make_grid(self.generator(self.validation_z), nrow=8)
            val_output.update({
                "sampling_img": sampled_imaged
            })

        return val_output