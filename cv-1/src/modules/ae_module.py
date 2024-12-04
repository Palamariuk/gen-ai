from ..models.ae import *
from ..pipeline.base_model import BaseModel
from ..metrics.basic import *
from ..metrics.ssim_loss import SSIMLoss

import torchvision


class AEModule(BaseModel):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.ae_model = AutoEncoder()
        self.lr = lr
        self.loss_function = mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.ae_model.parameters(), lr=self.lr)
        return {"ae_optimizer": optimizer}

    def training_step(self, batch, optimizers):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        optimizer = optimizers["ae_optimizer"]

        # Forward pass
        outputs = self.ae_model(imgs)

        loss = self.loss_function(outputs, imgs)
        ssim = SSIMLoss()(imgs, outputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item(), "ssim": ssim.item()}

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        # Forward pass
        outputs = self.ae_model(imgs)

        loss = self.loss_function(outputs, imgs)
        ssim = SSIMLoss()(imgs, outputs)
        val_output = {"loss": loss.item(), "ssim": ssim.item()}

        if batch_idx == 0:
            reconstructed_images = torchvision.utils.make_grid(torch.cat([imgs[:8], outputs[:8]], dim=0))
            val_output.update({
                "reconstruction_img": reconstructed_images,
            })

        return val_output
