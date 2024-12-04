from ..models.rnvp import *
from ..pipeline.base_model import BaseModel
from ..metrics.basic import *
from ..metrics.ssim_loss import SSIMLoss

import torchvision


class RNVPModule(BaseModel):
    def __init__(self, in_channels=3, hidden_channels=128, num_layers=8, lr=1e-3):
        super().__init__()
        self.rnvp_model = RealNVP(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lr = lr

        self.validation_z = torch.randn(40, 3, 32, 32, device=self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.rnvp_model.parameters(), lr=self.lr)
        return {"rnvp_optimizer": optimizer}

    @staticmethod
    def loss_function(outputs, targets):
        x, ldj = outputs
        loss = -ldj.mean() + (x ** 2).sum(dim=(1, 2, 3)).mean()
        return loss

    def training_step(self, batch, optimizers):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        optimizer = optimizers["rnvp_optimizer"]

        # Forward pass
        outputs = self.rnvp_model(imgs)
        loss = self.loss_function(outputs, imgs)

        ssim = SSIMLoss()(imgs, outputs[0])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item(), "ssim": ssim.item()}

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.to(self.device)

        # Forward pass
        outputs = self.rnvp_model(imgs)
        loss = self.loss_function(outputs, imgs)

        ssim = SSIMLoss()(imgs, outputs[0])

        val_output = {"loss": loss.item(), "ssim": ssim.item()}

        if batch_idx == 0:
            sampled_imaged = torchvision.utils.make_grid(self.rnvp_model.decode(self.validation_z))
            val_output.update({
                "sampling_img": sampled_imaged
            })

        return val_output
