import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import TensorDataset


class RectifiedFlowModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, mode="flow"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.mode = mode.lower()
        self.criterion = nn.MSELoss()

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        if self.mode == "flow":
            images, _ = batch
            batch_size = images.size(0)
            x0 = torch.randn_like(images)
            x_target = images
        elif self.mode == "reflow":
            x0, x_target = batch
            batch_size = x0.size(0)
        else:
            raise ValueError("Invalid mode. Use 'flow' or 'reflow'.")

        t = torch.rand(batch_size, device=x_target.device)
        t_expanded = t.view(batch_size, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x_target
        target_velocity = x_target - x0
        pred_velocity = self(x_t, t)
        loss = self.criterion(pred_velocity, target_velocity)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, x0, num_steps=100):
        dt = 1.0 / num_steps
        x = x0.clone()
        batch_size = x0.size(0)
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=x0.device)
            x = x + self.model(x, t) * dt
        return x

    def generate_fixed_pairs(self, num_fixed=10000, batch_size=64, num_steps=100):
        self.model.eval()
        fixed_pairs = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for _ in range(num_fixed // batch_size):
                x0 = torch.randn(batch_size, 1, 28, 28, device=device)
                x_generated = self.sample(x0, num_steps=num_steps)
                fixed_pairs.append((x0.cpu(), x_generated.cpu()))
        x0_all = torch.cat([pair[0] for pair in fixed_pairs], dim=0)
        x_target_all = torch.cat([pair[1] for pair in fixed_pairs], dim=0)
        return TensorDataset(x0_all, x_target_all)
