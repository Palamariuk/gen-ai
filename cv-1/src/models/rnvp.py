import torch
import torch.nn as nn


# Define the scale and translation networks for coupling layers
class ScaleTranslateNet(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(ScaleTranslateNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# Define a coupling layer
class CouplingLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, mask_type):
        super(CouplingLayer, self).__init__()
        self.mask_type = mask_type
        self.scale_net = ScaleTranslateNet(in_channels, hidden_channels)
        self.translate_net = ScaleTranslateNet(in_channels, hidden_channels)

    def forward(self, x, reverse=False):
        B, C, H, W = x.size()
        mask = self._get_mask(B, C, H, W, x.device)

        x_masked = x * mask
        scale = self.scale_net(x_masked) * (1 - mask)
        translate = self.translate_net(x_masked) * (1 - mask)

        if reverse:
            x = (x - translate) * torch.exp(-scale)
        else:
            x = x * torch.exp(scale) + translate
        return x, scale.sum(dim=(1, 2, 3))

    def _get_mask(self, B, C, H, W, device):
        mask = torch.zeros(B, C, H, W, device=device)
        mask[:, :, ::2, ::2] = 1 if self.mask_type == 'checkerboard0' else 0
        mask[:, :, 1::2, 1::2] = 1 if self.mask_type == 'checkerboard0' else 0
        return mask


# Define the full RealNVP model
class RealNVP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask_type = 'checkerboard0' if i % 2 == 0 else 'checkerboard1'
            self.layers.append(CouplingLayer(in_channels, hidden_channels, mask_type))

    def forward(self, x, reverse=False):
        log_det_jacobian = 0
        if not reverse:
            for layer in self.layers:
                x, ldj = layer(x, reverse)
                log_det_jacobian += ldj
        else:
            for layer in reversed(self.layers):
                x, ldj = layer(x, reverse)
                log_det_jacobian += ldj
        return x, log_det_jacobian
