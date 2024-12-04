import torch
import torch.nn.functional as F
from torch import nn

class SSIMLoss(nn.Module):
    # Such default parameters to be similar to PIQ implementation
    def __init__(self, window_size: int = 11, sigma: float = 1.5, max_val: float = 1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.max_val = max_val
        self.window = self._create_gaussian_window(window_size, sigma)

    @staticmethod
    def _create_gaussian_window(size, sigma):
        """Create a Gaussian window for convolution."""
        coords = torch.arange(size).float() - size // 2
        gauss = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
        gauss /= gauss.sum()
        window = gauss[:, None] * gauss[None, :]
        return window.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)

    def forward(self, x, y):
        """Calculate SSIM loss between two images."""
        device = x.device
        batch, channel, height, width = x.shape
        window = self.window.to(device).repeat(channel, 1, 1, 1)

        pad = self.window_size // 2
        mu1 = F.conv2d(x, window, padding=pad, groups=channel)
        mu2 = F.conv2d(y, window, padding=pad, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(x ** 2, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y ** 2, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=pad, groups=channel) - mu1_mu2

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim.mean([1, 2, 3])

        return 1 - ssim.mean()