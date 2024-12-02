import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # B, 16, 16, 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # B, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7),  # B, 64, 2, 2
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Latent space representation
        self.fc_mu = nn.Linear(64 * 2 * 2, self.latent_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(64 * 2 * 2, self.latent_dim)  # Log variance of the latent space

        # Decoder
        self.fc_decode = nn.Linear(self.latent_dim, 64 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # B, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # B,16,16,16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # B, 3,32,32
            nn.Sigmoid()  # To bring outputs between 0 and 1
        )

    def encode(self, x):
        # Pass through encoder layers
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        # Return mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        # Decode latent variable z back to image space
        z = self.fc_decode(z)
        z = z.view(z.size(0), 64, 2, 2)  # Reshape back to image shape
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu, logvar):
        # Reparameterization trick to sample z
        std = torch.exp(0.5 * logvar)  # Standard deviation from logvar
        eps = torch.randn_like(std)  # Random noise
        z = mu + eps * std  # Reparameterized latent variable
        return z

    def forward(self, x):
        # Encode input to mean and log variance
        mu, logvar = self.encode(x)
        # Reparameterize to get latent variable z
        z = self.reparameterize(mu, logvar)
        # Decode latent variable to output
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device='cpu', show=False):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)

        if show:
            fig, axes = plt.subplots(1, 10, figsize=(20, 2))
            for i in range(10):
                axes[i].imshow(samples[i].permute(1, 2, 0).cpu().numpy())
                axes[i].axis("off")
            plt.tight_layout()
            plt.show()

        return samples


