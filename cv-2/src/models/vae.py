import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # B, 16, 14, 14
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # B, 32, 7, 7
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7),  # B, 64, 1, 1
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Latent space representation
        self.fc_mu = nn.Linear(64 * 1 * 1, self.latent_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(64 * 1 * 1, self.latent_dim)  # Log variance of the latent space

        # Decoder
        self.fc_decode = nn.Linear(self.latent_dim, 64 * 1 * 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # B, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # B,16,16,16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),  # B, 1,32,32
            nn.Sigmoid()  # Outputs between 0 and 1
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 64, 1, 1)
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu, logvar):  # Reparameterization trick to sample z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
