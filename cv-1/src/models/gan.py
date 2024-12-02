from torch import nn

class Generator(nn.Module):
    def __init__(self, num_channels=3, latent_dim=100, base_channels=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),  # (base_channels*8) x 4 x 4
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),  # (base_channels*4) x 8 x 8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            # (base_channels*2) x 16 x 16
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),  # (base_channels) x 32 x 32
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels, num_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # (num_channels) x 32 x 32
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, 4, 2, 1, bias=False),  # num_channels x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),  # (base_channels) x 32 x 32
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),  # (base_channels*2) x 16 x 16
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),  # (base_channels*4) x 8 x 8
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, 2, 2, 0, bias=False),  # (base_channels*8) x 4 x 4
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)
