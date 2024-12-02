from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # B, 16, 16, 16
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # B, 32, 8, 8
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7)  # B, 64, 2, 2
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # B, 32, 8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # B,16,16,16
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # B, 3,32,32
            nn.Sigmoid()  # To bring outputs between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
