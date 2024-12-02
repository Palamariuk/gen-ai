from torch import nn
import matplotlib.pyplot as plt
import numpy as np


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


def visualize_reconstructions(model, test_loader, device, num_images=10):
    # Obtain one batch of test images
    dataiter = iter(test_loader)
    images, _ = next(dataiter)  # Ignore labels if not needed

    images = images.to(device)

    # Generate model outputs
    output = model(images)

    # Prepare images and outputs for display
    images = images.cpu().numpy().transpose((0, 2, 3, 1))  # Convert to HWC format
    output = output.view(output.size(0), 3, 32, 32)  # Reshape to original dimensions
    output = output.detach().cpu().numpy().transpose((0, 2, 3, 1))  # Convert to HWC format

    # Plot the input images and their reconstructions
    fig, axes = plt.subplots(nrows=2, ncols=num_images, sharex=True, sharey=True, figsize=(25, 5))

    # Display input images in the first row and reconstructions in the second
    for img_set, row in zip([images, output], axes):
        for img, ax in zip(img_set[:num_images], row):  # Limit to specified number of images
            ax.imshow(np.clip(img, 0, 1))  # Clip values to valid range
            ax.axis("off")  # Hide axis for cleaner visualization

    plt.tight_layout()
    plt.show()
