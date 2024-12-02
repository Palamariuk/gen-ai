import torch
import matplotlib.pyplot as plt
import numpy as np


def sample(model, num_samples, shape, device='cpu', show=False):
    with torch.no_grad():
        z = torch.randn(num_samples, *shape, device=device)
        samples = model(z)

    num_rows = num_samples // 10

    if show:
        fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
        for j in range(num_rows):
            for i in range(10):
                axes[j, i].imshow(samples[j * 10 + i].permute(1, 2, 0).cpu().numpy())
                axes[j, i].axis("off")
        plt.tight_layout()
        plt.show()

    return samples


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
