import torch
import matplotlib.pyplot as plt


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
