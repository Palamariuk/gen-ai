import torch
import torch.nn as nn
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy.linalg import sqrtm
import numpy as np

from torchmetrics.image.fid import FrechetInceptionDistance


def calculate_fid_custom(real_images, fake_images, device='cpu', batch_size=32):
    """
    Custom implementation that calculate FID score between two tensors.
    """
    # Load pretrained InceptionV3
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    def get_features(images):
        """Extract features using InceptionV3."""
        features = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(device)
                # Resize to InceptionV3 input size
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                batch_features = model(batch)
                features.append(batch_features)
        return torch.cat(features, dim=0)

    real_features = get_features(real_images)
    fake_features = get_features(fake_images)

    mu_real = real_features.mean(dim=0).cpu().numpy()
    mu_fake = fake_features.mean(dim=0).cpu().numpy()
    cov_real = torch.cov(real_features.T).cpu().numpy()
    cov_fake = torch.cov(fake_features.T).cpu().numpy()

    # Compute FID
    covmean = sqrtm(cov_real @ cov_fake)

    # Just for numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_real - mu_fake) ** 2) + np.trace(cov_real + cov_fake - 2 * covmean)

    return fid


def calculate_fid_torchmetrics(real_images, fake_images, device='cpu'):
    """
    Calculate FID using torchmetrics.
    """
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # Convert to uint8
    real_images_uint8 = (real_images * 255).clamp(0, 255).to(torch.uint8)
    fake_images_uint8 = (fake_images * 255).clamp(0, 255).to(torch.uint8)

    fid_metric.update(real_images_uint8, real=True)
    fid_metric.update(fake_images_uint8, real=False)
    return fid_metric.compute().item()


def calculate_fid(real_images, fake_images, device='cpu', impl='torchmetrics'):
    if impl == 'torchmetrics':
        return calculate_fid_torchmetrics(real_images, fake_images, device)
    elif impl == 'custom':
        return calculate_fid_custom(real_images, fake_images, device)
