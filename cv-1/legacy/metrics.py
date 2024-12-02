import torch
from torch.nn.functional import conv2d, adaptive_avg_pool2d
import torch.nn.functional as F

from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import numpy as np




def get_features_inception(images, device="cpu"):
    """
    Feature extraction using the InceptionV3 model.
    """
    from torchvision.models import inception_v3
    import torch.nn.functional as F

    # Load pretrained InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # Extract features from the last pooling layer
    def feature_extractor_hook(module, input, output):
        return output

    # Register hook for the last pooling layer
    handle = model.Mixed_7c.register_forward_hook(feature_extractor_hook)

    with torch.no_grad():
        # Resize images to 299x299 as required by InceptionV3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Forward pass through the network
        features = model(images)

    # Remove the hook
    handle.remove()

    # Ensure features are pooled
    pooled_features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
    return pooled_features.squeeze(-1).squeeze(-1)  # Shape: (N, 2048)



def fid_score(real_features, gen_features):
    # Compute mean and covariance
    mu1, sigma1 = real_features.mean(dim=0), torch.cov(real_features.T)
    mu2, sigma2 = gen_features.mean(dim=0), torch.cov(gen_features.T)

    # Calculate the squared difference of means
    diff = mu1 - mu2
    mean_diff = diff.dot(diff)

    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # For numerical stability
    if not torch.isfinite(covmean).all():
        covmean = torch.zeros_like(sigma1)

    return mean_diff + torch.trace(sigma1 + sigma2 - 2 * covmean)


def calculate_fid(real_images, generated_images, device="cuda"):
    """
    Custom implementation of Frechet Inception Distance (FID).
    """
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)

    real_features = get_features_inception(real_images, device).cpu().numpy()
    generated_features = get_features_inception(generated_images, device).cpu().numpy()

    return fid_score(real_features, generated_features)


def inception_score(predictions, eps=1e-7):
    marginal = predictions.mean(dim=0)
    kl_div = predictions * (torch.log(predictions + eps) - torch.log(marginal + eps))
    return torch.exp(kl_div.sum(dim=1).mean())
