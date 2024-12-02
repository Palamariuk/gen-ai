import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Set random seeds for reproducibility
SEED_NUMPY = 71
np.random.seed(SEED_NUMPY)
torch.manual_seed(SEED_NUMPY)

# Reconstruction loss function
def recons_loss(recon_x, x):
    d = 1 if len(recon_x.shape) == 2 else (1,2,3)
    return torch.sum(F.mse_loss(recon_x, x, reduction='none'), dim=d)

# ConvVAE class with necessary methods
class ConvVAE(nn.Module):
    def init(self, h_dim=256, z_dim=128):
        super(ConvVAE, self).init()

        self.lr = 0.0005  # Adjusted learning rate

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4 * 4 * 128, h_dim)
        self.fc_bn1 = nn.BatchNorm1d(h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)  # mu
        self.fc22 = nn.Linear(h_dim, z_dim)  # log_var

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc_bn3 = nn.BatchNorm1d(h_dim)
        self.fc4 = nn.Linear(h_dim, 4 * 4 * 128)
        self.fc_bn4 = nn.BatchNorm1d(4 * 4 * 128)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def encode(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 4 * 4 * 128)
        x = self.leaky_relu(self.fc_bn1(self.fc1(x)))
        mu = self.fc21(x)
        log_var = self.fc22(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x = self.leaky_relu(self.fc_bn3(self.fc3(z)))
        x = self.leaky_relu(self.fc_bn4(self.fc4(x)))
        x = x.view(-1, 128, 4, 4)
        x = self.leaky_relu(self.bn4(self.deconv1(x)))
        x = self.leaky_relu(self.bn5(self.deconv2(x)))
        x = self.tanh(self.deconv3(x))  # Output activation adjusted for normalized data
        return x  # Output shape: [B, 3, 32, 32]

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        # Total loss with beta parameter
        loss = recon_loss + beta * kl_loss
        return loss, recon_loss, kl_loss

    def save_model(self, epoch, log_path):
        model_path = os.path.join(log_path, f'conv_vae_epoch_{epoch}.pth')
        torch.save(self.state_dict(), model_path)

# Training and validation functions for ConvVAE
def train_conv_vae(model, train_loader, val_loader, num_epochs, device, log_path):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    writer = SummaryWriter(log_dir=log_path)  # TensorBoard writer

    beta = 0.1  # Initial beta value for beta-VAE

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        progress_bar = tqdm(train_loader, desc=f"ConvVAE Training Epoch {epoch}/{num_epochs}")
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var, beta=beta)
            loss.backward()
            train_loss += loss.item() * data.size(0)
            train_recon_loss += recon_loss.item() * data.size(0)
            train_kl_loss += kl_loss.item() * data.size(0)
            optimizer.step()
            progress_bar.set_postfix({'Loss': loss.item(), 'Recon': recon_loss.item(), 'KL': kl_loss.item()})

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_recon_loss = train_recon_loss / len(train_loader.dataset)
        avg_train_kl_loss = train_kl_loss / len(train_loader.dataset)

        # Validation
        val_loss, val_recon_loss, val_kl_loss = validate_conv_vae(model, val_loader, device, beta)
        print(f'Epoch [{epoch}/{num_epochs}], '
              f'Training Loss: {avg_train_loss:.6f}, Recon Loss: {avg_train_recon_loss:.6f}, KL Loss: {avg_train_kl_loss:.6f}, '
              f'Validation Loss: {val_loss:.6f}, Recon Loss: {val_recon_loss:.6f}, KL Loss: {val_kl_loss:.6f}')

        # TensorBoard logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Recon_Loss/Train', avg_train_recon_loss, epoch)
        writer.add_scalar('Recon_Loss/Validation', val_recon_loss, epoch)
        writer.add_scalar('KL_Loss/Train', avg_train_kl_loss, epoch)
        writer.add_scalar('KL_Loss/Validation', val_kl_loss, epoch)

        # Adjust beta (optional)
        beta = min(1.0, beta + 0.01)  # Gradually increase beta to 1.0

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Save the best model and reconstructed images
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(epoch, log_path)
            print(f'Best model saved with validation loss: {best_val_loss:.6f}')
            save_reconstructed_images(model, val_loader, device, epoch, log_path, writer)

        # Save reconstructed images after every epoch
        save_reconstructed_images(model, val_loader, device, epoch, log_path, writer)

    writer.close()

def validate_conv_vae(model, val_loader, device, beta):
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var, beta=beta)
            val_loss += loss.item() * data.size(0)
            val_recon_loss += recon_loss.item() * data.size(0)
            val_kl_loss += kl_loss.item() * data.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_recon_loss = val_recon_loss / len(val_loader.dataset)
    avg_val_kl_loss = val_kl_loss / len(val_loader.dataset)
    return avg_val_loss, avg_val_recon_loss, avg_val_kl_loss

def save_reconstructed_images(model, val_loader, device, epoch, log_path, writer):
    os.makedirs(log_path, exist_ok=True)  # Ensure the directory exists
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(val_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)
        num_images = 20
        orig_images = data[:num_images].cpu()
        recon_images = recon_batch[:num_images].cpu()
        # Denormalize images for visualization
        orig_images = orig_images * 0.5 + 0.5  # Assuming normalization mean and std of 0.5
        recon_images = recon_images * 0.5 + 0.5
        # Concatenate images side by side
        comparison = torch.cat([orig_images, recon_images], dim=3)  # Concatenate along width
        # Make a grid of images
        img_grid = torchvision.utils.make_grid(comparison, nrow=4, pad_value=1)
        # Save the grid
        img_path = os.path.join(log_path, f'reconstructions_epoch_{epoch}.png')
        torchvision.utils.save_image(img_grid, img_path)
        print(f'Saved reconstructed images to {img_path}')
        # TensorBoard logging
        writer.add_image('Reconstructed_Images', img_grid, epoch)

def get_dataloaders(batch_size=64, validation_split=0.2, shuffle=True, num_workers=2):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    # Download and load the dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    # Split the dataset into training and validation sets
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(SEED_NUMPY)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    return train_loader, val_loader

# Main execution
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Network configuration
    network_cfg = {
        'img_shape': [32, 32, 3],  # Height, Width, Channels
        'latent_dim': 128,
        'log_path': './conv_vae_logs/',
        'log_overwrite_save': True,
        'learning_rate': 0.0005,
    }

    # Get data loaders
    batch_size = 64
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    print(1)

    # Instantiate and train ConvVAE
    conv_vae_net = ConvVAE(h_dim=256, z_dim=network_cfg['latent_dim'])
    conv_vae_net.lr = network_cfg['learning_rate']
    num_epochs_conv_vae = 60
    log_path = network_cfg['log_path']

    print(2)

    # Handle log directory
    if network_cfg.get('log_overwrite_save', False):
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path, exist_ok=True)

    train_conv_vae(conv_vae_net, train_loader, val_loader, num_epochs_conv_vae, device, log_path)

if __name__ == 'main':
    main()