import torch
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter


class GeneralTrainer:
    def __init__(self, model, optimizer, loss_fn, device, train_loader, val_loader=None, test_loader=None,
                 log_dir="logs", checkpoint_dir="checkpoints", post_process_fn=None):
        """
        Initialize the trainer.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_fn
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.total_epochs = 0
        self.train_losses = []
        self.val_losses = []

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float('inf')

        self.post_process_fn = post_process_fn

        # Prepare fixed test images for visualization
        if self.test_loader is not None:
            test_images, _ = next(iter(test_loader))
            self.fixed_test_images = test_images[:8].to(self.device)
        else:
            self.fixed_test_images = None

    def train_epoch(self):
        """
        Perform one training epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        for images, _ in self.train_loader:
            images = images.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.loss_function(outputs, images)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        return epoch_loss / len(self.train_loader.dataset)

    def validate_epoch(self):
        """
        Perform one validation epoch.
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for images, _ in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.loss_function(outputs, images)
                epoch_loss += loss.item() * images.size(0)

        return epoch_loss / len(self.val_loader.dataset)

    def train(self, n_epochs):
        """
        Train the model for a specified number of epochs.
        """
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            self.total_epochs += 1

            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)

            # Log metrics
            self.writer.add_scalar("Loss/Train", train_loss, self.total_epochs)
            if val_loss is not None:
                self.writer.add_scalar("Loss/Validation", val_loss, self.total_epochs)

            print(
                f"Epoch {self.total_epochs}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}"
                if val_loss else f"Epoch {self.total_epochs}: Train Loss = {train_loss:.6f}")

            # Save best checkpoint
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

            # Log sample reconstructions
            if self.fixed_test_images is not None:
                self.log_reconstructions(self.total_epochs)

    def log_reconstructions(self, epoch):
        """
        Log reconstruction images to TensorBoard.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.fixed_test_images)
            if self.post_process_fn:
                outputs = self.post_process_fn(outputs)
            recon_batch = outputs.detach().cpu()
        grid = torch.cat([self.fixed_test_images.cpu(), recon_batch], dim=0)
        self.writer.add_images(f"Reconstructions/Epoch_{epoch}", grid, epoch)

    def plot_losses(self):
        """
        Plot training and validation losses.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.total_epochs + 1), self.train_losses, label="Training Loss")
        if self.val_losses:
            plt.plot(range(1, self.total_epochs + 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()
