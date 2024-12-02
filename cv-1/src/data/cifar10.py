import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR10DataModule:
    def __init__(self,
                 data_dir: str = './data',
                 batch_size: int = 64,
                 val_split: float = 0.2,
                 transform=None):
        """
        A reusable DataModule for the CIFAR-10 dataset.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Add normalization if needed
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Download and prepare datasets."""
        dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform)

        val_size = int(self.val_split * len(dataset))
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        """Return the training DataLoader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Return the test DataLoader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
