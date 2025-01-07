from __future__ import annotations
import torch
from torch.utils.data import TensorDataset

DATA_PATH = "C:\\Users\\mathi\\OneDrive - Danmarks Tekniske Universitet\\K - 1. Semester\\Machine Learning Operationer\\dtu_mlops\\data\\corruptmnist_v1"


def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # Load the corrupted MNIST dataset
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f'{DATA_PATH}\\\\train_images_{i}.pt'))
        train_target.append(torch.load(f'{DATA_PATH}\\train_target_{i}.pt'))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)
    
    test_images: torch.Tensor = torch.load(f'{DATA_PATH}\\test_images.pt')
    test_target: torch.Tensor = torch.load(f'{DATA_PATH}\\test_target.pt')

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images, train_target)
    test_dataset = TensorDataset(test_images, test_target)

    return train_dataset, test_dataset