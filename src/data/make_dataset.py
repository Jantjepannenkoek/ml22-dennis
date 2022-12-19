import gin
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Tuple
from torchvision import datasets
from torchvision.transforms import ToTensor

Tensor = torch.Tensor


@gin.configurable
def get_MNIST(  # noqa: N802
    data_dir: Path, batch_size: int
) -> Tuple[DataLoader, DataLoader]:

    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader