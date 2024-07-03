from torchvision import transforms
import torch
from typing import List, Tuple, Union


def get_transform(img_size: int, mean: float, std: float) -> transforms.Compose:
    """
    Gets the transforms for the dataset

    Parameters
    ----------
    img_size: int
        The size of the image
    mean: float
        The mean of the dataset
    std: float
        The standard deviation of the dataset

    Returns
    -------
    transforms.Compose
        The transforms for the dataset
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    return transform

def split_dataset(dataset, split_size: float ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Splits a dataset into training and test sets.

    Parameters
    ----------
    dataset: tensor 
        The dataset to be split
    split_size: float
        The percentage of dataset for test set

    Returns
    -------
    A tuple consisting of training and test tensor datasets

    """
    train_size = int((1-split_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset