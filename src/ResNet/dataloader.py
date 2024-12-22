import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


# Custom target_transform to switch labels 0 and 1
def label_switch(target):
    return 1 - target


# Calculate mean and standard deviation of a dataset
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        num_samples += batch_samples
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= num_samples
    std /= num_samples
    mean.tolist(), std.tolist()
    return mean.tolist(), std.tolist()


# Data loader with mean/std computation
def data_loader(
    data_dir,
    batch_size,
    random_seed=42,
    valid_size=0.2,
    shuffle=True,
    test=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    img_shape=(224, 224),
):
    # Define base transform (no normalization yet)
    base_transform = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
        ]
    )

    # Load the dataset (train or test based on `test` flag)
    dataset_path = "test" if test else "train"

    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, dataset_path),
        transform=base_transform,
        target_transform=label_switch,
    )

    # Compute mean and std for the dataset
    mean, std = calculate_mean_std(dataset)

    # Define normalize transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Update transform with normalization
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Update dataset's transform with normalization
    dataset.transform = transform

    if test:
        # Test DataLoader
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        return test_loader

    # Split dataset into training and validation sets
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Train DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    # Validation DataLoader
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, valid_loader
