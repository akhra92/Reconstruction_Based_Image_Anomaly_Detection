import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split

import config


def get_train_transform():
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transform():
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
    ])


def get_dataloaders():
    full_dataset = ImageFolder(root=config.TRAIN_DATA_PATH, transform=None)
    n_total = len(full_dataset)
    n_val = int(n_total * config.VAL_SPLIT)
    n_train = n_total - n_val
    train_indices, val_indices = random_split(range(n_total), [n_train, n_val])

    train_dataset = ImageFolder(root=config.TRAIN_DATA_PATH, transform=get_train_transform())
    val_dataset = ImageFolder(root=config.TRAIN_DATA_PATH, transform=get_val_transform())

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader
