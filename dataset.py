import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

import config


def get_transform():
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
    ])


def get_dataloaders():
    transform = get_transform()
    dataset = ImageFolder(root=config.TRAIN_DATA_PATH, transform=transform)
    trn_dataset, val_dataset = random_split(dataset, [1 - config.VAL_SPLIT, config.VAL_SPLIT])

    train_loader = DataLoader(trn_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader
