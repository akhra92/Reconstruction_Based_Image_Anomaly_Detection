import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from config import TRAIN_DIR, IMAGE_SIZE, BATCH_SIZE, TRAIN_VAL_SPLIT


transform = T.Compose([T.Resize(IMAGE_SIZE),
                       T.ToTensor()])


def get_dataloaders(root=TRAIN_DIR, batch_size=BATCH_SIZE, split=TRAIN_VAL_SPLIT):
    dataset = ImageFolder(root=root, transform=transform)
    trn_dataset, val_dataset = random_split(dataset, split)

    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
