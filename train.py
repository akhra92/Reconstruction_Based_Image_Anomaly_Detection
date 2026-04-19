import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import os

from config import (device, IN_CHANNELS, LATENT_DIM, LEARNING_RATE,
                    SCHEDULER_PATIENCE, SCHEDULER_FACTOR, NUM_EPOCHS,
                    EARLY_STOPPING_PATIENCE, CHECKPOINT_PATH, ASSETS_DIR)
from dataset import get_dataloaders
from models import AutoEncoder, ResnetFeatures


def train():
    train_loader, val_loader = get_dataloaders()

    model = AutoEncoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(device)
    feat_extractor = ResnetFeatures().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    Train_Loss = []
    Validation_Loss = []
    best_val_loss = np.inf
    patience = EARLY_STOPPING_PATIENCE
    counter = 0

    num_epochs = NUM_EPOCHS
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for data, _ in train_loader:
            with torch.no_grad():
                features = feat_extractor(data.to(device))
            output = model(features)
            loss = criterion(output, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Train_Loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            num_batches = 0
            for data, _ in val_loader:
                features = feat_extractor(data.to(device))
                output = model(features)
                val_loss = criterion(output, features)
                val_loss_sum += val_loss.item()
                num_batches += 1
            val_loss_avg = val_loss_sum / num_batches

            scheduler.step(val_loss_avg)
            Validation_Loss.append(val_loss_avg)

        if epoch % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(
                epoch + 1, num_epochs, loss.item(), val_loss_avg))

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early Stopping!')
                break

    plot_learning_curves(Train_Loss, Validation_Loss)

    torch.save(model.state_dict(), CHECKPOINT_PATH)

    return model, feat_extractor, train_loader, val_loader


def plot_learning_curves(Train_Loss, Validation_Loss):
    plt.figure()
    plt.plot(Train_Loss, label='Training Loss')
    plt.plot(Validation_Loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, 'learning_curves.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def load_model(model, path=CHECKPOINT_PATH):
    ckpoints = torch.load(path, map_location=device)
    model.load_state_dict(ckpoints)
    model.eval()
    return model


if __name__ == '__main__':
    train()
