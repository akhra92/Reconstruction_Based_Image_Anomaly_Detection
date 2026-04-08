import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import config


def train(model, feat_extractor, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.SCHEDULER_PATIENCE, factor=config.SCHEDULER_FACTOR
    )

    train_losses, val_losses = [], []
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in tqdm(range(config.NUM_EPOCHS)):
        # --- Training ---
        model.train()
        for data, _ in train_loader:
            features = feat_extractor(data.to(config.DEVICE))
            output = model(features)
            loss = criterion(output, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_loss_sum, num_batches = 0.0, 0
        with torch.no_grad():
            for data, _ in val_loader:
                features = feat_extractor(data.to(config.DEVICE))
                output = model(features)
                val_loss_sum += criterion(output, features).item()
                num_batches += 1
        val_loss_avg = val_loss_sum / num_batches
        val_losses.append(val_loss_avg)
        scheduler.step(val_loss_avg)

        if epoch % 5 == 0:
            print(f'Epoch [{epoch + 1}/{config.NUM_EPOCHS}]  '
                  f'Train Loss: {loss.item():.4f}  Val Loss: {val_loss_avg:.4f}')

        # --- Early stopping ---
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print('Early stopping triggered.')
                break

    return train_losses, val_losses


def plot_learning_curves(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.tight_layout()
    plt.show()
