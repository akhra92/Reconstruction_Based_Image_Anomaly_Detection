import torch

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# Dataset
TRAIN_DATA_PATH = 'dataset/train'
TEST_DATA_PATH = 'dataset/test'
IMAGE_SIZE = 224
BATCH_SIZE = 4
VAL_SPLIT = 0.2

# Model
IN_CHANNELS = 1536
LATENT_DIM = 100
IS_BN = True

# Training
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 5

# Checkpoints
MODEL_SAVE_PATH = 'AE_ResNet50.pth'

# Plots
PLOTS_DIR = 'plots'
