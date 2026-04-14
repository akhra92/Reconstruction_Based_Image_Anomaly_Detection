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

# Backbone fine-tuning
FINETUNE_LAYERS = ['layer3']  # ResNet layers to unfreeze; set to [] to disable fine-tuning
FINETUNE_LR = 1e-4            # learning rate for backbone params (lower than main LR)

# Anomaly scoring
TOP_K_PIXELS = 10           # number of top error pixels used in decision_function
BORDER_CROP = 3             # pixels to crop from each edge of the segmentation map
THRESHOLD_SIGMA = 3.0       # std multipliers for initial threshold estimate
HEATMAP_VMAX_SCALE = 10.0   # vmax = max_training_error * this factor
HEATMAP_SIZE = 128          # resize heatmap to (HEATMAP_SIZE x HEATMAP_SIZE)

# Data augmentation
AUG_ROTATION_DEGREES = 15
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.1
AUG_ERASING_PROB = 0.1
AUG_ERASING_SCALE = (0.02, 0.1)
