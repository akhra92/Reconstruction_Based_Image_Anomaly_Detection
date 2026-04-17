import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
TRAIN_VAL_SPLIT = [0.8, 0.2]

IN_CHANNELS = 1536
LATENT_DIM = 100

LEARNING_RATE = 0.001
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

CHECKPOINT_PATH = 'AE_ResNet50.pth'

SAMPLE_TEST_IMAGE = r'dataset/test/bad/20221017_T28_C3_S3.png'
