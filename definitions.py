import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(ROOT_DIR, 'saved_models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
TRAIN_LOGS_DIR = os.path.join(MODEL_DIR, 'training_logs')
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets', 'cached')

SEED = 42
