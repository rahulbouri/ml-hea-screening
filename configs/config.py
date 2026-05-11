"""
Configuration file for ML-HEA Screening Project
Adjust these paths based on your local environment
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
DATASET_REGRESSION_CSV = DATA_DIR / "dataset_for_hardness_regression_language_model.csv"
MLM_PRETRAINING_CSV = DATA_DIR / "mlm_pretraining_dataset.csv"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
MATSCIBERT_WEIGHTS = MODELS_DIR / "matscibert_weights"
FINETUNED_MATSCIBERT_WEIGHTS = MODELS_DIR / "finetuned_matscibert_weights"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Create directories if they don't exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-trained MatSci BERT model from Hugging Face Hub
MATSCIBERT_HF_ID = "m3rg-iitd/matscibert"

# Training configuration
RANDOM_SEED = 42
DEVICE = "cuda"  # Change to "cpu" if GPU not available
BATCH_SIZE = 1   # For small datasets, batch size 1 is used in original work
VAL_SPLIT = 0.2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-2  # For frozen encoder fine-tuning
WEIGHT_DECAY = 1e-4

# Model architecture
EMBEDDING_DIM = 16
NUM_TRANSFORMER_LAYERS = 2
NUM_ATTENTION_HEADS = 2
FEEDFORWARD_DIM = 128

# Bayesian Optimization
BO_INIT_POINTS = 5
BO_N_ITER = 45

# Feature scaling
STANDARD_SCALER_PICKLE = CHECKPOINTS_DIR / "scaler.pkl"

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
