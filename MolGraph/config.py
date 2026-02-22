# ============================================================
# MolGraph Configuration File
# Central config for all hyperparameters and settings
# ============================================================

import os

# ─────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DATA_DIR   = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

# ─────────────────────────────────────────
# DATASET CONFIG
# ─────────────────────────────────────────
DATASET_NAME    = "BBBP"        # Options: "BBBP", "HIV", "ESOL", "FreeSolv"
TASK_TYPE       = "classification"  # "classification" or "regression"
RANDOM_SEED     = 42
TRAIN_RATIO     = 0.8
VAL_RATIO       = 0.1
TEST_RATIO      = 0.1

# ─────────────────────────────────────────
# MOLECULAR FEATURE CONFIG
# ─────────────────────────────────────────
ATOM_FEATURE_DIM   = 9    # Number of atom features
BOND_FEATURE_DIM   = 3    # Number of bond features

# Atom feature categories (for one-hot encoding)
ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I',
    'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Other'
]

HYBRIDIZATION_TYPES = [
    'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'
]

# ─────────────────────────────────────────
# MODEL ARCHITECTURE CONFIG
# ─────────────────────────────────────────
MODEL_TYPE      = "GAT"         # Options: "GAT", "GCN", "GIN", "MPNN"
IN_CHANNELS     = 79            # Total node feature dim after one-hot encoding
HIDDEN_CHANNELS = 256           # Hidden layer dimension
OUT_CHANNELS    = 128           # Final embedding dim
NUM_LAYERS      = 4             # Number of GNN layers
NUM_HEADS       = 8             # Attention heads (for GAT)
DROPOUT_RATE    = 0.2           # Dropout probability
NUM_CLASSES     = 1             # Output classes (1 = binary/regression)

# ─────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────
EPOCHS          = 100
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001
WEIGHT_DECAY    = 1e-4
LR_SCHEDULER    = "cosine"      # "cosine", "step", "plateau"
PATIENCE        = 20            # Early stopping patience
CLIP_GRAD_NORM  = 1.0           # Gradient clipping

# ─────────────────────────────────────────
# DEVICE CONFIG
# ─────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────
# LOGGING CONFIG
# ─────────────────────────────────────────
USE_WANDB       = False         # Set True if you have wandb account
WANDB_PROJECT   = "MolGraph"
WANDB_ENTITY    = "your_username"  # Replace with your wandb username
LOG_INTERVAL    = 10            # Log every N batches

# ─────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────
API_HOST        = "0.0.0.0"
API_PORT        = 8000
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
