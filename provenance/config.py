"""Training configuration for Granite 4.2-3B TopK SAEs.

The streaming training dataset revision and consumed document identities were not
recorded. TOKEN_BUDGET is a base-input pacing target; TOTAL_SAE_STEPS is the
integer activation-draw target before the training loop's 50-step granularity.
Local position zero is excluded from every packed sequence. The stored mean and
per-dimension std define inference normalization; see README.md for limitations.
"""
MODEL_NAME = "ibm-granite/granite-4.2-3b"
MODEL_REVISION = "4e9c275e232f10211ec88644d106a1f6a1b60e06"
SEED = 0

D_MODEL = 2560
N_LAYERS_TOTAL = 40
TARGET_LAYERS = [10, 20, 30]

D_SAE = 81920
K = 32
LR = 4e-4
WARMUP_STEPS = 1000
RESAMPLE_EVERY_STEPS = 2000
DEAD_THRESHOLD_STEPS = 2000

AUX_K = 512
AUX_LOSS_COEF = 1.0 / 32
AUX_DEAD_THRESHOLD_STEPS = 500

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SEQ_LEN = 512
TOKEN_BUDGET = 200_000_000
FORWARD_BATCH_SEQS = 16
SHUFFLE_BUFFER_TOKENS = 500_000
BUFFER_REFILL_FRACTION = 0.5

EXCLUDE_LOCAL_POSITION_0 = True

TRAIN_BATCH_SIZE = 2048

TOTAL_SAE_STEPS = TOKEN_BUDGET // TRAIN_BATCH_SIZE

WANDB_PROJECT = "granite-4.2-3b-sae"
LOG_EVERY_STEPS = 50
CHECKPOINT_EVERY_STEPS = 5000
CHECKPOINT_DIR = "./checkpoints"

BASE_MODEL_DTYPE = "bfloat16"
SAE_DTYPE = "float32"
