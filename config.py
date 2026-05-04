from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "processed_svg_data" / "tokenized"
META_PATH = PROJECT_ROOT / "processed_svg_data" / "tokenizer" / "meta.pkl"

OUTPUT_DIR = PROJECT_ROOT / "part2_outputs"

LR_SWEEP_DIR = OUTPUT_DIR / "lr_sweep"
ALL_MODELS_DIR = OUTPUT_DIR / "all_models"
LOSS_CURVES_DIR = ALL_MODELS_DIR / "loss_curves"

ANALYSIS_DIR = OUTPUT_DIR / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
TABLES_DIR = ANALYSIS_DIR / "tables"


# --------------------------------------------------
# Model Configs
# --------------------------------------------------

TINY_MODEL_CONFIG = {
    "model_name": "tiny",
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

SMALL_MODEL_CONFIG = {
    "model_name": "small",
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 192,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

MEDIUM_MODEL_CONFIG = {
    "model_name": "medium",
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

LARGE_MODEL_CONFIG = {
    "model_name": "large",
    "n_layer": 10,
    "n_head": 8,
    "n_embd": 512,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

XL_MODEL_CONFIG = {
    "model_name": "xl",
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

ALL_MODEL_CONFIGS = [
    TINY_MODEL_CONFIG,
    SMALL_MODEL_CONFIG,
    MEDIUM_MODEL_CONFIG,
    LARGE_MODEL_CONFIG,
    XL_MODEL_CONFIG,
]


# --------------------------------------------------
# Training Config
# --------------------------------------------------

TRAIN_CONFIG = {
    "batch_size": 8,
    "max_epochs": 1,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "warmup_ratio": 0.05,
    "eval_iters": 50,
    "device": "cuda",
    "seed": 42,
    "loss_log_interval": 50,
}