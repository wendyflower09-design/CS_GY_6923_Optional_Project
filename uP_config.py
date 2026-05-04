from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "processed_svg_data" / "tokenized"
META_PATH = PROJECT_ROOT / "processed_svg_data" / "tokenizer" / "meta.pkl"

STANDARD_PART2_DIR = PROJECT_ROOT / "part2_outputs"

OUTPUT_DIR = PROJECT_ROOT / "part3_uP_outputs"

UP_LR_SWEEP_DIR = OUTPUT_DIR / "uP_lr_sweep"
UP_ALL_MODELS_DIR = OUTPUT_DIR / "uP_all_models"
UP_LOSS_CURVES_DIR = UP_ALL_MODELS_DIR / "loss_curves"

UP_ANALYSIS_DIR = OUTPUT_DIR / "analysis"
UP_PLOTS_DIR = UP_ANALYSIS_DIR / "plots"
UP_TABLES_DIR = UP_ANALYSIS_DIR / "tables"


# --------------------------------------------------
# μP Model Configs
# --------------------------------------------------

UP_TINY_MODEL_CONFIG = {
    "model_name": "tiny_uP",
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

UP_SMALL_MODEL_CONFIG = {
    "model_name": "small_uP",
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 192,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

UP_MEDIUM_MODEL_CONFIG = {
    "model_name": "medium_uP",
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

UP_LARGE_MODEL_CONFIG = {
    "model_name": "large_uP",
    "n_layer": 10,
    "n_head": 8,
    "n_embd": 512,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

UP_XL_MODEL_CONFIG = {
    "model_name": "xl_uP",
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "block_size": 1024,
    "dropout": 0.1,
    "bias": True,
}

UP_ALL_MODEL_CONFIGS = [
    UP_TINY_MODEL_CONFIG,
    UP_SMALL_MODEL_CONFIG,
    UP_MEDIUM_MODEL_CONFIG,
    UP_LARGE_MODEL_CONFIG,
    UP_XL_MODEL_CONFIG,
]


# --------------------------------------------------
# Training Config
# --------------------------------------------------

UP_TRAIN_CONFIG = {
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