import math
import pickle
import random
import time

import numpy as np
import torch
from tqdm import tqdm

import mup

from best_model_train import load_best_learning_rate
from dataset import TokenDataset
from uP_config import DATA_DIR, META_PATH, UP_TRAIN_CONFIG, UP_XL_MODEL_CONFIG
from uP_model import build_up_model_with_base_shapes


# --------------------------------------------------
# Memory Test Config
# --------------------------------------------------

TEST_STEPS = 100


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, max_steps: int, learning_rate: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(1.0, max(0.0, decay_ratio))

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * coeff


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    train_config = dict(UP_TRAIN_CONFIG)
    device = train_config["device"]

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")

    set_seed(train_config["seed"])
    learning_rate = load_best_learning_rate()

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    vocab_size = int(meta["vocab_size"])
    token_dtype = meta.get("token_dtype", "uint16")

    dataset = TokenDataset(
        data_dir=DATA_DIR,
        block_size=UP_XL_MODEL_CONFIG["block_size"],
        device=device,
        token_dtype=token_dtype,
    )

    model = build_up_model_with_base_shapes(
        model_config_dict=UP_XL_MODEL_CONFIG,
        vocab_size=vocab_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = mup.MuAdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(train_config["beta1"], train_config["beta2"]),
        weight_decay=train_config["weight_decay"],
    )

    batch_size = train_config["batch_size"]
    block_size = UP_XL_MODEL_CONFIG["block_size"]
    warmup_steps = max(1, int(TEST_STEPS * train_config["warmup_ratio"]))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("\n-----------------------------------")
    print("XL μP Memory Stability Test")
    print("-----------------------------------")
    print(f"Model: {UP_XL_MODEL_CONFIG['model_name']}")
    print("Parameterization: μP")
    print(f"Parameters: {total_params:,}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Learning rate: {learning_rate}")
    print("Optimizer: MuAdamW")
    print(f"Weight decay: {train_config['weight_decay']}")
    print(f"Betas: ({train_config['beta1']}, {train_config['beta2']})")
    print(f"Test steps: {TEST_STEPS}")
    print(f"GPU total memory: {gpu_total_memory:.2f} GB")

    model.train()

    start_time = time.time()

    progress_bar = tqdm(
        range(TEST_STEPS),
        desc="Testing XL",
        unit="step",
    )

    final_loss = None

    for step in progress_bar:
        lr = get_lr(
            step=step,
            max_steps=TEST_STEPS,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = dataset.get_batch("train", batch_size)

        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise ValueError(
                f"Training loss became NaN or Inf at step {step + 1}."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

        progress_bar.set_postfix(
            {
                "loss": f"{final_loss:.4f}",
                "lr": f"{lr:.2e}",
            }
        )

    training_time = time.time() - start_time

    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    allocated_percent = 100.0 * max_allocated / gpu_total_memory
    reserved_percent = 100.0 * max_reserved / gpu_total_memory

    print("\n-----------------------------------")
    print("Test Finished")
    print("-----------------------------------")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Training time: {training_time:.2f} sec")
    print(f"Max GPU allocated: {max_allocated:.2f} GB ({allocated_percent:.2f}%)")
    print(f"Max GPU reserved: {max_reserved:.2f} GB ({reserved_percent:.2f}%)")


if __name__ == "__main__":
    main()
