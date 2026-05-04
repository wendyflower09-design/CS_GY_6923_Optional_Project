import csv
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import mup

from dataset import TokenDataset
from uP_config import (
    DATA_DIR,
    META_PATH,
    UP_LR_SWEEP_DIR,
    UP_XL_MODEL_CONFIG,
    UP_TRAIN_CONFIG,
)
from uP_model import build_up_model_with_base_shapes


# --------------------------------------------------
# Part 4 Best Model Training Config
# --------------------------------------------------

OUTPUT_DIR = Path("part4_outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAINING_DIR = OUTPUT_DIR / "training"

BEST_MODEL_NAME = "xl_uP_best"

DEFAULT_LEARNING_RATE = 0.01
MAX_EPOCHS = 5

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{BEST_MODEL_NAME}.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "xl_uP_last.pt"
LOSS_CURVE_PATH = TRAINING_DIR / f"{BEST_MODEL_NAME}_loss.csv"
VAL_CURVE_PATH = TRAINING_DIR / f"{BEST_MODEL_NAME}_val_loss.csv"
RESULTS_PATH = TRAINING_DIR / f"{BEST_MODEL_NAME}_results.json"
UP_LR_SWEEP_RESULTS_PATH = UP_LR_SWEEP_DIR / "uP_lr_sweep_results.json"

EVAL_INTERVAL_STEPS = 1000


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_best_learning_rate() -> float:
    if not UP_LR_SWEEP_RESULTS_PATH.exists():
        print(
            f"μP LR sweep results not found at {UP_LR_SWEEP_RESULTS_PATH}. "
            f"Using default learning rate {DEFAULT_LEARNING_RATE}."
        )
        return DEFAULT_LEARNING_RATE

    with open(UP_LR_SWEEP_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return float(data["best_lr"])


def get_lr(
    step: int,
    max_steps: int,
    learning_rate: float,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(1.0, max(0.0, decay_ratio))

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * coeff


@torch.no_grad()
def estimate_val_loss(model, dataset, batch_size, eval_iters):
    model.eval()

    losses = []

    for _ in range(eval_iters):
        x, y = dataset.get_batch("val", batch_size)
        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise ValueError("Validation loss became NaN or Inf.")

        losses.append(loss.item())

    model.train()

    return float(sum(losses) / len(losses))


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    model_config,
    train_config,
    vocab_size,
    token_dtype,
    learning_rate,
    epoch,
    step,
    train_loss,
    val_loss,
    total_params,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "train_config": train_config,
        "vocab_size": vocab_size,
        "token_dtype": token_dtype,
        "learning_rate": learning_rate,
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "total_params": total_params,
    }

    torch.save(checkpoint, path)


# --------------------------------------------------
# Main Training
# --------------------------------------------------

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    train_config = dict(UP_TRAIN_CONFIG)
    train_config["max_epochs"] = MAX_EPOCHS

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

    tokens_per_step = batch_size * block_size
    steps_per_epoch = len(dataset.train_data) // tokens_per_step
    max_steps = steps_per_epoch * train_config["max_epochs"]
    warmup_steps = max(1, int(max_steps * train_config["warmup_ratio"]))

    loss_log_interval = train_config.get("loss_log_interval", 50)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        gpu_total_memory = 0

    print("\n-----------------------------------")
    print("Part 4: Best Model Training")
    print("-----------------------------------")
    print(f"Model: {BEST_MODEL_NAME}")
    print(f"Parameterization: μP")
    print(f"Parameters: {total_params:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {MAX_EPOCHS}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Tokens per step: {tokens_per_step}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {max_steps}")
    print(f"Eval interval: {EVAL_INTERVAL_STEPS} steps")
    print(f"Best checkpoint path: {BEST_CHECKPOINT_PATH}")
    print(f"Last checkpoint path: {LAST_CHECKPOINT_PATH}")

    loss_records = []
    val_records = []

    best_val_loss = float("inf")
    best_step = None
    best_epoch = None

    model.train()
    start_time = time.time()

    final_train_loss = None
    final_val_loss = None

    progress_bar = tqdm(
        range(max_steps),
        desc="Training xl_uP best model",
        unit="step",
    )

    for step in progress_bar:
        current_step = step + 1
        current_epoch = current_step / steps_per_epoch

        lr = get_lr(
            step=step,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = dataset.get_batch("train", batch_size)

        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise ValueError(f"Training loss became NaN or Inf at step {current_step}.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_train_loss = loss.item()

        if current_step % loss_log_interval == 0 or current_step == max_steps:
            loss_records.append(
                {
                    "step": current_step,
                    "epoch": current_epoch,
                    "train_loss": final_train_loss,
                    "learning_rate": lr,
                }
            )

        if current_step % EVAL_INTERVAL_STEPS == 0 or current_step == max_steps:
            val_loss = estimate_val_loss(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                eval_iters=train_config["eval_iters"],
            )

            final_val_loss = val_loss

            val_records.append(
                {
                    "step": current_step,
                    "epoch": current_epoch,
                    "val_loss": val_loss,
                    "train_loss": final_train_loss,
                    "learning_rate": lr,
                }
            )

            print(
                f"\nEval step {current_step}/{max_steps} | "
                f"epoch {current_epoch:.3f} | "
                f"train_loss {final_train_loss:.4f} | "
                f"val_loss {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = current_step
                best_epoch = current_epoch

                save_checkpoint(
                    path=BEST_CHECKPOINT_PATH,
                    model=model,
                    optimizer=optimizer,
                    model_config=UP_XL_MODEL_CONFIG,
                    train_config=train_config,
                    vocab_size=vocab_size,
                    token_dtype=token_dtype,
                    learning_rate=learning_rate,
                    epoch=current_epoch,
                    step=current_step,
                    train_loss=final_train_loss,
                    val_loss=val_loss,
                    total_params=total_params,
                )

                print(
                    f"New best checkpoint saved | "
                    f"step {best_step} | "
                    f"epoch {best_epoch:.3f} | "
                    f"best_val_loss {best_val_loss:.4f}"
                )

        progress_bar.set_postfix(
            {
                "loss": f"{final_train_loss:.4f}",
                "lr": f"{lr:.2e}",
                "best_val": f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "N/A",
            }
        )

    training_time_seconds = round(time.time() - start_time, 2)

    if final_val_loss is None:
        final_val_loss = estimate_val_loss(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            eval_iters=train_config["eval_iters"],
        )

    save_checkpoint(
        path=LAST_CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        model_config=UP_XL_MODEL_CONFIG,
        train_config=train_config,
        vocab_size=vocab_size,
        token_dtype=token_dtype,
        learning_rate=learning_rate,
        epoch=MAX_EPOCHS,
        step=max_steps,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        total_params=total_params,
    )

    with open(LOSS_CURVE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "epoch", "train_loss", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(loss_records)

    with open(VAL_CURVE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "epoch", "val_loss", "train_loss", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(val_records)

    if device == "cuda":
        max_memory_allocated = torch.cuda.max_memory_allocated()
        max_memory_reserved = torch.cuda.max_memory_reserved()

        max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
        max_memory_reserved_gb = max_memory_reserved / (1024 ** 3)
        gpu_total_memory_gb = gpu_total_memory / (1024 ** 3)

        memory_allocated_percent = 100.0 * max_memory_allocated / gpu_total_memory
        memory_reserved_percent = 100.0 * max_memory_reserved / gpu_total_memory
    else:
        max_memory_allocated_gb = 0.0
        max_memory_reserved_gb = 0.0
        gpu_total_memory_gb = 0.0
        memory_allocated_percent = 0.0
        memory_reserved_percent = 0.0

    total_tokens_processed = max_steps * tokens_per_step
    tokens_per_second = total_tokens_processed / max(1e-8, training_time_seconds)

    results = {
        "model_name": BEST_MODEL_NAME,
        "parameterization": "uP",
        "learning_rate": learning_rate,
        "epochs": MAX_EPOCHS,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time_seconds,
        "max_gpu_memory_allocated_GB": round(max_memory_allocated_gb, 3),
        "max_gpu_memory_reserved_GB": round(max_memory_reserved_gb, 3),
        "gpu_total_memory_GB": round(gpu_total_memory_gb, 3),
        "memory_allocated_percent": round(memory_allocated_percent, 2),
        "memory_reserved_percent": round(memory_reserved_percent, 2),
        "tokens_per_second": round(tokens_per_second, 2),
        "total_params": total_params,
        "batch_size": batch_size,
        "block_size": block_size,
        "tokens_per_step": tokens_per_step,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
        "last_checkpoint_path": str(LAST_CHECKPOINT_PATH),
        "loss_curve_path": str(LOSS_CURVE_PATH),
        "val_curve_path": str(VAL_CURVE_PATH),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n-----------------------------------")
    print("Training Finished")
    print("-----------------------------------")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best step: {best_step}")
    print(f"Best epoch: {best_epoch:.3f}")
    print(f"Training time: {training_time_seconds:.2f} sec")
    print(
        f"Max GPU reserved: {results['max_gpu_memory_reserved_GB']} GB "
        f"({results['memory_reserved_percent']}%)"
    )
    print(f"Tokens/sec: {results['tokens_per_second']}")
    print(f"Best checkpoint saved to: {BEST_CHECKPOINT_PATH}")
    print(f"Last checkpoint saved to: {LAST_CHECKPOINT_PATH}")
    print(f"Loss curve saved to: {LOSS_CURVE_PATH}")
    print(f"Validation curve saved to: {VAL_CURVE_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
