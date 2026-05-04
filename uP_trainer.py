import csv
import math
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

import mup

from dataset import TokenDataset
from uP_model import build_up_model_with_base_shapes, UPTransformerLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
def estimate_val_loss(
    model: UPTransformerLM,
    dataset: TokenDataset,
    batch_size: int,
    eval_iters: int,
) -> float:
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


def train_one_up_model(
    model_config_dict: Dict[str, Any],
    train_config: Dict[str, Any],
    learning_rate: float,
    data_dir: Path,
    meta_path: Path,
    loss_curve_path: Optional[Path] = None,
) -> Dict[str, Any]:
    set_seed(train_config["seed"])

    device = train_config["device"]

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    vocab_size = int(meta["vocab_size"])
    token_dtype = meta.get("token_dtype", "uint16")

    dataset = TokenDataset(
        data_dir=data_dir,
        block_size=model_config_dict["block_size"],
        device=device,
        token_dtype=token_dtype,
    )

    model = build_up_model_with_base_shapes(
        model_config_dict=model_config_dict,
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
    tokens_per_step = batch_size * model_config_dict["block_size"]

    steps_per_epoch = len(dataset.train_data) // tokens_per_step
    max_steps = steps_per_epoch * train_config["max_epochs"]

    if max_steps <= 0:
        raise ValueError(
            "max_steps is 0. Check train.bin size, batch_size, and block_size."
        )

    warmup_steps = max(1, int(max_steps * train_config["warmup_ratio"]))
    loss_log_interval = train_config.get("loss_log_interval", 50)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        gpu_total_memory = 0

    print(f"\nTraining μP model: {model_config_dict['model_name']}")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {model_config_dict['block_size']}")
    print(f"Tokens per step: {tokens_per_step}")
    print(f"Parameters: {total_params:,}")

    loss_curve_records = []

    start_time = time.time()

    final_train_loss = None

    model.train()

    progress_bar = tqdm(
        range(max_steps),
        desc=f"Training {model_config_dict['model_name']}",
        unit="step",
    )

    for step in progress_bar:
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
            raise ValueError(
                f"Training loss became NaN or Inf at step {step + 1} "
                f"with learning_rate={learning_rate}."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_train_loss = loss.item()

        if (step + 1) % loss_log_interval == 0 or (step + 1) == max_steps:
            loss_curve_records.append(
                {
                    "step": step + 1,
                    "train_loss": final_train_loss,
                    "learning_rate": lr,
                }
            )

        progress_bar.set_postfix(
            {
                "loss": f"{final_train_loss:.4f}",
                "lr": f"{lr:.2e}",
            }
        )

    training_time_seconds = round(time.time() - start_time, 2)

    final_val_loss = estimate_val_loss(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        eval_iters=train_config["eval_iters"],
    )

    if loss_curve_path is not None:
        loss_curve_path = Path(loss_curve_path)
        loss_curve_path.parent.mkdir(parents=True, exist_ok=True)

        with open(loss_curve_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "train_loss", "learning_rate"],
            )
            writer.writeheader()
            writer.writerows(loss_curve_records)

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

    result = {
        "model_name": model_config_dict["model_name"],
        "parameterization": "uP",
        "learning_rate": learning_rate,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "training_time_seconds": training_time_seconds,
        "max_gpu_memory_allocated_GB": round(max_memory_allocated_gb, 3),
        "max_gpu_memory_reserved_GB": round(max_memory_reserved_gb, 3),
        "gpu_total_memory_GB": round(gpu_total_memory_gb, 3),
        "memory_allocated_percent": round(memory_allocated_percent, 2),
        "memory_reserved_percent": round(memory_reserved_percent, 2),
        "tokens_per_second": round(tokens_per_second, 2),
        "total_params": total_params,
        "batch_size": batch_size,
        "block_size": model_config_dict["block_size"],
        "tokens_per_step": tokens_per_step,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "loss_curve_path": str(loss_curve_path) if loss_curve_path is not None else None,
    }

    print("\nμP training finished")
    print(f"Model: {model_config_dict['model_name']}")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Training time: {training_time_seconds:.2f} sec")
    print(
        f"Max GPU allocated: {result['max_gpu_memory_allocated_GB']} GB "
        f"({result['memory_allocated_percent']}%)"
    )
    print(
        f"Max GPU reserved: {result['max_gpu_memory_reserved_GB']} GB "
        f"({result['memory_reserved_percent']}%)"
    )
    print(f"Tokens/sec: {result['tokens_per_second']}")

    return result