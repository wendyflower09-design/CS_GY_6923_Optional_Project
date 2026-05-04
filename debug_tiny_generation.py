import csv
import json
import math
import pickle
import random
import time
from pathlib import Path

import cairosvg
import numpy as np
import torch
from tqdm import tqdm

from config import DATA_DIR, META_PATH, TINY_MODEL_CONFIG, TRAIN_CONFIG
from dataset import TokenDataset
from models import TransformerConfig, TransformerLM
from tokenization import decode_token_ids, encode_svg, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "debug_tiny_outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "tiny_debug_checkpoint.pt"
TRAINING_LOG_PATH = OUTPUT_DIR / "tiny_debug_training_loss.csv"
GENERATION_MANIFEST_PATH = OUTPUT_DIR / "tiny_debug_generation_manifest.json"
DECODED_RESULTS_PATH = OUTPUT_DIR / "tiny_debug_decoded_generations.txt"
SVG_OUTPUT_DIR = OUTPUT_DIR / "generated_svgs"
PNG_OUTPUT_DIR = OUTPUT_DIR / "rendered_pngs"

LEARNING_RATE = 0.01
DEBUG_EPOCHS = 5
DEBUG_BATCH_SIZE = 64
GENERATE_COUNT = 10
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.5
TOP_K = 20
TOP_P = 0.8

PREFIX = "<svg"


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


def build_tiny_model(vocab_size: int) -> TransformerLM:
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=TINY_MODEL_CONFIG["block_size"],
        n_layer=TINY_MODEL_CONFIG["n_layer"],
        n_head=TINY_MODEL_CONFIG["n_head"],
        n_embd=TINY_MODEL_CONFIG["n_embd"],
        dropout=TINY_MODEL_CONFIG["dropout"],
        bias=TINY_MODEL_CONFIG["bias"],
    )

    return TransformerLM(model_config)


@torch.no_grad()
def estimate_val_metrics(
    model: TransformerLM,
    dataset: TokenDataset,
    batch_size: int,
    eval_iters: int,
) -> tuple[float, float]:
    model.eval()

    losses = []
    correct = 0
    total = 0

    for _ in tqdm(range(eval_iters), desc="Validating tiny debug model", unit="batch"):
        x, y = dataset.get_batch("val", batch_size)
        logits, loss = model(x, y)

        if not torch.isfinite(loss):
            raise ValueError("Validation loss became NaN or Inf.")

        predictions = torch.argmax(logits, dim=-1)
        correct += int((predictions == y).sum().item())
        total += int(y.numel())
        losses.append(loss.item())

    model.train()

    val_loss = float(sum(losses) / len(losses))
    token_accuracy = correct / total if total > 0 else 0.0

    return val_loss, token_accuracy


def train_tiny_model() -> dict:
    train_config = dict(TRAIN_CONFIG)
    train_config["batch_size"] = DEBUG_BATCH_SIZE
    train_config["max_epochs"] = DEBUG_EPOCHS

    device = train_config["device"]

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")

    set_seed(train_config["seed"])

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    vocab_size = int(meta["vocab_size"])
    token_dtype = meta.get("token_dtype", "uint16")

    dataset = TokenDataset(
        data_dir=DATA_DIR,
        block_size=TINY_MODEL_CONFIG["block_size"],
        device=device,
        token_dtype=token_dtype,
    )

    model = build_tiny_model(vocab_size).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(train_config["beta1"], train_config["beta2"]),
        weight_decay=train_config["weight_decay"],
    )

    batch_size = train_config["batch_size"]
    block_size = TINY_MODEL_CONFIG["block_size"]
    tokens_per_step = batch_size * block_size
    steps_per_epoch = len(dataset.train_data) // tokens_per_step
    max_steps = steps_per_epoch * train_config["max_epochs"]

    if max_steps <= 0:
        raise ValueError("max_steps is 0. Check train.bin size, batch_size, and block_size.")

    warmup_steps = max(1, int(max_steps * train_config["warmup_ratio"]))
    loss_log_interval = train_config.get("loss_log_interval", 50)
    eval_iters = train_config.get("eval_iters", 50)

    print("\n-----------------------------------")
    print("Train Tiny Debug Model")
    print("-----------------------------------")
    print(f"Model config: {TINY_MODEL_CONFIG}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {DEBUG_EPOCHS}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {max_steps}")

    loss_records = []
    final_loss = None

    model.train()
    start_time = time.time()

    progress_bar = tqdm(range(max_steps), desc="Training tiny debug model", unit="step")

    for step in progress_bar:
        current_step = step + 1
        current_epoch = current_step / steps_per_epoch

        lr = get_lr(
            step=step,
            max_steps=max_steps,
            learning_rate=LEARNING_RATE,
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

        final_loss = loss.item()

        if current_step % loss_log_interval == 0 or current_step == max_steps:
            loss_records.append(
                {
                    "step": current_step,
                    "epoch": current_epoch,
                    "train_loss": final_loss,
                    "learning_rate": lr,
                }
            )

        progress_bar.set_postfix({"loss": f"{final_loss:.4f}", "lr": f"{lr:.2e}"})

    training_time_seconds = round(time.time() - start_time, 2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_val_loss, final_val_token_accuracy = estimate_val_metrics(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        eval_iters=eval_iters,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": TINY_MODEL_CONFIG,
        "train_config": train_config,
        "vocab_size": vocab_size,
        "token_dtype": token_dtype,
        "learning_rate": LEARNING_RATE,
        "epochs": DEBUG_EPOCHS,
        "max_steps": max_steps,
        "final_loss": final_loss,
        "final_val_loss": final_val_loss,
        "final_val_token_accuracy": final_val_token_accuracy,
        "total_params": total_params,
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    with open(TRAINING_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "epoch", "train_loss", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(loss_records)

    print("\nTiny debug training finished.")
    print(f"Final train loss: {final_loss:.4f}")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Final val token accuracy: {final_val_token_accuracy:.4f}")
    print(f"Training time: {training_time_seconds:.2f} sec")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Training log saved to: {TRAINING_LOG_PATH}")

    return checkpoint


def load_checkpoint_model():
    device = TRAIN_CONFIG["device"]

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_config = checkpoint["model_config"]
    vocab_size = int(checkpoint["vocab_size"])

    config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=model_config["block_size"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_embd=model_config["n_embd"],
        dropout=model_config["dropout"],
        bias=model_config["bias"],
    )

    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def top_k_top_p_filtering(logits, top_k: int, top_p: float):
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, -float("inf")),
            logits,
        )

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        remove_mask[:, 1:] = remove_mask[:, :-1].clone()
        remove_mask[:, 0] = False

        logits_to_remove = torch.full_like(logits, False, dtype=torch.bool)
        logits_to_remove.scatter_(1, sorted_indices, remove_mask)
        logits = logits.masked_fill(logits_to_remove, -float("inf"))

    return logits


@torch.no_grad()
def generate_svg_text(model, tokenizer, device: str) -> str:
    encoded = encode_svg(tokenizer, PREFIX, add_eos=False)

    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    for _ in range(MAX_NEW_TOKENS):
        idx_cond = idx[:, -model.config.block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / TEMPERATURE
        logits = top_k_top_p_filtering(logits, top_k=TOP_K, top_p=TOP_P)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        decoded_so_far = decode_token_ids(tokenizer, idx[0].tolist())

        if "</svg>" in decoded_so_far:
            break

    return decode_token_ids(tokenizer, idx[0].tolist())


def clean_generated_svg(text: str) -> str:
    start = text.find("<svg")

    if start == -1:
        return text.strip()

    text = text[start:]

    end = text.find("</svg>")

    if end != -1:
        text = text[: end + len("</svg>")]
    else:
        text = text.strip()

        if not text.endswith(">"):
            text += ">"

        text += "</svg>"

    return text.strip()


def save_and_render_svg(raw_text: str, sample_id: int) -> dict:
    base_name = f"prefix_sample_{sample_id:02d}"
    decoded_path = SVG_OUTPUT_DIR / f"{base_name}_decoded.txt"
    svg_path = SVG_OUTPUT_DIR / f"{base_name}.svg"
    png_path = PNG_OUTPUT_DIR / f"{base_name}.png"

    cleaned_svg = clean_generated_svg(raw_text)

    with open(decoded_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(cleaned_svg)

    render_success = False
    render_error = None

    try:
        cairosvg.svg2png(
            bytestring=cleaned_svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=256,
            output_height=256,
        )
        render_success = True
    except Exception as e:
        render_error = str(e)

    return {
        "sample_id": sample_id,
        "prefix": PREFIX,
        "decoded_path": str(decoded_path),
        "svg_path": str(svg_path),
        "png_path": str(png_path) if render_success else None,
        "render_success": render_success,
        "render_error": render_error,
    }


def generate_and_render_samples() -> list[dict]:
    tokenizer_path = PROJECT_ROOT / "processed_svg_data" / "tokenizer" / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    model, device = load_checkpoint_model()

    manifest = []
    decoded_records = []

    print("\n-----------------------------------")
    print("Generate And Render Prefix SVG Samples")
    print("-----------------------------------")

    for sample_id in tqdm(range(GENERATE_COUNT), desc="Generating SVGs", unit="svg"):
        raw_text = generate_svg_text(model=model, tokenizer=tokenizer, device=device)
        result = save_and_render_svg(raw_text=raw_text, sample_id=sample_id)
        manifest.append(result)
        decoded_records.append((sample_id, raw_text, result))

    with open(GENERATION_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(DECODED_RESULTS_PATH, "w", encoding="utf-8") as f:
        for sample_id, decoded_text, result in decoded_records:
            f.write("=" * 80 + "\n")
            f.write(f"Sample {sample_id}\n")
            f.write(f"Render success: {result['render_success']}\n")
            f.write(f"Render error: {result['render_error']}\n")
            f.write("=" * 80 + "\n")
            f.write(decoded_text)
            f.write("\n\n")

    render_success_count = sum(1 for item in manifest if item["render_success"])

    print("\nGeneration finished.")
    print(f"Total samples: {len(manifest)}")
    print(f"Rendered successfully: {render_success_count}/{len(manifest)}")
    print(f"Manifest saved to: {GENERATION_MANIFEST_PATH}")
    print(f"Decoded generations saved to: {DECODED_RESULTS_PATH}")

    return manifest


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_tiny_model()
    generate_and_render_samples()


if __name__ == "__main__":
    main()
