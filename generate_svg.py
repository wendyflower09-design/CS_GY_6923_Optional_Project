import json
import csv
from pathlib import Path

import torch
import cairosvg
import matplotlib.pyplot as plt
from tqdm import tqdm

from tokenization import decode_token_ids, encode_svg, load_tokenizer
from uP_model import build_up_model_with_base_shapes


# --------------------------------------------------
# Paths
# --------------------------------------------------

CHECKPOINT_PATH = Path("part4_outputs/checkpoints/xl_uP_best.pt")
TOKENIZER_PATH = Path("processed_svg_data/tokenizer/tokenizer.json")

SVG_OUTPUT_DIR = Path("part4_outputs/generated_svgs")
PNG_OUTPUT_DIR = Path("part4_outputs/rendered_pngs")
MANIFEST_PATH = Path("part4_outputs/generated_manifest.json")
TRAIN_LOSS_PATH = Path("part4_outputs/training/xl_uP_best_loss.csv")
VAL_LOSS_PATH = Path("part4_outputs/training/xl_uP_best_val_loss.csv")
LOSS_PLOT_PATH = Path("part4_outputs/training/xl_uP_best_training_curves.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Generation Config
# --------------------------------------------------

MAX_NEW_TOKENS = 1024

TOP_K = 20
TOP_P = 0.8

UNCONDITIONAL_COUNT = 10

DEFAULT_PREFIX = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'

PREFIXES = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path',
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect',
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle',
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g',
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><polygon',
]

TEMPERATURES = [0.3, 0.5, 0.8]


# --------------------------------------------------
# Load Model
# --------------------------------------------------

def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model_config = checkpoint["model_config"]
    vocab_size = checkpoint["vocab_size"]

    model = build_up_model_with_base_shapes(
        model_config_dict=model_config,
        vocab_size=vocab_size,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# --------------------------------------------------
# Sampling Helpers
# --------------------------------------------------

def top_k_top_p_filtering(logits, top_k=20, top_p=0.8):
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
def generate(model, tokenizer, prefix, max_new_tokens, temperature):
    encoded = encode_svg(tokenizer, prefix, add_eos=False)

    idx = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE,
    )

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        logits = top_k_top_p_filtering(
            logits,
            top_k=TOP_K,
            top_p=TOP_P,
        )

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

        decoded_so_far = decode_token_ids(tokenizer, idx[0].tolist())

        if "</svg>" in decoded_so_far:
            break

    return decode_token_ids(tokenizer, idx[0].tolist())


# --------------------------------------------------
# SVG Cleaning / Rendering
# --------------------------------------------------

def clean_generated_svg(text):
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


def save_svg_and_render(raw_text, base_name):
    raw_path = SVG_OUTPUT_DIR / f"{base_name}_raw.txt"
    svg_path = SVG_OUTPUT_DIR / f"{base_name}.svg"
    png_path = PNG_OUTPUT_DIR / f"{base_name}.png"

    cleaned_svg = clean_generated_svg(raw_text)

    with open(raw_path, "w", encoding="utf-8") as f:
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
        "raw_path": str(raw_path),
        "svg_path": str(svg_path),
        "png_path": str(png_path) if render_success else None,
        "render_success": render_success,
        "render_error": render_error,
    }


# --------------------------------------------------
# Training Curve Plot
# --------------------------------------------------

def read_loss_curve(path, loss_column):
    steps = []
    losses = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row[loss_column]))

    return steps, losses


def plot_training_curves():
    if not TRAIN_LOSS_PATH.exists() or not VAL_LOSS_PATH.exists():
        print("\nTraining curve plot skipped.")
        print(f"Missing train loss file: {TRAIN_LOSS_PATH}")
        print(f"Missing val loss file: {VAL_LOSS_PATH}")
        return

    train_steps, train_losses = read_loss_curve(TRAIN_LOSS_PATH, "train_loss")
    val_steps, val_losses = read_loss_curve(VAL_LOSS_PATH, "val_loss")

    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="Train loss", alpha=0.8)
    plt.plot(val_steps, val_losses, label="Validation loss", marker="o", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("XL-uP Final Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close()

    print(f"Training curve plot saved to: {LOSS_PLOT_PATH}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    SVG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_model()

    manifest = []
    sample_id = 0

    for i in tqdm(range(UNCONDITIONAL_COUNT), desc="Generating unconditional samples"):
        raw_text = generate(
            model=model,
            tokenizer=tokenizer,
            prefix=DEFAULT_PREFIX,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.5,
        )

        base_name = f"unconditional_{i:02d}"
        saved = save_svg_and_render(raw_text, base_name)

        manifest.append(
            {
                "sample_id": sample_id,
                "type": "unconditional",
                "prefix": DEFAULT_PREFIX,
                "temperature": 0.5,
                **saved,
            }
        )

        sample_id += 1

    for i, prefix in enumerate(tqdm(PREFIXES, desc="Generating prefix samples")):
        raw_text = generate(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.5,
        )

        base_name = f"prefix_{i:02d}"
        saved = save_svg_and_render(raw_text, base_name)

        manifest.append(
            {
                "sample_id": sample_id,
                "type": "prefix",
                "prefix": prefix,
                "temperature": 0.5,
                **saved,
            }
        )

        sample_id += 1

    for temp in TEMPERATURES:
        for i in range(3):
            raw_text = generate(
                model=model,
                tokenizer=tokenizer,
                prefix=DEFAULT_PREFIX,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temp,
            )

            base_name = f"temperature_{str(temp).replace('.', '_')}_{i:02d}"
            saved = save_svg_and_render(raw_text, base_name)

            manifest.append(
                {
                    "sample_id": sample_id,
                    "type": "temperature",
                    "prefix": DEFAULT_PREFIX,
                    "temperature": temp,
                    **saved,
                }
            )

            sample_id += 1

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    render_success_count = sum(1 for item in manifest if item["render_success"])

    print("\nGeneration finished.")
    print(f"Total samples: {len(manifest)}")
    print(f"Rendered successfully: {render_success_count}/{len(manifest)}")
    print(f"SVGs saved to: {SVG_OUTPUT_DIR}")
    print(f"PNGs saved to: {PNG_OUTPUT_DIR}")
    print(f"Manifest saved to: {MANIFEST_PATH}")

    plot_training_curves()


if __name__ == "__main__":
    main()
