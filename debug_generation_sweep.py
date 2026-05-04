import json
from pathlib import Path

import cairosvg
import torch
from tqdm import tqdm

from config import TINY_MODEL_CONFIG, TRAIN_CONFIG
from models import TransformerConfig, TransformerLM
from tokenization import decode_token_ids, encode_svg, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = PROJECT_ROOT / "debug_tiny_outputs" / "tiny_debug_checkpoint.pt"
TOKENIZER_PATH = PROJECT_ROOT / "processed_svg_data" / "tokenizer" / "tokenizer.json"

OUTPUT_DIR = PROJECT_ROOT / "debug_generation_sweep_outputs"
SVG_OUTPUT_DIR = OUTPUT_DIR / "generated_svgs"
PNG_OUTPUT_DIR = OUTPUT_DIR / "rendered_pngs"
MANIFEST_PATH = OUTPUT_DIR / "generation_sweep_manifest.json"
DECODED_RESULTS_PATH = OUTPUT_DIR / "generation_sweep_decoded_results.txt"

DEVICE = TRAIN_CONFIG["device"]

MAX_NEW_TOKENS = 1024
TOP_K = 20
TOP_P = 0.8
SAMPLES_PER_SETTING = 3

PREFIXES = [
    "<svg",
    '<svg xmlns="http://www.w3.org/2000/svg"',
    '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0 0 24 24" width="200px">',
    '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0 0 24 24" width="200px"><path',
    '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0 0 24 24" width="200px"><path d="',
]

TEMPERATURES = [
    0.1,
    0.2,
    0.3,
    0.5,
]


def load_model():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model_config = checkpoint.get("model_config", TINY_MODEL_CONFIG)
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

    model = TransformerLM(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


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
def generate_svg_text(model, tokenizer, prefix: str, temperature: float) -> str:
    encoded = encode_svg(tokenizer, prefix, add_eos=False)
    idx = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    eos_id = tokenizer.token_to_id("<EOS>")

    for _ in range(MAX_NEW_TOKENS):
        idx_cond = idx[:, -model.config.block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        logits = top_k_top_p_filtering(logits, top_k=TOP_K, top_p=TOP_P)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        next_token_id = int(next_token.item())
        decoded_so_far = decode_token_ids(tokenizer, idx[0].tolist())

        if next_token_id == eos_id or "</svg>" in decoded_so_far:
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


def save_and_render(raw_text: str, base_name: str) -> dict:
    decoded_path = SVG_OUTPUT_DIR / f"{base_name}_decoded.txt"
    svg_path = SVG_OUTPUT_DIR / f"{base_name}.svg"
    png_path = PNG_OUTPUT_DIR / f"{base_name}.png"

    cleaned_svg = clean_generated_svg(raw_text)

    decoded_path.write_text(raw_text, encoding="utf-8")
    svg_path.write_text(cleaned_svg, encoding="utf-8")

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
        "decoded_path": str(decoded_path),
        "svg_path": str(svg_path),
        "png_path": str(png_path) if render_success else None,
        "render_success": render_success,
        "render_error": render_error,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_model()

    manifest = []
    decoded_records = []
    sample_id = 0

    total_runs = len(PREFIXES) * len(TEMPERATURES) * SAMPLES_PER_SETTING
    progress = tqdm(total=total_runs, desc="Generation sweep", unit="sample")

    for prefix_index, prefix in enumerate(PREFIXES):
        for temperature in TEMPERATURES:
            for repeat_index in range(SAMPLES_PER_SETTING):
                base_name = (
                    f"sample_{sample_id:03d}"
                    f"_prefix_{prefix_index}"
                    f"_temp_{str(temperature).replace('.', '_')}"
                    f"_repeat_{repeat_index}"
                )

                raw_text = generate_svg_text(
                    model=model,
                    tokenizer=tokenizer,
                    prefix=prefix,
                    temperature=temperature,
                )
                saved = save_and_render(raw_text=raw_text, base_name=base_name)

                item = {
                    "sample_id": sample_id,
                    "prefix_index": prefix_index,
                    "prefix": prefix,
                    "temperature": temperature,
                    "repeat_index": repeat_index,
                    **saved,
                }

                manifest.append(item)
                decoded_records.append((item, raw_text))
                sample_id += 1
                progress.update(1)

    progress.close()

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with open(DECODED_RESULTS_PATH, "w", encoding="utf-8") as f:
        for item, decoded_text in decoded_records:
            f.write("=" * 80 + "\n")
            f.write(f"Sample {item['sample_id']}\n")
            f.write(f"Prefix index: {item['prefix_index']}\n")
            f.write(f"Prefix: {item['prefix']}\n")
            f.write(f"Temperature: {item['temperature']}\n")
            f.write(f"Render success: {item['render_success']}\n")
            f.write(f"Render error: {item['render_error']}\n")
            f.write("=" * 80 + "\n")
            f.write(decoded_text)
            f.write("\n\n")

    render_success_count = sum(1 for item in manifest if item["render_success"])

    print("\nGeneration sweep finished.")
    print(f"Total samples: {len(manifest)}")
    print(f"Rendered successfully: {render_success_count}/{len(manifest)}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print(f"Decoded results saved to: {DECODED_RESULTS_PATH}")


if __name__ == "__main__":
    main()
