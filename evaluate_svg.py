import json
import math
import pickle
from pathlib import Path

import cairosvg
import numpy as np
import torch
from lxml import etree
from tqdm import tqdm

from uP_model import build_up_model_with_base_shapes


CHECKPOINT_PATH = Path("part4_outputs/checkpoints/xl_uP_best.pt")
META_PATH = Path("processed_svg_data/tokenizer/meta.pkl")
TEST_BIN_PATH = Path("processed_svg_data/tokenized/test.bin")

MANIFEST_PATH = Path("part4_outputs/generated_manifest.json")
EVALUATION_DIR = Path("part4_outputs/evaluation")
METRICS_PATH = EVALUATION_DIR / "evaluation_metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
BLOCK_SIZE = 1024
EVAL_ITERS = 100


def is_valid_xml(svg_text):
    try:
        root = etree.fromstring(svg_text.encode("utf-8"))
        return root.tag.lower().endswith("svg")
    except Exception:
        return False


def can_render(svg_text):
    try:
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=256,
            output_height=256,
        )
        return True
    except Exception:
        return False


def is_structurally_valid(svg_text):
    text = svg_text.lower()

    if "<svg" not in text:
        return False

    if "</svg>" not in text:
        return False

    common_svg_tokens = [
        "<path",
        "<rect",
        "<circle",
        "<ellipse",
        "<line",
        "<polyline",
        "<polygon",
        "<g",
    ]

    return any(token in text for token in common_svg_tokens)


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

    return model, checkpoint


def get_test_batch(test_data):
    max_start = len(test_data) - BLOCK_SIZE - 1

    indices = torch.randint(
        low=0,
        high=max_start,
        size=(BATCH_SIZE,),
    )

    x = torch.stack([
        torch.from_numpy(
            np.asarray(test_data[i : i + BLOCK_SIZE], dtype=np.int64)
        )
        for i in indices
    ])

    y = torch.stack([
        torch.from_numpy(
            np.asarray(test_data[i + 1 : i + 1 + BLOCK_SIZE], dtype=np.int64)
        )
        for i in indices
    ])

    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def compute_test_loss_and_perplexity(model, token_dtype):
    test_data = np.memmap(
        TEST_BIN_PATH,
        dtype=np.dtype(token_dtype),
        mode="r",
    )

    losses = []

    for _ in tqdm(range(EVAL_ITERS), desc="Computing test loss"):
        x, y = get_test_batch(test_data)
        _, loss = model(x, y)
        losses.append(loss.item())

    test_loss = float(sum(losses) / len(losses))
    test_perplexity = float(math.exp(test_loss))

    return test_loss, test_perplexity


def main():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    detailed_results = []

    xml_valid_count = 0
    render_valid_count = 0
    structural_valid_count = 0

    for item in tqdm(manifest, desc="Evaluating generated SVGs"):
        svg_path = Path(item["svg_path"])

        with open(svg_path, "r", encoding="utf-8") as f:
            svg_text = f.read()

        xml_valid = is_valid_xml(svg_text)
        render_valid = can_render(svg_text)
        structural_valid = is_structurally_valid(svg_text)

        xml_valid_count += int(xml_valid)
        render_valid_count += int(render_valid)
        structural_valid_count += int(structural_valid)

        detailed_results.append({
            **item,
            "xml_valid": xml_valid,
            "render_valid": render_valid,
            "structural_valid": structural_valid,
        })

    total = len(manifest)

    model, checkpoint = load_model()
    token_dtype = checkpoint.get("token_dtype", "uint16")

    test_loss, test_perplexity = compute_test_loss_and_perplexity(
        model=model,
        token_dtype=token_dtype,
    )

    metrics = {
        "num_generated_samples": total,
        "xml_valid_count": xml_valid_count,
        "xml_valid_rate": xml_valid_count / total if total > 0 else 0.0,
        "render_valid_count": render_valid_count,
        "render_valid_rate": render_valid_count / total if total > 0 else 0.0,
        "structural_valid_count": structural_valid_count,
        "structural_valid_rate": structural_valid_count / total if total > 0 else 0.0,
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "detailed_results": detailed_results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation finished.")
    print(f"XML valid rate: {metrics['xml_valid_rate']:.4f}")
    print(f"Render valid rate: {metrics['render_valid_rate']:.4f}")
    print(f"Structural valid rate: {metrics['structural_valid_rate']:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_perplexity:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()