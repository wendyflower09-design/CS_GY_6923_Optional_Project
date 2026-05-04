import json
from pathlib import Path

from tokenization import decode_token_ids, encode_svg, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "processed_svg_data" / "text_splits" / "train.jsonl"
TOKENIZER_PATH = PROJECT_ROOT / "processed_svg_data" / "tokenizer" / "tokenizer.json"
OUTPUT_DIR = PROJECT_ROOT / "tokenization_testing_output"
OUTPUT_TXT_PATH = OUTPUT_DIR / "bpe_roundtrip_results.txt"

NUM_SAMPLES = 10


def load_svg_samples(path: Path, count: int) -> list[str]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= count:
                break

            record = json.loads(line)
            svg = record.get("svg")

            if isinstance(svg, str):
                samples.append(svg)

    if len(samples) < count:
        raise ValueError(f"Expected {count} SVG samples, found {len(samples)}.")

    return samples


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normalized_svgs = load_svg_samples(INPUT_PATH, NUM_SAMPLES)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("Trained ByteLevel BPE Tokenizer Round-trip Test\n")
        f.write(f"Input path: {INPUT_PATH}\n")
        f.write(f"Tokenizer path: {TOKENIZER_PATH}\n")
        f.write(f"Samples: {NUM_SAMPLES}\n")
        f.write(f"Vocab size: {tokenizer.get_vocab_size()}\n\n")

        for index, svg in enumerate(normalized_svgs):
            token_ids = encode_svg(tokenizer, svg, add_eos=False)
            decoded_svg = decode_token_ids(tokenizer, token_ids)

            f.write("=" * 80 + "\n")
            f.write(f"Sample {index}\n")
            f.write("=" * 80 + "\n\n")

            f.write("----- ORIGINAL NORMALIZED SVG -----\n")
            f.write(svg + "\n\n")

            f.write("----- ENCODED TOKEN IDS -----\n")
            f.write(json.dumps(token_ids) + "\n\n")

            f.write("----- DECODED SVG -----\n")
            f.write(decoded_svg + "\n\n")

            f.write("----- ROUNDTRIP CHECK -----\n")
            f.write(f"Exact match: {svg == decoded_svg}\n")
            f.write(f"Original length: {len(svg)}\n")
            f.write(f"Decoded length: {len(decoded_svg)}\n")
            f.write(f"Token count: {len(token_ids)}\n\n")

    print("Tokenizer testing finished.")
    print(f"Results saved to: {OUTPUT_TXT_PATH}")


if __name__ == "__main__":
    main()
