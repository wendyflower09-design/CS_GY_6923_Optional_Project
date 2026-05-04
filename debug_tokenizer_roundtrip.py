import json
from pathlib import Path

import cairosvg

from tokenization import decode_token_ids, encode_svg, load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent

TEXT_SPLIT_PATH = PROJECT_ROOT / "processed_svg_data" / "text_splits" / "train.jsonl"
TOKENIZER_PATH = PROJECT_ROOT / "processed_svg_data" / "tokenizer" / "tokenizer.json"

OUTPUT_DIR = PROJECT_ROOT / "debug_tokenizer_roundtrip_outputs"
ORIGINAL_SVG_PATH = OUTPUT_DIR / "original.svg"
TOKENIZER_DECODED_SVG_PATH = OUTPUT_DIR / "tokenizer_decoded.svg"
REPORT_PATH = OUTPUT_DIR / "roundtrip_report.json"


def load_first_svg() -> str:
    with open(TEXT_SPLIT_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline()

    if not first_line:
        raise ValueError(f"No SVG records found in {TEXT_SPLIT_PATH}")

    record = json.loads(first_line)
    return record["svg"]


def can_render(svg_text: str, output_path: Path) -> tuple[bool, str | None]:
    try:
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(output_path),
            output_width=256,
            output_height=256,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def inspect_text(svg_text: str) -> dict:
    return {
        "length": len(svg_text),
        "contains_svgxmlns": "<svgxmlns" in svg_text,
        "contains_pathd": "<pathd" in svg_text,
        "contains_viewbox_002424": 'viewBox="002424"' in svg_text,
        "contains_svg_space_xmlns": "<svg xmlns" in svg_text,
        "contains_path_space_d": "<path d" in svg_text,
        "preview": svg_text[:500],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    original_svg = load_first_svg()

    ids = encode_svg(tokenizer, original_svg, add_eos=False)
    tokenizer_decoded_svg = decode_token_ids(tokenizer, ids)

    ORIGINAL_SVG_PATH.write_text(original_svg, encoding="utf-8")
    TOKENIZER_DECODED_SVG_PATH.write_text(tokenizer_decoded_svg, encoding="utf-8")

    original_render_success, original_render_error = can_render(
        original_svg,
        OUTPUT_DIR / "original.png",
    )
    tokenizer_render_success, tokenizer_render_error = can_render(
        tokenizer_decoded_svg,
        OUTPUT_DIR / "tokenizer_decoded.png",
    )
    report = {
        "text_split_path": str(TEXT_SPLIT_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "num_token_ids": len(ids),
        "original": {
            **inspect_text(original_svg),
            "render_success": original_render_success,
            "render_error": original_render_error,
            "path": str(ORIGINAL_SVG_PATH),
        },
        "tokenizer_decode": {
            **inspect_text(tokenizer_decoded_svg),
            "render_success": tokenizer_render_success,
            "render_error": tokenizer_render_error,
            "path": str(TOKENIZER_DECODED_SVG_PATH),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n===== ORIGINAL SVG =====")
    print(original_svg)

    print("\n===== TOKENIZER DECODED SVG =====")
    print(tokenizer_decoded_svg)

    print("\nTokenizer round-trip finished.")
    print(f"Token ids: {len(ids)}")
    print(f"Original render: {original_render_success}")
    print(f"Tokenizer decode render: {tokenizer_render_success}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
