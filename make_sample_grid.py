import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MANIFEST_PATH = Path("part4_outputs/generated_manifest.json")
FIGURES_DIR = Path("part4_outputs/figures")

CELL_SIZE = 256
LABEL_HEIGHT = 40
BACKGROUND = (255, 255, 255)


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def open_png(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((CELL_SIZE, CELL_SIZE))
    return img


def make_grid(items, output_path, columns=5, title=None):
    valid_items = [item for item in items if item.get("png_path") is not None]

    if len(valid_items) == 0:
        print(f"No valid PNGs for {output_path}")
        return

    rows = (len(valid_items) + columns - 1) // columns

    title_height = 40 if title else 0

    width = columns * CELL_SIZE
    height = title_height + rows * (CELL_SIZE + LABEL_HEIGHT)

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((10, 10), title, fill=(0, 0, 0))

    for idx, item in enumerate(valid_items):
        row = idx // columns
        col = idx % columns

        x = col * CELL_SIZE
        y = title_height + row * (CELL_SIZE + LABEL_HEIGHT)

        img = open_png(item["png_path"])
        canvas.paste(img, (x, y))

        label = f"{item['type']} | T={item['temperature']}"
        if item["type"] == "prefix":
            label += f" | {item['prefix'][:20]}"

        draw.text((x + 5, y + CELL_SIZE + 5), label, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    unconditional_items = [
        item for item in manifest
        if item["type"] == "unconditional"
    ]

    prefix_items = [
        item for item in manifest
        if item["type"] == "prefix"
    ]

    temperature_items = [
        item for item in manifest
        if item["type"] == "temperature"
    ]

    make_grid(
        unconditional_items,
        FIGURES_DIR / "unconditional_grid.png",
        columns=5,
        title="Unconditional SVG Samples",
    )

    make_grid(
        prefix_items,
        FIGURES_DIR / "prefix_completion_grid.png",
        columns=5,
        title="Prefix-Conditioned SVG Samples",
    )

    make_grid(
        temperature_items,
        FIGURES_DIR / "temperature_comparison_grid.png",
        columns=3,
        title="Temperature Comparison",
    )

    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()