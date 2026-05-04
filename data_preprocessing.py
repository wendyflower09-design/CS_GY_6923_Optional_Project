from pathlib import Path
import random

from data_download import download_and_merge_datasets
from normalization import normalize_dataset
from tokenization import train_and_tokenize
from stats import create_statistics_report


# --------------------------------------------------
# Global Config
# --------------------------------------------------

DATAS = [
    "starvector/svg-icons-simple",
    "starvector/svg-emoji-simple",
    "starvector/svg-fonts-simple",
]

MAX_SAMPLES_PER_DATASET = {
    "starvector/svg-icons-simple": None,
    "starvector/svg-emoji-simple": None,
    "starvector/svg-fonts-simple": 100_000,
}

OUTPUT_DIR = Path("processed_svg_data")
RENDER_SAMPLE_DIR = OUTPUT_DIR / "render_samples"

RANDOM_SEED = 42

TRAIN_RATIO = 0.98
VAL_RATIO = 0.01
TEST_RATIO = 0.01

MIN_CHARS = 50
MAX_APPROX_TOKENS = 1024

ROUND_DECIMALS = 1

# 9 samples total: 3 simple, 3 medium, 3 complex
RENDER_SAMPLE_SIZE = 9

VOCAB_SIZE = 4096


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def print_step(step_id: int, title: str) -> None:
    print("\n-----------------------------------")
    print(f"Step {step_id}: {title}")
    print("-----------------------------------")


def split_dataset(data, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    data = list(data)

    rng = random.Random(seed)
    rng.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    split_stats = {
        "total": n,
        "train_files": len(train_data),
        "val_files": len(val_data),
        "test_files": len(test_data),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "shuffle_seed": seed,
    }

    return train_data, val_data, test_data, split_stats


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_step(1, "Download and Merge Data")

    raw_svgs, download_stats = download_and_merge_datasets(
        dataset_names=DATAS,
        split="train",
        svg_field=None,
        cache_dir=None,
        max_samples_per_dataset=MAX_SAMPLES_PER_DATASET,
    )

    print_step(2, "Normalize, Filter, XML Validate, and Render Validate")

    normalized_svgs, normalization_stats = normalize_dataset(
        svgs=raw_svgs,
        min_chars=MIN_CHARS,
        max_approx_tokens=MAX_APPROX_TOKENS,
        round_decimals=ROUND_DECIMALS,
        render_sample_size=RENDER_SAMPLE_SIZE,
        render_output_dir=RENDER_SAMPLE_DIR,
        seed=RANDOM_SEED,
    )

    print_step(3, "Shuffle and Split Data")

    train_svgs, val_svgs, test_svgs, split_stats = split_dataset(
        data=normalized_svgs,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    print_step(4, "Train Tokenizer and Tokenize Splits")

    tokenization_stats = train_and_tokenize(
        train_svgs=train_svgs,
        val_svgs=val_svgs,
        test_svgs=test_svgs,
        output_dir=OUTPUT_DIR,
        vocab_size=VOCAB_SIZE,
        save_text_splits=True,
        add_eos=True,
    )

    print_step(5, "Final Documentation and Statistics")

    create_statistics_report(
        output_dir=OUTPUT_DIR,
        download_stats=download_stats,
        normalization_stats=normalization_stats,
        split_stats=split_stats,
        tokenization_stats=tokenization_stats,
    )


if __name__ == "__main__":
    main()
