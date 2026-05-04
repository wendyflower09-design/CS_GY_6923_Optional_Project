import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _length_summary(lengths: List[int]) -> Dict[str, Any]:
    if len(lengths) == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    arr = np.asarray(lengths)

    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def _plot_length_histogram(
    split_name: str,
    lengths: List[int],
    output_path: Path,
) -> None:
    if len(lengths) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50)
    plt.xlabel("Tokens per SVG")
    plt.ylabel("Number of SVG files")
    plt.title(f"{split_name} sequence length distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _make_json_safe_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove large raw length lists before writing JSON report.
    """

    safe_report = json.loads(json.dumps(report, default=str))

    if "tokenization" in safe_report:
        for split in ["train", "val", "test"]:
            if split in safe_report["tokenization"]["splits"]:
                safe_report["tokenization"]["splits"][split].pop("lengths", None)

    return safe_report


def _print_dataset_stats(download_stats: Dict[str, Any]) -> None:
    print("\n[Download]")
    print(f"Total extracted SVG files: {download_stats.get('total_extracted_svg', 0):,}")

    for item in download_stats.get("datasets", []):
        print(
            f"- {item['name']}: "
            f"extracted={item['extracted_svg']:,}, "
            f"skipped_no_svg={item['skipped_no_svg']:,}"
        )


def _print_normalization_stats(normalization_stats: Dict[str, Any]) -> None:
    print("\n[Normalization / Filtering / Validation]")
    print(f"Input files: {normalization_stats.get('input_count', 0):,}")
    print(f"Kept files: {normalization_stats.get('kept_count', 0):,}")

    dropped = normalization_stats.get("dropped", {})
    print("Dropped:")
    for key, value in dropped.items():
        print(f"- {key}: {value:,}")

    render_stats = normalization_stats.get("render_validation", {})
    print("\n[Render Validation]")
    print(f"Requested render samples: {render_stats.get('sample_size_requested', 0):,}")
    print(f"Used render samples: {render_stats.get('sample_size_used', 0):,}")
    print(f"Render success: {render_stats.get('render_success', 0):,}")
    print(f"Render failed: {render_stats.get('render_failed', 0):,}")
    print(f"Render output directory: {render_stats.get('output_dir')}")


def _print_split_stats(split_stats: Dict[str, Any]) -> None:
    print("\n[Split]")
    print(f"Total files: {split_stats.get('total', 0):,}")
    print(f"Train files: {split_stats.get('train_files', 0):,}")
    print(f"Validation files: {split_stats.get('val_files', 0):,}")
    print(f"Test files: {split_stats.get('test_files', 0):,}")
    print(f"Shuffle seed: {split_stats.get('shuffle_seed')}")


def _print_tokenization_stats(tokenization_stats: Dict[str, Any]) -> None:
    print("\n[Tokenization]")
    print(f"Requested vocab size: {tokenization_stats.get('requested_vocab_size'):,}")
    print(f"Actual vocab size: {tokenization_stats.get('actual_vocab_size'):,}")
    print(f"URL tokens added: {tokenization_stats.get('num_url_tokens_added'):,}")
    print(f"Token dtype: {tokenization_stats.get('token_dtype')}")

    print("\n[Token Counts]")
    for split in ["train", "val", "test"]:
        info = tokenization_stats["splits"][split]
        print(
            f"- {split}: "
            f"files={info['files']:,}, "
            f"tokens={info['tokens']:,}, "
            f"bin={info['bin_path']}"
        )

    train_tokens = tokenization_stats["splits"]["train"]["tokens"]
    print("\n[100M Token Check]")
    if train_tokens >= 100_000_000:
        print(f"Train token count is sufficient: {train_tokens:,} >= 100,000,000")
    else:
        print(f"Train token count is below target: {train_tokens:,} < 100,000,000")


def _print_length_stats(length_summaries: Dict[str, Dict[str, Any]]) -> None:
    print("\n[Sequence Length Distribution]")
    for split, summary in length_summaries.items():
        print(f"\n{split}:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.2f}")
            else:
                print(f"- {key}: {value}")


def create_statistics_report(
    output_dir: Path,
    download_stats: Dict[str, Any],
    normalization_stats: Dict[str, Any],
    split_stats: Dict[str, Any],
    tokenization_stats: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(output_dir)

    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    length_summaries = {}

    for split in ["train", "val", "test"]:
        lengths = tokenization_stats["splits"][split]["lengths"]

        length_summaries[split] = _length_summary(lengths)

        histogram_path = stats_dir / f"{split}_length_histogram.png"
        _plot_length_histogram(
            split_name=split,
            lengths=lengths,
            output_path=histogram_path,
        )

        tokenization_stats["splits"][split]["histogram_path"] = str(histogram_path)

    report = {
        "download": download_stats,
        "normalization": normalization_stats,
        "split": split_stats,
        "tokenization": tokenization_stats,
        "sequence_length_summary": length_summaries,
    }

    safe_report = _make_json_safe_report(report)

    json_path = stats_dir / "data_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=2, ensure_ascii=False)

    print("\n===================================")
    print("Final Data Statistics")
    print("===================================")

    _print_dataset_stats(download_stats)
    _print_normalization_stats(normalization_stats)
    _print_split_stats(split_stats)
    _print_tokenization_stats(tokenization_stats)
    _print_length_stats(length_summaries)

    print("\n[Saved Files]")
    print(f"Statistics JSON: {json_path}")
    print(f"Histogram directory: {stats_dir}")

    return report