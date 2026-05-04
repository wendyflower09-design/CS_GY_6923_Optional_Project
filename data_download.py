from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def _contains_svg(text: str) -> bool:
    return isinstance(text, str) and "<svg" in text.lower()


def _find_svg_string(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        if _contains_svg(obj):
            return obj
        return None

    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_svg_string(value)
            if found is not None:
                return found

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_svg_string(value)
            if found is not None:
                return found

    return None


def _extract_svg_from_record(record: Dict[str, Any], svg_field: Optional[str]) -> Optional[str]:
    if svg_field is not None:
        value = record.get(svg_field)
        if _contains_svg(value):
            return value

    common_fields = [
        "svg",
        "SVG",
        "Svg",
        "code",
        "text",
        "content",
        "image",
        "xml",
    ]

    for field in common_fields:
        if field in record and _contains_svg(record[field]):
            return record[field]

    return _find_svg_string(record)


def _load_single_dataset(dataset_name: str, split: str, cache_dir: Optional[str]):
    try:
        return load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception:
        dataset_dict = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        if split in dataset_dict:
            return dataset_dict[split]

        all_splits = list(dataset_dict.values())
        return concatenate_datasets(all_splits)


def _resolve_max_samples(
    dataset_name: str,
    max_samples_per_dataset: Optional[Union[int, Dict[str, Optional[int]]]],
) -> Optional[int]:
    if max_samples_per_dataset is None:
        return None

    if isinstance(max_samples_per_dataset, int):
        return max_samples_per_dataset

    if isinstance(max_samples_per_dataset, dict):
        return max_samples_per_dataset.get(dataset_name, None)

    raise TypeError(
        "max_samples_per_dataset must be None, int, or dict[str, Optional[int]]."
    )


def download_and_merge_datasets(
    dataset_names: List[str],
    split: str = "train",
    svg_field: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_samples_per_dataset: Optional[Union[int, Dict[str, Optional[int]]]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Download multiple HuggingFace datasets, extract SVG strings, and merge them.

    max_samples_per_dataset:
        None:
            Read all samples from all datasets.
        int:
            Read at most this many samples from each dataset.
        dict:
            Specify a different limit per dataset.
            Example:
                {
                    "starvector/svg-icons-simple": None,
                    "starvector/svg-emoji-simple": None,
                    "starvector/svg-fonts-simple": 100_000,
                }

    Returns:
        all_svgs:
            A merged list of raw SVG strings.
        stats:
            Dataset-level download and extraction statistics.
    """

    all_svgs: List[str] = []

    stats: Dict[str, Any] = {
        "datasets": [],
        "total_extracted_svg": 0,
    }

    for dataset_name in dataset_names:
        dataset = _load_single_dataset(
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir,
        )

        total_items = len(dataset) if hasattr(dataset, "__len__") else None

        dataset_limit = _resolve_max_samples(
            dataset_name=dataset_name,
            max_samples_per_dataset=max_samples_per_dataset,
        )

        if dataset_limit is not None:
            dataset = dataset.select(range(min(dataset_limit, len(dataset))))

        extracted_count = 0
        skipped_count = 0

        for record in tqdm(
            dataset,
            desc=f"Reading {dataset_name}",
            total=len(dataset) if hasattr(dataset, "__len__") else total_items,
            unit="item",
        ):
            svg = _extract_svg_from_record(record, svg_field)

            if svg is None:
                skipped_count += 1
                continue

            all_svgs.append(svg)
            extracted_count += 1

        stats["datasets"].append(
            {
                "name": dataset_name,
                "raw_items": total_items,
                "sample_limit": dataset_limit,
                "processed_items": len(dataset),
                "extracted_svg": extracted_count,
                "skipped_no_svg": skipped_count,
            }
        )

    stats["total_extracted_svg"] = len(all_svgs)

    return all_svgs, stats