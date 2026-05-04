import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cairosvg
from lxml import etree
from tqdm import tqdm


COMMENT_PATTERN = re.compile(r"<!--.*?-->", flags=re.DOTALL)
METADATA_PATTERN = re.compile(
    r"<metadata\b[^>]*>.*?</metadata>",
    flags=re.IGNORECASE | re.DOTALL,
)

URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+",
    flags=re.IGNORECASE,
)

DECIMAL_NUMBER_PATTERN = re.compile(
    r"(?<![#A-Za-z0-9])[-+]?(?:\d+\.\d+|\.\d+)(?:[eE][-+]?\d+)?"
    r"|(?<![#A-Za-z0-9])[-+]?\d+(?:[eE][-+]?\d+)"
)

APPROX_TOKEN_PATTERN = re.compile(
    r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+"
    r"|</?[A-Za-z][A-Za-z0-9:_-]*"
    r"|/>|>|=|[\"']"
    r"|#[0-9A-Fa-f]{3,8}"
    r"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"|[A-Za-z_:-][A-Za-z0-9_:\.-]*"
    r"|[,;(){}]"
)


def remove_comments(svg: str) -> str:
    return COMMENT_PATTERN.sub("", svg)


def strip_metadata(svg: str) -> str:
    return METADATA_PATTERN.sub("", svg)


def normalize_whitespace(svg: str) -> str:
    svg = re.sub(r">\s+<", "><", svg)
    svg = re.sub(r"\s+", " ", svg)
    return svg.strip()


def _format_number(value: float, decimals: int) -> str:
    rounded = round(value, decimals)

    if abs(rounded) == 0:
        return "0"

    if rounded.is_integer():
        return str(int(rounded))

    text = f"{rounded:.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    return text


def _round_numbers_in_non_url_text(text: str, decimals: int) -> str:
    def replace_number(match: re.Match) -> str:
        value = float(match.group(0))
        return _format_number(value, decimals)

    return DECIMAL_NUMBER_PATTERN.sub(replace_number, text)


def round_numbers(svg: str, decimals: int = 1) -> str:
    pieces = []
    last_end = 0

    for match in URL_PATTERN.finditer(svg):
        before = svg[last_end:match.start()]
        url = match.group(0)

        pieces.append(_round_numbers_in_non_url_text(before, decimals))
        pieces.append(url)

        last_end = match.end()

    pieces.append(_round_numbers_in_non_url_text(svg[last_end:], decimals))

    return "".join(pieces)


def estimate_approx_token_count(svg: str) -> int:
    return len(APPROX_TOKEN_PATTERN.findall(svg))


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _remove_metadata_elements(root: etree._Element) -> None:
    for element in list(root.iter()):
        if _local_name(element.tag).lower() == "metadata":
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)


def _sort_attributes(element: etree._Element) -> None:
    if element.attrib:
        sorted_items = sorted(element.attrib.items())
        element.attrib.clear()
        for key, value in sorted_items:
            element.attrib[key] = value

    for child in element:
        _sort_attributes(child)


def parse_xml(svg: str) -> etree._Element:
    parser = etree.XMLParser(
        remove_blank_text=True,
        remove_comments=True,
        recover=False,
        resolve_entities=False,
    )
    return etree.fromstring(svg.encode("utf-8"), parser=parser)


def is_valid_svg_xml(svg: str) -> bool:
    try:
        root = parse_xml(svg)
        return _local_name(root.tag).lower() == "svg"
    except Exception:
        return False


def canonicalize_svg(svg: str) -> str:
    root = parse_xml(svg)

    if _local_name(root.tag).lower() != "svg":
        raise ValueError("Root element is not <svg>.")

    _remove_metadata_elements(root)
    _sort_attributes(root)

    return etree.tostring(
        root,
        encoding="unicode",
        method="xml",
        pretty_print=False,
    )


def clean_svg(svg: str, round_decimals: int) -> str:
    svg = remove_comments(svg)
    svg = strip_metadata(svg)
    svg = normalize_whitespace(svg)
    svg = round_numbers(svg, decimals=round_decimals)
    svg = normalize_whitespace(svg)
    return svg


def render_svg_to_png(svg: str, output_path: Path) -> None:
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(output_path),
        output_width=256,
        output_height=256,
    )


def _select_complexity_samples(
    svgs: List[str],
    sample_size: int,
) -> List[Tuple[str, int, str, int]]:
    if len(svgs) == 0 or sample_size <= 0:
        return []

    samples_per_level = max(1, sample_size // 3)

    indexed_lengths = [
        (idx, estimate_approx_token_count(svg))
        for idx, svg in enumerate(svgs)
    ]

    indexed_lengths.sort(key=lambda x: x[1])

    n = len(indexed_lengths)

    simple_pool = indexed_lengths[:max(samples_per_level * 10, samples_per_level)]
    complex_pool = indexed_lengths[-max(samples_per_level * 10, samples_per_level):]

    mid = n // 2
    half_window = max(samples_per_level * 5, samples_per_level)
    medium_pool = indexed_lengths[
        max(0, mid - half_window): min(n, mid + half_window)
    ]

    selected = []

    for level_name, pool in [
        ("simple", simple_pool),
        ("medium", medium_pool),
        ("complex", complex_pool),
    ]:
        if len(pool) == 0:
            continue

        if len(pool) <= samples_per_level:
            chosen = pool
        else:
            step = max(1, len(pool) // samples_per_level)
            chosen = pool[::step][:samples_per_level]

        for rank, (idx, length) in enumerate(chosen):
            selected.append((level_name, rank, "idx_" + str(idx), idx))

    return selected


def run_render_validation(
    svgs: List[str],
    sample_size: int,
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_samples = _select_complexity_samples(
        svgs=svgs,
        sample_size=sample_size,
    )

    success = 0
    failed = 0
    failed_indices = []

    for level_name, rank, idx_label, svg_index in tqdm(
        selected_samples,
        desc="CairoSVG render validation by complexity",
        unit="svg",
    ):
        svg = svgs[svg_index]

        approx_tokens = estimate_approx_token_count(svg)

        base_name = f"{level_name}_{rank}_idx_{svg_index}_tokens_{approx_tokens}"
        png_path = output_dir / f"{base_name}.png"
        svg_path = output_dir / f"{base_name}.svg"

        try:
            render_svg_to_png(svg, png_path)

            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg)

            success += 1

        except Exception:
            failed += 1
            failed_indices.append(svg_index)

    return {
        "sample_size_requested": sample_size,
        "sample_size_used": len(selected_samples),
        "samples_per_level": max(1, sample_size // 3),
        "selection_strategy": "3 simple, 3 medium, 3 complex based on approximate token length",
        "render_success": success,
        "render_failed": failed,
        "failed_indices": failed_indices,
        "output_dir": str(output_dir),
    }


def normalize_dataset(
    svgs: List[str],
    min_chars: int,
    max_approx_tokens: int,
    round_decimals: int,
    render_sample_size: int,
    render_output_dir: Path,
    seed: int,
) -> Tuple[List[str], Dict[str, Any]]:
    normalized_svgs: List[str] = []

    stats: Dict[str, Any] = {
        "input_count": len(svgs),
        "kept_count": 0,
        "dropped": {
            "non_string": 0,
            "too_short": 0,
            "too_long_approx_tokens": 0,
            "xml_invalid": 0,
            "canonicalization_failed": 0,
        },
        "min_chars": min_chars,
        "max_approx_tokens": max_approx_tokens,
        "round_decimals": round_decimals,
    }

    for svg in tqdm(svgs, desc="Cleaning/filtering/XML validation", unit="svg"):
        if not isinstance(svg, str):
            stats["dropped"]["non_string"] += 1
            continue

        try:
            cleaned = clean_svg(svg, round_decimals=round_decimals)
        except Exception:
            stats["dropped"]["xml_invalid"] += 1
            continue

        if len(cleaned) < min_chars:
            stats["dropped"]["too_short"] += 1
            continue

        approx_tokens = estimate_approx_token_count(cleaned)
        if approx_tokens > max_approx_tokens:
            stats["dropped"]["too_long_approx_tokens"] += 1
            continue

        if not is_valid_svg_xml(cleaned):
            stats["dropped"]["xml_invalid"] += 1
            continue

        try:
            canonicalized = canonicalize_svg(cleaned)
        except Exception:
            stats["dropped"]["canonicalization_failed"] += 1
            continue

        normalized_svgs.append(canonicalized)

    stats["kept_count"] = len(normalized_svgs)

    render_stats = run_render_validation(
        svgs=normalized_svgs,
        sample_size=render_sample_size,
        output_dir=render_output_dir,
        seed=seed,
    )

    stats["render_validation"] = render_stats

    return normalized_svgs, stats