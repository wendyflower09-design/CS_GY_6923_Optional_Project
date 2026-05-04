import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]


def build_svg_tokenizer() -> Tokenizer:
    """
    Build a reversible ByteLevel BPE tokenizer for SVG text.

    ByteLevel tokenization preserves whitespace, XML attribute spacing, path
    numbers, URLs, and punctuation so token ids can be decoded back into valid
    SVG source text.
    """

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


def encode_svg(
    tokenizer: Tokenizer,
    svg: str,
    add_eos: bool = False,
) -> List[int]:
    ids = tokenizer.encode(svg).ids

    if add_eos:
        eos_id = tokenizer.token_to_id(EOS_TOKEN)

        if eos_id is None:
            raise ValueError("EOS token is missing from tokenizer.")

        ids.append(eos_id)

    return ids


def decode_token_ids(
    tokenizer: Tokenizer,
    ids: List[int],
) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _tqdm_text_iterator(texts: List[str], desc: str):
    for text in tqdm(texts, desc=desc, unit="svg"):
        yield text


def _save_jsonl(texts: List[str], path: Path, desc: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for svg in tqdm(texts, desc=desc, unit="svg"):
            f.write(json.dumps({"svg": svg}, ensure_ascii=False) + "\n")


def encode_split(
    tokenizer: Tokenizer,
    texts: List[str],
    add_eos: bool,
    desc: str,
) -> Tuple[List[int], List[int]]:
    all_ids: List[int] = []
    svg_lengths: List[int] = []

    for svg in tqdm(texts, desc=desc, unit="svg"):
        ids_without_eos = encode_svg(
            tokenizer=tokenizer,
            svg=svg,
            add_eos=False,
        )
        ids = list(ids_without_eos)

        if add_eos:
            eos_id = tokenizer.token_to_id(EOS_TOKEN)

            if eos_id is None:
                raise ValueError("EOS token is missing from tokenizer.")

            ids.append(eos_id)

        svg_lengths.append(len(ids_without_eos))
        all_ids.extend(ids)

    return all_ids, svg_lengths


def _select_token_dtype(vocab_size: int):
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def save_token_ids(ids: List[int], path: Path, dtype) -> None:
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(path)


def save_lengths(lengths: List[int], path: Path) -> None:
    arr = np.asarray(lengths, dtype=np.int32)
    np.save(path, arr)


def train_and_tokenize(
    train_svgs: List[str],
    val_svgs: List[str],
    test_svgs: List[str],
    output_dir: Path,
    vocab_size: int = 4096,
    save_text_splits: bool = True,
    add_eos: bool = True,
) -> Dict[str, Any]:
    """
    Train a reversible ByteLevel BPE tokenizer on train_svgs and tokenize all
    splits into binary token id files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_dir = output_dir / "text_splits"
    token_dir = output_dir / "tokenized"
    tokenizer_dir = output_dir / "tokenizer"

    text_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if save_text_splits:
        _save_jsonl(train_svgs, text_dir / "train.jsonl", "Saving train text split")
        _save_jsonl(val_svgs, text_dir / "val.jsonl", "Saving val text split")
        _save_jsonl(test_svgs, text_dir / "test.jsonl", "Saving test text split")

    tokenizer = build_svg_tokenizer()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

    tokenizer.train_from_iterator(
        _tqdm_text_iterator(train_svgs, "Training ByteLevel BPE tokenizer on train split"),
        trainer=trainer,
        length=len(train_svgs),
    )

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    train_ids, train_lengths = encode_split(
        tokenizer=tokenizer,
        texts=train_svgs,
        add_eos=add_eos,
        desc="Encoding train split",
    )

    val_ids, val_lengths = encode_split(
        tokenizer=tokenizer,
        texts=val_svgs,
        add_eos=add_eos,
        desc="Encoding val split",
    )

    test_ids, test_lengths = encode_split(
        tokenizer=tokenizer,
        texts=test_svgs,
        add_eos=add_eos,
        desc="Encoding test split",
    )

    actual_vocab_size = tokenizer.get_vocab_size()
    dtype = _select_token_dtype(actual_vocab_size)

    train_bin = token_dir / "train.bin"
    val_bin = token_dir / "val.bin"
    test_bin = token_dir / "test.bin"

    save_token_ids(train_ids, train_bin, dtype)
    save_token_ids(val_ids, val_bin, dtype)
    save_token_ids(test_ids, test_bin, dtype)

    train_lengths_path = token_dir / "train_lengths.npy"
    val_lengths_path = token_dir / "val_lengths.npy"
    test_lengths_path = token_dir / "test_lengths.npy"

    save_lengths(train_lengths, train_lengths_path)
    save_lengths(val_lengths, val_lengths_path)
    save_lengths(test_lengths, test_lengths_path)

    meta = {
        "tokenizer_type": "bytelevel_bpe",
        "decoder": "bytelevel",
        "vocab_size": actual_vocab_size,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_id": tokenizer.token_to_id(PAD_TOKEN),
        "unk_id": tokenizer.token_to_id(UNK_TOKEN),
        "bos_id": tokenizer.token_to_id(BOS_TOKEN),
        "eos_id": tokenizer.token_to_id(EOS_TOKEN),
        "tokenizer_file": str(tokenizer_path),
        "token_dtype": np.dtype(dtype).name,
        "add_eos": add_eos,
    }

    with open(tokenizer_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    with open(tokenizer_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tokenization_stats: Dict[str, Any] = {
        "tokenizer_type": "bytelevel_bpe",
        "decoder": "bytelevel",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "num_url_tokens_added": 0,
        "add_eos": add_eos,
        "tokenizer_path": str(tokenizer_path),
        "meta_pkl_path": str(tokenizer_dir / "meta.pkl"),
        "meta_json_path": str(tokenizer_dir / "meta.json"),
        "token_dtype": np.dtype(dtype).name,
        "special_token_ids": {
            "pad_id": tokenizer.token_to_id(PAD_TOKEN),
            "unk_id": tokenizer.token_to_id(UNK_TOKEN),
            "bos_id": tokenizer.token_to_id(BOS_TOKEN),
            "eos_id": tokenizer.token_to_id(EOS_TOKEN),
        },
        "splits": {
            "train": {
                "files": len(train_svgs),
                "tokens": len(train_ids),
                "bin_path": str(train_bin),
                "lengths_path": str(train_lengths_path),
                "lengths": train_lengths,
            },
            "val": {
                "files": len(val_svgs),
                "tokens": len(val_ids),
                "bin_path": str(val_bin),
                "lengths_path": str(val_lengths_path),
                "lengths": val_lengths,
            },
            "test": {
                "files": len(test_svgs),
                "tokens": len(test_ids),
                "bin_path": str(test_bin),
                "lengths_path": str(test_lengths_path),
                "lengths": test_lengths,
            },
        },
    }

    if save_text_splits:
        tokenization_stats["text_splits"] = {
            "train": str(text_dir / "train.jsonl"),
            "val": str(text_dir / "val.jsonl"),
            "test": str(text_dir / "test.jsonl"),
        }

    return tokenization_stats
