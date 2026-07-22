"""Streaming corpus batches for J_l estimation.

Streams a Hugging Face dataset (no local download) and tokenizes documents,
keeping only those long enough to fill a full seq_len chunk — batches are
unpadded, so every position is a real token and no attention mask is needed.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from datasets import load_dataset


def stream_batches(
    tokenizer,
    dataset: str,
    dataset_config: str | None,
    *,
    seq_len: int,
    batch_size: int,
    skip: int = 0,
    limit: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield (batch_size, seq_len) input_ids tensors.

    skip fast-forwards past the first N usable prompts (for --resume).
    limit caps total usable prompts counted from the stream start, including
    skipped ones, so (skip, limit) partitions are reproducible.
    """
    ds = load_dataset(dataset, dataset_config, split="train", streaming=True)
    seen = 0
    buf: list[list[int]] = []
    for row in ds:
        if limit is not None and seen >= limit:
            break
        ids = tokenizer(row["text"], truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) < seq_len:
            continue
        seen += 1
        if seen <= skip:
            continue
        buf.append(ids)
        if len(buf) == batch_size:
            yield torch.tensor(buf, dtype=torch.long)
            buf = []
    if buf:
        yield torch.tensor(buf, dtype=torch.long)
