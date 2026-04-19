from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from src.utils.io import dump_json, ensure_dir


CLICK_COLUMNS = ["session_id", "timestamp", "item_id", "category"]
BUY_COLUMNS = ["session_id", "timestamp", "item_id", "price", "quantity"]


@dataclass
class DatasetSplit:
    sequences: List[List[int]]
    targets: List[int]
    session_ids: List[int]

    def to_dict(self) -> Dict[str, List[int] | List[List[int]]]:
        return {
            "sequences": self.sequences,
            "targets": self.targets,
            "session_ids": self.session_ids,
        }


def _load_events(
    path: str | Path,
    columns: Sequence[str],
    max_rows: int | None = None,
    fraction: float | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    step = None
    if fraction is not None and fraction < 1.0:
        step = round(1 / fraction)

    reader = pd.read_csv(
        path,
        names=columns,
        usecols=[0, 1, 2],
        nrows=max_rows,
        parse_dates=["timestamp"],
        dtype={"session_id": "int64", "item_id": "int64"},
        chunksize=chunksize if max_rows is None else None,
    )

    if isinstance(reader, pd.DataFrame):
        frame = reader
        if step is not None:
            frame = frame[frame["session_id"] % step == 0]
        return frame.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    chunks = []
    for chunk in reader:
        if step is not None:
            chunk = chunk[chunk["session_id"] % step == 0]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["session_id", "timestamp", "item_id"])
    return pd.concat(chunks, ignore_index=True).sort_values(["session_id", "timestamp"]).reset_index(drop=True)


def load_clicks(clicks_path: str | Path, max_rows: int | None = None, fraction: float | None = None) -> pd.DataFrame:
    return _load_events(clicks_path, CLICK_COLUMNS, max_rows=max_rows, fraction=fraction)


def load_buys(buys_path: str | Path, max_rows: int | None = None, fraction: float | None = None) -> pd.DataFrame:
    return _load_events(buys_path, BUY_COLUMNS, max_rows=max_rows, fraction=fraction)


def merge_clicks_and_buys(clicks: pd.DataFrame, buys: pd.DataFrame) -> pd.DataFrame:
    clicks = clicks.copy()
    buys = buys.copy()
    clicks["event_type"] = "click"
    buys["event_type"] = "buy"
    merged = pd.concat([clicks, buys], ignore_index=True)
    return merged.sort_values(["session_id", "timestamp", "event_type"]).reset_index(drop=True)


def apply_session_sampling(frame: pd.DataFrame, fraction: float | None = None) -> pd.DataFrame:
    if fraction is None or fraction >= 1.0:
        return frame
    if fraction <= 0:
        raise ValueError("Sampling fraction must be positive.")

    step = round(1 / fraction)
    session_ids = frame["session_id"].drop_duplicates()
    keep_ids = session_ids[session_ids % step == 0]
    sampled = frame[frame["session_id"].isin(set(keep_ids))]
    return sampled.reset_index(drop=True)


def filter_sessions_and_items(
    frame: pd.DataFrame,
    min_session_length: int,
    min_item_support: int,
) -> pd.DataFrame:
    filtered = frame.copy()

    while True:
        session_lengths = filtered.groupby("session_id").size()
        keep_sessions = session_lengths[session_lengths >= min_session_length].index
        filtered = filtered[filtered["session_id"].isin(keep_sessions)]

        item_support = filtered["item_id"].value_counts()
        keep_items = item_support[item_support >= min_item_support].index
        updated = filtered[filtered["item_id"].isin(keep_items)]

        if len(updated) == len(filtered):
            break
        filtered = updated

    return filtered.sort_values(["session_id", "timestamp"]).reset_index(drop=True)


def compute_split_cutoffs(frame: pd.DataFrame, validation_days: int, test_days: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    session_end = frame.groupby("session_id")["timestamp"].max()
    max_time = session_end.max()
    test_from = max_time - pd.Timedelta(days=test_days)
    validation_from = test_from - pd.Timedelta(days=validation_days)
    return validation_from, test_from


def temporal_session_split(
    frame: pd.DataFrame,
    validation_days: int,
    test_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_from, test_from = compute_split_cutoffs(frame, validation_days, test_days)
    session_end = frame.groupby("session_id")["timestamp"].max().rename("session_end")
    tagged = frame.merge(session_end, on="session_id")

    train = tagged[tagged["session_end"] < validation_from]
    validation = tagged[(tagged["session_end"] >= validation_from) & (tagged["session_end"] < test_from)]
    test = tagged[tagged["session_end"] >= test_from]

    return (
        train.drop(columns=["session_end"]).reset_index(drop=True),
        validation.drop(columns=["session_end"]).reset_index(drop=True),
        test.drop(columns=["session_end"]).reset_index(drop=True),
    )


def encode_items(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    drop_unseen_items: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int]]:
    train_items = sorted(train["item_id"].unique().tolist())
    encoder = {item_id: index + 1 for index, item_id in enumerate(train_items)}
    decoder = {index: item_id for item_id, index in encoder.items()}
    decoder[0] = 0

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        mapped = frame.copy()
        mapped["item_id"] = mapped["item_id"].map(encoder)
        if drop_unseen_items:
            mapped = mapped.dropna(subset=["item_id"])
        mapped["item_id"] = mapped["item_id"].astype("int64")
        session_lengths = mapped.groupby("session_id").size()
        valid_sessions = session_lengths[session_lengths >= 2].index
        return mapped[mapped["session_id"].isin(valid_sessions)].reset_index(drop=True)

    return _apply(train), _apply(validation), _apply(test), encoder, decoder


def frame_to_examples(frame: pd.DataFrame) -> DatasetSplit:
    sequences: List[List[int]] = []
    targets: List[int] = []
    session_ids: List[int] = []

    for session_id, group in frame.groupby("session_id", sort=False):
        items = group["item_id"].tolist()
        for index in range(1, len(items)):
            prefix = items[:index]
            target = items[index]
            sequences.append(prefix)
            targets.append(target)
            session_ids.append(int(session_id))

    return DatasetSplit(sequences=sequences, targets=targets, session_ids=session_ids)


def save_split(split: DatasetSplit, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("wb") as handle:
        pickle.dump(split.to_dict(), handle)


def preprocessing_summary(
    variant: str,
    include_buys: bool,
    raw_frame: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_examples: DatasetSplit,
    validation_examples: DatasetSplit,
    test_examples: DatasetSplit,
    encoder: Dict[int, int],
) -> Dict[str, object]:
    return {
        "variant": variant,
        "include_buys": include_buys,
        "raw_events": int(len(raw_frame)),
        "train_events": int(len(train)),
        "validation_events": int(len(validation)),
        "test_events": int(len(test)),
        "train_sessions": int(train["session_id"].nunique()),
        "validation_sessions": int(validation["session_id"].nunique()),
        "test_sessions": int(test["session_id"].nunique()),
        "train_examples": len(train_examples.targets),
        "validation_examples": len(validation_examples.targets),
        "test_examples": len(test_examples.targets),
        "num_items": len(encoder),
        "max_train_sequence_length": max((len(seq) for seq in train_examples.sequences), default=0),
    }


def save_processed_dataset(
    output_dir: str | Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_examples: DatasetSplit,
    validation_examples: DatasetSplit,
    test_examples: DatasetSplit,
    summary: Dict[str, object],
    encoder: Dict[int, int],
    decoder: Dict[int, int],
) -> None:
    output_dir = ensure_dir(output_dir)
    train_frame.to_parquet(output_dir / "train_events.parquet", index=False)
    validation_frame.to_parquet(output_dir / "validation_events.parquet", index=False)
    test_frame.to_parquet(output_dir / "test_events.parquet", index=False)

    save_split(train_examples, output_dir / "train_examples.pkl")
    save_split(validation_examples, output_dir / "validation_examples.pkl")
    save_split(test_examples, output_dir / "test_examples.pkl")

    with (output_dir / "item_encoder.pkl").open("wb") as handle:
        pickle.dump(encoder, handle)
    with (output_dir / "item_decoder.pkl").open("wb") as handle:
        pickle.dump(decoder, handle)

    dump_json(summary, output_dir / "metadata.json")


def preprocess_from_config(config: Dict[str, object]) -> Dict[str, object]:
    source = config["source"]
    output = config["output"]
    sampling = config.get("sampling", {})
    filtering = config["filtering"]
    split = config["split"]

    clicks = load_clicks(
        clicks_path=source["clicks_path"],
        max_rows=sampling.get("max_rows"),
        fraction=sampling.get("fraction"),
    )
    raw_frame = clicks

    include_buys = bool(output.get("include_buys", False))
    if include_buys:
        buys = load_buys(
            source["buys_path"],
            max_rows=sampling.get("max_rows"),
            fraction=sampling.get("fraction"),
        )
        raw_frame = merge_clicks_and_buys(raw_frame, buys)

    filtered = filter_sessions_and_items(
        frame=raw_frame,
        min_session_length=filtering["min_session_length"],
        min_item_support=filtering["min_item_support"],
    )

    train, validation, test = temporal_session_split(
        frame=filtered,
        validation_days=split["validation_days"],
        test_days=split["test_days"],
    )
    train, validation, test, encoder, decoder = encode_items(
        train=train,
        validation=validation,
        test=test,
        drop_unseen_items=split.get("drop_unseen_items", True),
    )

    train_examples = frame_to_examples(train)
    validation_examples = frame_to_examples(validation)
    test_examples = frame_to_examples(test)

    summary = preprocessing_summary(
        variant=config["variant"],
        include_buys=include_buys,
        raw_frame=raw_frame,
        train=train,
        validation=validation,
        test=test,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        encoder=encoder,
    )
    save_processed_dataset(
        output_dir=output["processed_dir"],
        train_frame=train,
        validation_frame=validation,
        test_frame=test,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        summary=summary,
        encoder=encoder,
        decoder=decoder,
    )

    return summary
