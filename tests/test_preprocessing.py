from __future__ import annotations

import pandas as pd

from src.data.preprocessing import (
    encode_items,
    filter_sessions_and_items,
    frame_to_examples,
    temporal_session_split,
)


def test_temporal_split_and_encoding_drop_unseen() -> None:
    frame = pd.DataFrame(
        {
            "session_id": [1, 1, 2, 2, 3, 3],
            "timestamp": pd.to_datetime(
                    [
                    "2014-09-26 00:00:01",
                    "2014-09-26 00:00:02",
                    "2014-09-27 00:00:01",
                    "2014-09-27 00:00:02",
                    "2014-09-29 00:00:01",
                    "2014-09-29 00:00:02",
                ]
            ),
            "item_id": [10, 11, 10, 12, 10, 99],
        }
    )

    train, validation, test = temporal_session_split(frame, validation_days=1, test_days=1)
    train, validation, test, encoder, _ = encode_items(train, validation, test, drop_unseen_items=True)

    assert set(train["item_id"].unique()) == {1, 2}
    assert 99 not in test["item_id"].tolist()


def test_filter_and_example_generation() -> None:
    frame = pd.DataFrame(
        {
            "session_id": [1, 1, 2, 3, 3, 3],
            "timestamp": pd.to_datetime(
                [
                    "2014-09-27 00:00:01",
                    "2014-09-27 00:00:02",
                    "2014-09-27 00:00:03",
                    "2014-09-28 00:00:01",
                    "2014-09-28 00:00:02",
                    "2014-09-28 00:00:03",
                ]
            ),
            "item_id": [10, 10, 10, 10, 11, 10],
        }
    )

    filtered = filter_sessions_and_items(frame, min_session_length=2, min_item_support=2)
    examples = frame_to_examples(filtered)

    assert filtered["session_id"].nunique() == 2
    assert len(examples.targets) == 2
