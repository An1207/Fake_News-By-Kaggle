from pathlib import Path

import pandas as pd
import pytest

from news_ai.data import make_groups, prepare_data
from news_ai.model import split_data
from news_ai.text import fingerprint


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    fake, real = [], []
    for label, target in [(1, fake), (0, real)]:
        for index in range(40):
            phrase = "fabricated conspiracy shocking rumor imaginary" if label else "official government confirmed report published"
            target.append({
                "title": f"Unique news headline for event category {label} number {index}",
                "text": f"Story identifier {label} {index}. " + (phrase + " ") * 7,
            })
    pd.DataFrame(fake).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame(real).to_csv(tmp_path / "True.csv", index=False)
    pd.DataFrame([dict(row, label=1) for row in fake] + [dict(row, label=0) for row in real]).to_csv(
        tmp_path / "WELFake_Dataset.csv", index=False,
    )
    return tmp_path


def test_label_verification_deduplication_and_provenance(datasets):
    frame, audit = prepare_data(datasets)
    assert audit["welfake_label_verification"]["raw_fake_label"] == 1
    assert audit["welfake_label_verification"]["overlap_rows_used"] == 80
    assert len(frame) == 80
    assert audit["duplicate_body_rows_removed"] == 80
    assert frame["sources"].eq("isot|welfake").all()
    assert frame.loc[frame["label"] == 1, "text"].str.contains("fabricated").all()


def test_reversed_welfake_labels_are_inferred(datasets):
    path = datasets / "WELFake_Dataset.csv"
    raw = pd.read_csv(path)
    raw["label"] = 1 - raw["label"]
    raw.to_csv(path, index=False)
    frame, audit = prepare_data(datasets)
    assert audit["welfake_label_verification"]["raw_fake_label"] == 0
    assert frame.loc[frame["label"] == 1, "text"].str.contains("fabricated").all()
    with pytest.raises(ValueError, match="conflicts"):
        prepare_data(datasets, "1")


def test_conflicting_labels_removed_and_blank_body_not_learned(datasets):
    path = datasets / "WELFake_Dataset.csv"
    raw = pd.read_csv(path)
    raw.loc[0, "label"] = 0
    raw = pd.concat([raw, pd.DataFrame([{"title": "Empty body", "text": "", "label": 1}])])
    raw.to_csv(path, index=False)
    frame, audit = prepare_data(datasets)
    assert len(frame) == 79
    assert audit["conflicting_label_rows_removed"] == 2
    assert audit["empty_or_short_body_rows_removed"] == 1


def test_invalid_labels_fail_instead_of_silent_coercion(datasets):
    path = datasets / "WELFake_Dataset.csv"
    raw = pd.read_csv(path)
    raw.loc[0, "label"] = 7
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels"):
        prepare_data(datasets)


def test_groups_connect_titles_and_prefixes_transitively():
    opening = " ".join(f"word{number}" for number in range(80))
    frame = pd.DataFrame([
        {"title": "This is a sufficiently long shared headline", "text": "First body words"},
        {"title": "This is a sufficiently long shared headline", "text": opening + " tail one"},
        {"title": "Different long headline about the same event", "text": opening + " tail two"},
        {"title": "Another independent report about new events", "text": "Independent unique body"},
    ])
    frame["body_hash"] = frame["text"].map(fingerprint)
    groups = make_groups(frame)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_group_splits_are_disjoint_and_reproducible(datasets):
    frame, _ = prepare_data(datasets)
    # Introduce distinct article versions of existing groups.
    versions = frame.iloc[:20].copy()
    versions["body_hash"] = "version:" + versions["body_hash"]
    frame = pd.concat([frame, versions], ignore_index=True)
    partitions = split_data(frame)
    repeated = split_data(frame)
    assert sum(map(len, partitions)) == len(frame)
    for left, right in zip(partitions, repeated):
        assert left["body_hash"].tolist() == right["body_hash"].tolist()
        assert set(left["label"]) == {0, 1}
    for i, left in enumerate(partitions):
        for right in partitions[i + 1:]:
            assert not set(left["group_id"]) & set(right["group_id"])
