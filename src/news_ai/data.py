"""Dataset adapters, label verification, deduplication, and group construction."""

from pathlib import Path
import shutil

import pandas as pd

from .text import fingerprint, normalize_text, word_count

DATASETS = {
    "isot": "clmentbisaillon/fake-and-real-news-dataset",
    "welfake": "saurabhshahane/fake-news-classification",
}
LABELS = {0: "real", 1: "fake"}


def download_datasets(destination: Path) -> None:
    import kagglehub

    destination.mkdir(parents=True, exist_ok=True)
    for name, handle in DATASETS.items():
        cache = Path(kagglehub.dataset_download(handle))
        filenames = ["Fake.csv", "True.csv"] if name == "isot" else ["WELFake_Dataset.csv"]
        for filename in filenames:
            matches = list(cache.rglob(filename))
            if len(matches) != 1:
                raise ValueError(f"Expected one {filename} in {cache}; found {len(matches)}.")
            target = destination / filename
            if target.exists():
                print(f"Already exists, preserved: {target}", flush=True)
                continue
            shutil.copy2(matches[0], target)
            print(f"Saved: {target}", flush=True)


def _read(path: Path, source: str, label: int | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Use news-ai download or --data-dir.")
    fields = ["title", "text"] + (["label"] if label is None else [])
    frame = pd.read_csv(path, usecols=fields, dtype={"title": str, "text": str})
    frame[["title", "text"]] = frame[["title", "text"]].fillna("")
    if label is not None:
        frame["label"] = label
    else:
        if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{path.name}: labels must be exactly 0 or 1, without missing values.")
        frame["label"] = frame["label"].astype(int)
    frame["source"] = source
    frame["row_id"] = [f"{path.name}:{index}" for index in frame.index]
    frame["title"] = frame["title"].map(normalize_text)
    frame["text"] = frame["text"].map(normalize_text)
    frame["body_hash"] = frame["text"].map(fingerprint)
    return frame


def verify_welfake_labels(isot: pd.DataFrame, welfake: pd.DataFrame, fake_label: str) -> dict:
    """Use known Fake.csv/True.csv labels as empirical anchors, not webpage claims."""
    anchors = isot[isot["text"].map(word_count) >= 20]
    consistent = anchors.groupby("body_hash")["label"].nunique()
    anchors = anchors[anchors["body_hash"].isin(consistent[consistent == 1].index)]
    anchors = anchors.drop_duplicates("body_hash")[["body_hash", "label"]]
    overlap = welfake.merge(anchors, on="body_hash", suffixes=("_raw", "_known"))
    agreement = float((overlap["label_raw"] == overlap["label_known"]).mean()) if len(overlap) else None
    if fake_label == "auto":
        if len(overlap) < 50 or agreement is None or max(agreement, 1 - agreement) < 0.95:
            raise ValueError("Cannot verify WELFake labels reliably. Inspect data and set --welfake-fake-label 0 or 1.")
        selected = 1 if agreement >= 0.95 else 0
    else:
        selected = int(fake_label)
        if len(overlap) >= 50:
            selected_agreement = agreement if selected == 1 else 1 - agreement
            if selected_agreement < 0.95:
                raise ValueError("Explicit WELFake mapping conflicts with Fake.csv/True.csv overlap.")
    return {
        "raw_fake_label": selected,
        "raw_label_mapping": {str(selected): "fake", str(1 - selected): "real"},
        "overlap_rows_used": len(overlap),
        "raw_equals_canonical_agreement": agreement,
        "method": "normalized body matches against Fake.csv/True.csv",
    }


def make_groups(frame: pd.DataFrame) -> list[str]:
    """Connect equal substantial titles or body prefixes before splitting."""
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[str, int] = {}
    for index, row in enumerate(frame.itertuples()):
        keys = [f"body:{row.body_hash}"]
        if word_count(row.title) >= 5 and len(row.title) >= 30:
            keys.append(f"title:{fingerprint(row.title)}")
        # Catches identical opening text with different headlines or added tails.
        words = normalize_text(row.text).casefold().split()
        if len(words) >= 80:
            keys.append(f"prefix:{fingerprint(' '.join(words[:80]))}")
        for key in keys:
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    return [f"group:{find(index)}" for index in range(len(frame))]


def prepare_data(data_dir: Path, fake_label: str = "auto") -> tuple[pd.DataFrame, dict]:
    print("Reading three CSV files...", flush=True)
    isot = pd.concat([
        _read(data_dir / "Fake.csv", "isot", 1),
        _read(data_dir / "True.csv", "isot", 0),
    ], ignore_index=True)
    welfake = _read(data_dir / "WELFake_Dataset.csv", "welfake", None)
    mapping = verify_welfake_labels(isot, welfake, fake_label)
    welfake["label"] = (welfake["label"] == mapping["raw_fake_label"]).astype(int)
    frame = pd.concat([isot, welfake], ignore_index=True)
    audit = {
        "raw_rows": len(frame),
        "raw_rows_by_source": frame["source"].value_counts().to_dict(),
        "canonical_labels": LABELS,
        "welfake_label_verification": mapping,
        "min_body_words": 20,
    }
    enough = frame["text"].map(word_count) >= 20
    audit["empty_or_short_body_rows_removed"] = int((~enough).sum())
    frame = frame[enough].copy()
    conflicts = frame.groupby("body_hash")["label"].nunique()
    conflict_keys = conflicts[conflicts > 1].index
    audit["conflicting_body_groups_removed"] = len(conflict_keys)
    audit["conflicting_label_rows_removed"] = int(frame["body_hash"].isin(conflict_keys).sum())
    frame = frame[~frame["body_hash"].isin(conflict_keys)].copy()
    # Preserve provenance for articles shared by both datasets.
    provenance = frame.groupby("body_hash")["source"].agg(lambda values: "|".join(sorted(set(values))))
    cross_source = frame.groupby("body_hash")["source"].nunique()
    audit["shared_unique_bodies_between_datasets"] = int((cross_source > 1).sum())
    before = len(frame)
    frame = frame.drop_duplicates("body_hash").reset_index(drop=True)
    audit["duplicate_body_rows_removed"] = before - len(frame)
    frame["sources"] = frame["body_hash"].map(provenance)
    frame["group_id"] = make_groups(frame)
    audit["clean_rows"] = len(frame)
    audit["clean_label_counts"] = {LABELS[int(k)]: int(v) for k, v in frame["label"].value_counts().items()}
    audit["clean_provenance_counts"] = frame["sources"].value_counts().to_dict()
    audit["groups"] = frame["group_id"].nunique()
    audit["largest_group_rows"] = int(frame.groupby("group_id").size().max())
    audit["near_duplicate_policy"] = "normalized title (>=5 words, >=30 chars) or first 80 body words; connected groups"
    print(f"Clean articles: {len(frame):,}; groups: {audit['groups']:,}", flush=True)
    return frame, audit
