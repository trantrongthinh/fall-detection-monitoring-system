import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_DIM = 52
DEFAULT_FALL_RATIO_THRESHOLD = 0.3


def csv_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = value.split(",")
    return [str(item).strip().lower() for item in items if str(item).strip()]


def is_copied_mcfd_urfd_row(row: dict) -> bool:
    if row.get("dataset", "").lower() != "urfd":
        return False

    record_id = row.get("record_id", "").lower()
    video_name = row.get("video_name", "").lower()
    video_path = row.get("video_path", "").lower()
    return (
        record_id.startswith("urfd_mcfd_")
        or video_name.startswith("mcfd_")
        or "\\mcfd_" in video_path
        or "/mcfd_" in video_path
    )


@dataclass
class SequenceBuildConfig:
    cache_dir: Path
    sequence_len: int
    stride: int | None = None
    train_fall_stride: int | None = None
    fall_ratio_threshold: float = DEFAULT_FALL_RATIO_THRESHOLD
    train_datasets: tuple[str, ...] = ("gmdcsa24", "urfd", "leifall")
    val_datasets: tuple[str, ...] = ("gmdcsa24", "urfd", "leifall")
    test_datasets: tuple[str, ...] = ("gmdcsa24", "urfd", "leifall")
    train_splits: tuple[str, ...] = ("train",)
    val_splits: tuple[str, ...] = ("val",)
    test_splits: tuple[str, ...] = ("test",)
    output_dir: Path | None = None

    def resolved_stride(self) -> int:
        return self.stride if self.stride is not None else max(self.sequence_len // 2, 1)


def load_manifest(cache_dir: Path) -> list[dict]:
    manifest_path = cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.csv in {cache_dir}")

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_matches(row: dict, datasets: set[str], splits: set[str]) -> bool:
    if is_copied_mcfd_urfd_row(row):
        return False
    return row.get("dataset", "").lower() in datasets and row.get("split", "").lower() in splits


def cache_path(cache_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else cache_dir / path


def make_windows(
    features: np.ndarray,
    frame_labels: np.ndarray,
    sequence_len: int,
    stride: int,
    fall_ratio_threshold: float,
    fall_stride: int | None = None,
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Features must have shape (frames, {FEATURE_DIM}), got {features.shape}")
    if frame_labels.ndim != 1:
        raise ValueError(f"Frame labels must be 1D, got {frame_labels.shape}")
    if len(features) != len(frame_labels):
        raise ValueError(f"Feature/label length mismatch: {len(features)} vs {len(frame_labels)}")

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    sequence_rows: list[dict] = []

    max_start = len(features) - sequence_len
    if max_start < 0:
        return X_rows, y_rows, sequence_rows

    base_starts = set(range(0, max_start + 1, stride))
    if fall_stride is not None and fall_stride > 0 and fall_stride < stride:
        candidate_starts = sorted(base_starts | set(range(0, max_start + 1, fall_stride)))
    else:
        candidate_starts = sorted(base_starts)

    for start in candidate_starts:
        end = start + sequence_len
        labels = frame_labels[start:end]
        fall_ratio = float(np.mean(labels == 1))
        sequence_label = int(fall_ratio >= fall_ratio_threshold)
        if start not in base_starts and sequence_label == 0:
            continue

        X_rows.append(features[start:end].astype(np.float32, copy=False))
        y_rows.append(sequence_label)
        sequence_rows.append(
            {
                "start_index": start,
                "end_index": end - 1,
                "fall_ratio": round(fall_ratio, 6),
                "sequence_label": sequence_label,
            }
        )

    return X_rows, y_rows, sequence_rows


def build_split(
    cache_dir: Path,
    rows: list[dict],
    split_name: str,
    datasets: set[str],
    splits: set[str],
    config: SequenceBuildConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    selected_rows = [row for row in rows if row_matches(row, datasets, splits)]
    stride = config.resolved_stride()
    fall_stride = config.train_fall_stride if split_name == "train" else None

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    sequence_manifest: list[dict] = []

    for row in selected_rows:
        features = np.load(cache_path(cache_dir, row["features_path"]))
        labels = np.load(cache_path(cache_dir, row["labels_path"]))
        windows, window_labels, window_rows = make_windows(
            features=features,
            frame_labels=labels.astype(np.int64, copy=False),
            sequence_len=config.sequence_len,
            stride=stride,
            fall_ratio_threshold=config.fall_ratio_threshold,
            fall_stride=fall_stride,
        )

        all_X.extend(windows)
        all_y.extend(window_labels)

        for window_row in window_rows:
            manifest_row = {
                "split": split_name,
                "dataset": row.get("dataset", ""),
                "source_split": row.get("split", ""),
                "record_id": row.get("record_id", ""),
                "video_path": row.get("video_path", ""),
                "video_name": row.get("video_name", ""),
                **window_row,
            }
            sequence_manifest.append(manifest_row)

    if not all_X:
        raise RuntimeError(
            f"No sequences built for {split_name}. "
            f"datasets={sorted(datasets)}, splits={sorted(splits)}, sequence_len={config.sequence_len}"
        )

    X = np.stack(all_X).astype(np.float32, copy=False)
    y = np.array(all_y, dtype=np.int64)
    summary = {
        "shape": list(X.shape),
        "samples": int(len(y)),
        "normal": int(np.sum(y == 0)),
        "fall": int(np.sum(y == 1)),
        "source_videos": int(len(selected_rows)),
        "datasets": sorted(datasets),
        "source_splits": sorted(splits),
        "stride": int(stride),
        "fall_stride": int(fall_stride) if fall_stride is not None else None,
        "nan": int(np.isnan(X).sum()),
        "inf": int(np.isinf(X).sum()),
    }
    return X, y, sequence_manifest, summary


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_dataset(output_dir: Path, arrays: dict, summary: dict, sequence_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        X, y = arrays[split_name]
        np.save(output_dir / f"X_{split_name}.npy", X)
        np.save(output_dir / f"y_{split_name}.npy", y)
        print(
            f"[SAVE] {split_name}: X={X.shape}, y={y.shape}, "
            f"normal={summary['splits'][split_name]['normal']}, fall={summary['splits'][split_name]['fall']}"
        )

    write_manifest(output_dir / "sequence_manifest.csv", sequence_rows)
    with (output_dir / "sequence_build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def build_arrays_from_cache(config: SequenceBuildConfig) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    cache_dir = Path(config.cache_dir)
    manifest_rows = load_manifest(cache_dir)

    split_specs = {
        "train": (set(csv_list(config.train_datasets)), set(csv_list(config.train_splits))),
        "val": (set(csv_list(config.val_datasets)), set(csv_list(config.val_splits))),
        "test": (set(csv_list(config.test_datasets)), set(csv_list(config.test_splits))),
    }

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sequence_rows: list[dict] = []
    summary = {
        "cache_dir": str(cache_dir),
        "sequence_len": int(config.sequence_len),
        "stride": int(config.resolved_stride()),
        "train_fall_stride": int(config.train_fall_stride) if config.train_fall_stride is not None else None,
        "fall_ratio_threshold": float(config.fall_ratio_threshold),
        "splits": {},
    }

    for split_name, (datasets, splits) in split_specs.items():
        X, y, rows, split_summary = build_split(
            cache_dir=cache_dir,
            rows=manifest_rows,
            split_name=split_name,
            datasets=datasets,
            splits=splits,
            config=config,
        )
        arrays[split_name] = (X, y)
        sequence_rows.extend(rows)
        summary["splits"][split_name] = split_summary

    if config.output_dir is not None:
        save_dataset(Path(config.output_dir), arrays, summary, sequence_rows)

    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]
    return X_train, y_train, X_val, y_val, X_test, y_test, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sequence npy files from frame-level feature cache.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-len", type=int, required=True)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--train-fall-stride", type=int, default=None)
    parser.add_argument("--fall-ratio-threshold", type=float, default=DEFAULT_FALL_RATIO_THRESHOLD)
    parser.add_argument("--train-datasets", default="gmdcsa24,urfd,leifall")
    parser.add_argument("--val-datasets", default="gmdcsa24,urfd,leifall")
    parser.add_argument("--test-datasets", default="gmdcsa24,urfd,leifall")
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--val-splits", default="val")
    parser.add_argument("--test-splits", default="test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SequenceBuildConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        sequence_len=args.sequence_len,
        stride=args.stride,
        train_fall_stride=args.train_fall_stride,
        fall_ratio_threshold=args.fall_ratio_threshold,
        train_datasets=tuple(csv_list(args.train_datasets)),
        val_datasets=tuple(csv_list(args.val_datasets)),
        test_datasets=tuple(csv_list(args.test_datasets)),
        train_splits=tuple(csv_list(args.train_splits)),
        val_splits=tuple(csv_list(args.val_splits)),
        test_splits=tuple(csv_list(args.test_splits)),
    )
    build_arrays_from_cache(config)


if __name__ == "__main__":
    main()
