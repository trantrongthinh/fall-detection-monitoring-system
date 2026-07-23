import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fall_features import FEATURE_DIM, extract_yolo_pose_features


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "frame_feature_cache_v3"
DEFAULT_YOLO_MODEL = PROJECT_DIR / "yolo11n-pose.pt"
DEFAULT_URFD_ROOT = PROJECT_DIR / "datasets" / "videos"
DEFAULT_LEIFALL_ROOT = PROJECT_DIR / "datasets Lei2Fall"
DEFAULT_GMDCSA_ROOT = (
    PROJECT_DIR
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
DEFAULT_MCFD_ROOT = PROJECT_DIR / "datasets MCFD"
DEFAULT_MCFD_LABEL_FILE = PROJECT_DIR / "fall_labels.csv"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
TARGET_FPS = 10
IMG_SIZE = 640


@dataclass
class VideoItem:
    record_id: str
    dataset: str
    split: str
    class_name: str
    video_path: Path
    video_label: int
    label_mode: str
    start_frame: int | None = None
    end_frame: int | None = None
    fall_intervals: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def csv_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def slugify(text: str) -> str:
    text = text.lower().replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_")


def make_record_id(dataset: str, video_path: Path, extra: str = "") -> str:
    raw = f"{dataset}|{video_path}|{extra}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    stem = slugify(f"{dataset}_{video_path.stem}_{extra}")[:70]
    return f"{stem}_{digest}"


def select_device() -> int | str:
    import torch

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        return 0
    print("[WARN] No CUDA found. YOLO extraction will run on CPU.")
    return "cpu"


def extract_feature(result) -> np.ndarray:
    feature, _ = extract_yolo_pose_features(result, prefer_largest=True)
    return feature


def normalize_name(text: str) -> str:
    text = text.lower()
    text = text.replace("annotations_files_", "")
    text = text.replace("annotation_files_", "")
    text = text.replace("annotation_file_", "")
    text = text.replace("annotations_file_", "")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_video_number(filename: str) -> int | None:
    match = re.search(r"video\s*\((\d+)\)", filename.lower())
    return int(match.group(1)) if match else None


def parse_leifall_annotation(txt_path: Path) -> tuple[int | None, int | None]:
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) >= 3 and lines[0].isdigit() and lines[1].isdigit() and "," in lines[2]:
        return int(lines[0]), int(lines[1])
    return None, None


def build_leifall_video_index(video_root: Path) -> dict[str, list[Path]]:
    video_index: dict[str, list[Path]] = {}
    for path in video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            video_index.setdefault(path.name.lower(), []).append(path)
    return video_index


def find_leifall_video(txt_path: Path, video_index: dict[str, list[Path]]) -> Path | None:
    video_no = extract_video_number(txt_path.name)
    if video_no is None:
        return None

    expected_name = f"video ({video_no}).avi"
    candidates = video_index.get(expected_name.lower(), [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    ann_tokens = normalize_name(txt_path.parent.name).split()
    scored = []
    for video_path in candidates:
        folder_name = normalize_name(video_path.parent.name)
        score = sum(1 for token in ann_tokens if token in folder_name)
        scored.append((score, video_path))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored[0][0] > 0 else candidates[0]


def split_group(items: list[dict], rng: random.Random) -> tuple[list[dict], list[dict], list[dict]]:
    items = list(items)
    rng.shuffle(items)
    n_total = len(items)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def collect_urfd(root_dir: Path) -> list[VideoItem]:
    items: list[VideoItem] = []
    for split in ["train", "val", "test"]:
        split_dir = root_dir / split
        if not split_dir.exists():
            continue

        for label_name, label in [("fall", 1), ("non-fall", 0)]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue

            for video_path in sorted(label_dir.iterdir()):
                if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                if video_path.stem.lower().startswith("mcfd_"):
                    print(f"[SKIP] Ignoring copied MCFD video in URFD split: {video_path}")
                    continue
                items.append(
                    VideoItem(
                        record_id=make_record_id("urfd", video_path, split),
                        dataset="urfd",
                        split=split,
                        class_name=label_name,
                        video_path=video_path,
                        video_label=label,
                        label_mode="constant",
                    )
                )
    return items


def collect_leifall(root_dir: Path, seed: int) -> list[VideoItem]:
    video_root = root_dir / "Videos"
    ann_root = root_dir / "Annotation_files"
    video_index = build_leifall_video_index(video_root)

    raw_items = []
    missing = []
    for txt_path in sorted(ann_root.rglob("*.txt")):
        video_path = find_leifall_video(txt_path, video_index)
        if video_path is None:
            missing.append(str(txt_path.relative_to(root_dir)))
            continue
        fall_start, fall_end = parse_leifall_annotation(txt_path)
        has_fall = fall_start is not None and fall_end is not None
        raw_items.append(
            {
                "video_path": video_path,
                "fall_start": fall_start,
                "fall_end": fall_end,
                "has_fall": has_fall,
                "annotation_path": txt_path,
            }
        )

    if missing:
        print(f"[WARN] Lei2Fall missing videos: {len(missing)}")
        for name in missing[:10]:
            print(" -", name)

    rng = random.Random(seed)
    fall_items = [item for item in raw_items if item["has_fall"]]
    normal_items = [item for item in raw_items if not item["has_fall"]]
    fall_train, fall_val, fall_test = split_group(fall_items, rng)
    normal_train, normal_val, normal_test = split_group(normal_items, rng)

    split_map = {
        "train": fall_train + normal_train,
        "val": fall_val + normal_val,
        "test": fall_test + normal_test,
    }

    items: list[VideoItem] = []
    for split, split_items in split_map.items():
        rng.shuffle(split_items)
        for item in split_items:
            has_fall = bool(item["has_fall"])
            video_path = Path(item["video_path"])
            items.append(
                VideoItem(
                    record_id=make_record_id(
                        "leifall",
                        video_path,
                        f"{split}_{Path(item['annotation_path']).stem}",
                    ),
                    dataset="leifall",
                    split=split,
                    class_name="fall" if has_fall else "non-fall",
                    video_path=video_path,
                    video_label=int(has_fall),
                    label_mode="frame_interval" if has_fall else "constant",
                    start_frame=item["fall_start"],
                    end_frame=item["fall_end"],
                    metadata={"annotation_path": str(item["annotation_path"])},
                )
            )
    return items


def parse_gmdcsa_intervals(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    intervals_by_file: dict[str, list[tuple[float, float]]] = {}
    if not csv_path.exists():
        return intervals_by_file

    pattern = re.compile(
        r"fall\w*[^[]*\[\s*(\d+(?:\.\d+)?)\s*(?:to\s*)?(\d+(?:\.\d+)?)\s*\]",
        re.IGNORECASE,
    )

    for line in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        if not line.strip():
            continue
        filename = line.split(",", 1)[0].strip()
        intervals = []
        for start_text, end_text in pattern.findall(line):
            start, end = float(start_text), float(end_text)
            if end < start:
                start, end = end, start
            intervals.append((start, end))
        intervals_by_file[filename.lower()] = intervals
    return intervals_by_file


def collect_gmdcsa(
    root_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> list[VideoItem]:
    """Collect GMDCSA-24 and split by subject into train/val/test.

    Splitting by subject is safer than random video splitting because videos from
    the same subject should not appear in both train and evaluation sets.
    """
    records = []
    for subject_dir in sorted(root_dir.glob("Subject *")):
        if not subject_dir.is_dir():
            continue

        adl_dir = subject_dir / "ADL"
        for video_path in sorted(adl_dir.iterdir()) if adl_dir.exists() else []:
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append((subject_dir.name, "ADL", video_path, 0, []))

        fall_dir = subject_dir / "Fall"
        fall_intervals = parse_gmdcsa_intervals(subject_dir / "Fall.csv")
        for video_path in sorted(fall_dir.iterdir()) if fall_dir.exists() else []:
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append(
                    (
                        subject_dir.name,
                        "Fall",
                        video_path,
                        1,
                        fall_intervals.get(video_path.name.lower(), []),
                    )
                )

    subjects = sorted({record[0] for record in records})
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    if n_subjects <= 2:
        n_test = 0
        n_val = 0
    else:
        n_test = max(1, int(round(n_subjects * test_ratio))) if test_ratio > 0 else 0
        n_val = max(1, int(round(n_subjects * val_ratio))) if val_ratio > 0 else 0

        # Keep at least one subject for training when possible.
        if n_test + n_val >= n_subjects:
            n_val = max(0, n_subjects - n_test - 1)
        if n_test >= n_subjects:
            n_test = max(0, n_subjects - 1)

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])

    print("[GMDCSA SPLIT BY SUBJECT]")
    print("  train subjects:", sorted(train_subjects))
    print("  val subjects  :", sorted(val_subjects))
    print("  test subjects :", sorted(test_subjects))

    items: list[VideoItem] = []
    for subject, class_name, video_path, label, intervals in records:
        if subject in test_subjects:
            split = "test"
        elif subject in val_subjects:
            split = "val"
        else:
            split = "train"

        items.append(
            VideoItem(
                record_id=make_record_id("gmdcsa24", video_path, split),
                dataset="gmdcsa24",
                split=split,
                class_name=class_name.lower(),
                video_path=video_path,
                video_label=label,
                label_mode="time_intervals" if intervals else "constant",
                fall_intervals=intervals,
                metadata={"subject": subject},
            )
        )
    return items


def collect_mcfd(root_dir: Path, label_file: Path) -> list[VideoItem]:
    video_paths = sorted(path for path in root_dir.rglob("*.avi") if path.is_file())
    if not label_file.exists():
        raise FileNotFoundError(f"MCFD label file not found: {label_file}")

    with label_file.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(video_paths) != len(rows):
        print(f"[WARN] MCFD videos ({len(video_paths)}) != label rows ({len(rows)}). Matching by sorted index.")

    items: list[VideoItem] = []
    for index, video_path in enumerate(video_paths[: len(rows)]):
        row = rows[index]
        start_frame = int(float(row["start_frame"]))
        end_frame = int(float(row["end_frame"]))
        chute_name = video_path.parent.parent.name if video_path.parent.parent else video_path.parent.name
        items.append(
            VideoItem(
                record_id=make_record_id("mcfd", video_path, f"demo_{index}_{chute_name}"),
                dataset="mcfd",
                split="demo",
                class_name="chute",
                video_path=video_path,
                video_label=1,
                label_mode="frame_interval",
                start_frame=start_frame,
                end_frame=end_frame,
                metadata={
                    "label_video_name": row.get("video_name", ""),
                    "chute": chute_name,
                    "label_index": index,
                },
            )
        )
    return items


def label_for_frame(item: VideoItem, frame_index: int, timestamp: float) -> int:
    if item.label_mode == "constant":
        return int(item.video_label)

    if item.label_mode == "frame_interval":
        if item.start_frame is None or item.end_frame is None:
            return int(item.video_label)
        return int(item.start_frame <= frame_index <= item.end_frame)

    if item.label_mode == "time_intervals":
        if not item.fall_intervals:
            return int(item.video_label)
        return int(any(start <= timestamp <= end for start, end in item.fall_intervals))

    raise ValueError(f"Unknown label_mode: {item.label_mode}")


def extract_video(
    item: VideoItem,
    model,
    device: int | str,
    output_dir: Path,
    target_fps: int,
    image_size: int,
    skip_existing: bool,
) -> dict:
    import cv2

    features_dir = output_dir / "features"
    labels_dir = output_dir / "labels"
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    features_path = features_dir / f"{item.record_id}.npy"
    labels_path = labels_dir / f"{item.record_id}.npy"

    if skip_existing and features_path.exists() and labels_path.exists():
        features = np.load(features_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        used_frames = int(len(labels))
        fall_frames = int(np.sum(labels == 1))
        return make_manifest_row(item, features_path, labels_path, output_dir, used_frames, fall_frames, "cached")

    cap = cv2.VideoCapture(str(item.video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {item.video_path}")
        return make_manifest_row(item, features_path, labels_path, output_dir, 0, 0, "failed_open")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0
    frame_interval = max(int(original_fps / target_fps), 1)

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        if frame_index % frame_interval != 0:
            continue

        timestamp = (frame_index - 1) / original_fps
        try:
            results = model(frame, verbose=False, device=device, imgsz=image_size)
            feature = extract_feature(results[0]) if results else np.zeros(FEATURE_DIM, dtype=np.float32)
        except Exception as exc:
            print(f"[WARN] YOLO failed at {item.video_path.name} frame {frame_index}: {exc}")
            feature = np.zeros(FEATURE_DIM, dtype=np.float32)

        features_list.append(feature)
        labels_list.append(label_for_frame(item, frame_index, timestamp))

    cap.release()

    features = np.array(features_list, dtype=np.float32).reshape((-1, FEATURE_DIM))
    labels = np.array(labels_list, dtype=np.int64)
    np.save(features_path, features)
    np.save(labels_path, labels)

    used_frames = int(len(labels))
    fall_frames = int(np.sum(labels == 1))
    return make_manifest_row(item, features_path, labels_path, output_dir, used_frames, fall_frames, "ok")


def rel_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def make_manifest_row(
    item: VideoItem,
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
    used_frames: int,
    fall_frames: int,
    status: str,
) -> dict:
    row = {
        "record_id": item.record_id,
        "dataset": item.dataset,
        "split": item.split,
        "class_name": item.class_name,
        "video_path": str(item.video_path),
        "video_name": item.video_path.name,
        "video_label": int(item.video_label),
        "label_mode": item.label_mode,
        "start_frame": "" if item.start_frame is None else int(item.start_frame),
        "end_frame": "" if item.end_frame is None else int(item.end_frame),
        "fall_intervals": json.dumps(item.fall_intervals),
        "features_path": rel_to(features_path, output_dir),
        "labels_path": rel_to(labels_path, output_dir),
        "frames": int(used_frames),
        "fall_frames": int(fall_frames),
        "normal_frames": int(used_frames - fall_frames),
        "status": status,
    }
    for key, value in item.metadata.items():
        row[f"meta_{key}"] = value
    return row


def collect_items(args) -> list[VideoItem]:
    datasets = set(csv_list(args.datasets))
    all_items: list[VideoItem] = []

    if "urfd" in datasets:
        items = collect_urfd(Path(args.urfd_root))
        print(f"[COLLECT] URFD/base videos: {len(items)}")
        all_items.extend(items)

    if "leifall" in datasets:
        items = collect_leifall(Path(args.leifall_root), args.seed)
        print(f"[COLLECT] Lei2Fall videos: {len(items)}")
        all_items.extend(items)

    if "gmdcsa24" in datasets:
        items = collect_gmdcsa(
            Path(args.gmdcsa_root),
            args.seed,
            args.gmdcsa_val_ratio,
            args.gmdcsa_test_ratio,
        )
        print(f"[COLLECT] GMDCSA24 videos: {len(items)}")
        all_items.extend(items)

    if "mcfd" in datasets:
        items = collect_mcfd(Path(args.mcfd_root), Path(args.mcfd_label_file))
        print(f"[COLLECT] MCFD videos: {len(items)}")
        all_items.extend(items)

    if args.max_videos_per_dataset is not None:
        limited: list[VideoItem] = []
        for dataset in sorted({item.dataset for item in all_items}):
            limited.extend([item for item in all_items if item.dataset == dataset][: args.max_videos_per_dataset])
        all_items = limited

    return all_items


def write_manifest(output_dir: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], args) -> dict:
    summary = {
        "output_dir": str(args.output_dir),
        "datasets": csv_list(args.datasets),
        "target_fps": int(args.target_fps),
        "img_size": int(args.img_size),
        "videos": int(len(rows)),
        "by_dataset_split": {},
    }

    for row in rows:
        key = f"{row['dataset']}:{row['split']}"
        entry = summary["by_dataset_split"].setdefault(
            key,
            {
                "videos": 0,
                "frames": 0,
                "normal_frames": 0,
                "fall_frames": 0,
                "failed": 0,
            },
        )
        entry["videos"] += 1
        entry["frames"] += int(row["frames"])
        entry["normal_frames"] += int(row["normal_frames"])
        entry["fall_frames"] += int(row["fall_frames"])
        if row["status"] != "ok" and row["status"] != "cached":
            entry["failed"] += 1

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frame-level 52-D YOLO pose features once for all datasets.")
    parser.add_argument("--datasets", default="urfd,leifall,gmdcsa24,mcfd")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--urfd-root", type=Path, default=DEFAULT_URFD_ROOT)
    parser.add_argument("--leifall-root", type=Path, default=DEFAULT_LEIFALL_ROOT)
    parser.add_argument("--gmdcsa-root", type=Path, default=DEFAULT_GMDCSA_ROOT)
    parser.add_argument("--mcfd-root", type=Path, default=DEFAULT_MCFD_ROOT)
    parser.add_argument("--mcfd-label-file", type=Path, default=DEFAULT_MCFD_LABEL_FILE)
    parser.add_argument("--target-fps", type=int, default=TARGET_FPS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gmdcsa-val-ratio", type=float, default=0.15)
    parser.add_argument("--gmdcsa-test-ratio", type=float, default=0.15)
    parser.add_argument("--max-videos-per-dataset", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.yolo_model.exists():
        raise FileNotFoundError(f"YOLO model not found: {args.yolo_model}")

    items = collect_items(args)
    if not items:
        raise RuntimeError("No videos collected. Check dataset paths.")

    from ultralytics import YOLO

    device = select_device()
    model = YOLO(str(args.yolo_model))

    rows = []
    for item in tqdm(items, desc="Extracting frame cache"):
        row = extract_video(
            item=item,
            model=model,
            device=device,
            output_dir=args.output_dir,
            target_fps=args.target_fps,
            image_size=args.img_size,
            skip_existing=args.skip_existing,
        )
        rows.append(row)

    write_manifest(args.output_dir, rows)
    summary = summarize(rows, args)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] Frame cache saved to:", args.output_dir)
    print("Manifest:", args.output_dir / "manifest.csv")
    print("Summary :", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
