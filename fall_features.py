from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KEYPOINT_COUNT = 17
FEATURE_DIM = 52
KEYPOINT_CONF_THRESHOLD = 0.25


@dataclass(frozen=True)
class PoseFeatureMeta:
    person_idx: int | None
    bbox_ratio: float
    valid_keypoints: int
    lower_body_keypoints: int
    nose_y: float | None


def _empty_result(person_idx: int | None = None) -> tuple[np.ndarray, PoseFeatureMeta]:
    return (
        np.zeros(FEATURE_DIM, dtype=np.float32),
        PoseFeatureMeta(
            person_idx=person_idx,
            bbox_ratio=0.0,
            valid_keypoints=0,
            lower_body_keypoints=0,
            nose_y=None,
        ),
    )


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def select_largest_person(result) -> int | None:
    if result.boxes is None or len(result.boxes.xyxy) == 0:
        return 0 if result.keypoints is not None and len(result.keypoints.xy) > 0 else None

    boxes = _to_numpy(result.boxes.xyxy)
    if boxes is None or len(boxes) == 0:
        return None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def get_track_ids(result) -> list[tuple[int, int]]:
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return []

    person_count = len(result.keypoints.xy)
    if result.boxes is not None and result.boxes.id is not None:
        raw_ids = _to_numpy(result.boxes.id).astype(int).tolist()
    else:
        raw_ids = list(range(person_count))

    people = []
    for person_idx in range(person_count):
        track_id = raw_ids[person_idx] if person_idx < len(raw_ids) else person_idx
        people.append((person_idx, int(track_id)))
    return people


def extract_yolo_pose_features(
    result,
    person_idx: int | None = None,
    *,
    prefer_largest: bool = False,
) -> tuple[np.ndarray, PoseFeatureMeta]:
    """Return the 52-D feature vector used by the LSTM/Transformer datasets.

    Layout: 17 keypoints * (x_norm, y_norm, confidence) + bbox_width / bbox_height.
    The x/y normalization intentionally follows the existing training extractors:
    keypoints are normalized relative to the person's bounding box and clipped to
    [-1, 2]. Keeping this identical is critical for deployed model quality.
    """
    if result.keypoints is None or result.boxes is None:
        return _empty_result(person_idx)

    if len(result.keypoints.data) == 0 or len(result.boxes.xyxy) == 0:
        return _empty_result(person_idx)

    if person_idx is None:
        person_idx = select_largest_person(result) if prefer_largest else 0

    if person_idx is None:
        return _empty_result(person_idx)

    keypoints_all = _to_numpy(result.keypoints.data)
    boxes = _to_numpy(result.boxes.xyxy)

    if keypoints_all is None or boxes is None:
        return _empty_result(person_idx)
    if person_idx >= len(keypoints_all) or person_idx >= len(boxes):
        return _empty_result(person_idx)

    kp = keypoints_all[person_idx].astype(np.float32, copy=True)
    if kp.shape[0] < KEYPOINT_COUNT:
        return _empty_result(person_idx)

    kp = kp[:KEYPOINT_COUNT]
    x1, y1, x2, y2 = boxes[person_idx].astype(np.float32)
    bbox_w = max(float(x2 - x1), 1.0)
    bbox_h = max(float(y2 - y1), 1.0)

    kp_norm = kp.copy()
    kp_norm[:, 0] = (kp_norm[:, 0] - x1) / bbox_w
    kp_norm[:, 1] = (kp_norm[:, 1] - y1) / bbox_h
    kp_norm[:, 0] = np.clip(kp_norm[:, 0], -1.0, 2.0)
    kp_norm[:, 1] = np.clip(kp_norm[:, 1], -1.0, 2.0)

    bbox_ratio = bbox_w / bbox_h
    feature = np.concatenate(
        [
            kp_norm.flatten(),
            np.array([bbox_ratio], dtype=np.float32),
        ]
    ).astype(np.float32)

    if feature.shape[0] != FEATURE_DIM:
        fixed = np.zeros(FEATURE_DIM, dtype=np.float32)
        fixed[: min(FEATURE_DIM, feature.shape[0])] = feature[:FEATURE_DIM]
        feature = fixed

    xy = kp[:, :2]
    keypoint_conf = kp[:, 2] if kp.shape[1] >= 3 else np.ones(KEYPOINT_COUNT, dtype=np.float32)
    confident = (xy[:, 0] > 0) & (xy[:, 1] > 0) & (keypoint_conf >= KEYPOINT_CONF_THRESHOLD)
    valid_keypoints = int(np.sum(confident))
    lower_body_indices = [11, 12, 13, 14, 15, 16]
    lower_body_keypoints = int(np.sum(confident[lower_body_indices]))
    nose_y = float(xy[0, 1]) if xy[0, 1] > 0 else None

    return feature, PoseFeatureMeta(
        person_idx=person_idx,
        bbox_ratio=float(bbox_ratio),
        valid_keypoints=valid_keypoints,
        lower_body_keypoints=lower_body_keypoints,
        nose_y=nose_y,
    )
