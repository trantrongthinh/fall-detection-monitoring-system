import argparse
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime
import cv2
import numpy as np
import torch
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from ultralytics import YOLO
from fall_features import FEATURE_DIM, extract_yolo_pose_features, get_track_ids


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LSTM_MODEL_PATH = os.path.join(PROJECT_DIR, "models_4_clean", "seq30", "lstm_best_seq30.h5")
VIDEO_SOURCE_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}
LOGGER = logging.getLogger("fall_detection")


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


def load_env_file(path=".env"):
    env_path = project_path(path)
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except OSError as exc:
        print(f"[WARN] Cannot read {env_path}: {exc}")


def load_telegram_config(path="telegram_config.json"):
    config_path = project_path(path)
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Cannot read {config_path}: {exc}")
        return {}


load_env_file()
TELEGRAM_CONFIG = load_telegram_config()


def load_runtime_config(path="models_lstm/lstm_runtime_config.json"):
    config_path = project_path(path)
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Cannot read {config_path}: {exc}")
        return {}


RUNTIME_CONFIG = load_runtime_config()


def config_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_lstm_model_path():
    return DEFAULT_LSTM_MODEL_PATH


def configured_lstm_model_path():
    env_path = os.getenv("LSTM_MODEL_PATH")
    if env_path:
        return env_path
    config_path = RUNTIME_CONFIG.get("model_path")
    if config_path:
        return config_path if os.path.isabs(config_path) else os.path.join(PROJECT_DIR, config_path)
    return default_lstm_model_path()


class DenseCompat(layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


def load_lstm_model(path):
    with custom_object_scope({"Dense": DenseCompat}):
        return load_model(path, compile=False)


CFG = {
    "yolo_model": "yolo11m-pose.pt",
    "lstm_model": configured_lstm_model_path(),
    "camera_index": os.getenv("CAMERA_INDEX", "0"),
    "camera_name": os.getenv("CAMERA_NAME", "Laptop Camera"),
    "sequence_len": int(os.getenv("SEQUENCE_LEN", RUNTIME_CONFIG.get("sequence_len", 40))),
    "feature_dim": FEATURE_DIM,
    "conf_threshold": float(os.getenv("FALL_THRESHOLD", RUNTIME_CONFIG.get("threshold", 0.50))),
    "low_conf_threshold": float(os.getenv("FALL_LOW_THRESHOLD", RUNTIME_CONFIG.get("low_threshold", 0.35))),
    "high_conf_threshold": float(os.getenv("FALL_HIGH_THRESHOLD", RUNTIME_CONFIG.get("high_threshold", 0.70))),
    "prob_smooth_window": int(os.getenv("FALL_SMOOTH_WINDOW", RUNTIME_CONFIG.get("smooth_window", 5))),
    "pose_ratio_threshold": float(os.getenv("FALL_POSE_RATIO", RUNTIME_CONFIG.get("pose_ratio_threshold", 1.15))),
    "model_fall_pose_ratio": float(os.getenv("MODEL_FALL_POSE_RATIO", RUNTIME_CONFIG.get("model_fall_pose_ratio", 0.50))),
    "min_valid_keypoints": int(os.getenv("FALL_MIN_KEYPOINTS", RUNTIME_CONFIG.get("min_valid_keypoints", 6))),
    "recovery_frames": int(os.getenv("FALL_RECOVERY_FRAMES", RUNTIME_CONFIG.get("recovery_frames", 10))),
    "recovery_probability": float(os.getenv("FALL_RECOVERY_PROB", RUNTIME_CONFIG.get("recovery_probability", 0.70))),
    "recovery_pose_ratio": float(os.getenv("FALL_RECOVERY_POSE_RATIO", RUNTIME_CONFIG.get("recovery_pose_ratio", 1.05))),
    "motion_probability": float(os.getenv("FALL_MOTION_PROB", RUNTIME_CONFIG.get("motion_probability", 0.30))),
    "fall_motion_threshold": float(os.getenv("FALL_MOTION_THRESHOLD", RUNTIME_CONFIG.get("fall_motion_threshold", 10.0))),
    "fall_counter_decay": int(os.getenv("FALL_COUNTER_DECAY", RUNTIME_CONFIG.get("fall_counter_decay", 1))),
    "early_fall_enabled": config_bool("EARLY_FALL_ENABLED", RUNTIME_CONFIG.get("early_fall_enabled", True)),
    "early_fall_pose_ratio": float(os.getenv("EARLY_FALL_POSE_RATIO", RUNTIME_CONFIG.get("early_fall_pose_ratio", 1.55))),
    "early_fall_probability": float(os.getenv("EARLY_FALL_PROB", RUNTIME_CONFIG.get("early_fall_probability", 0.90))),
    "early_fall_min_keypoints": int(os.getenv("EARLY_FALL_MIN_KEYPOINTS", RUNTIME_CONFIG.get("early_fall_min_keypoints", 6))),
    "early_fall_min_lower_body_keypoints": int(
        os.getenv("EARLY_FALL_MIN_LOWER_BODY_KEYPOINTS", RUNTIME_CONFIG.get("early_fall_min_lower_body_keypoints", 2))
    ),
    "yolo_conf_threshold": 0.5,
    # Compatibility values for the optional debug/filter helpers. Classic mode does not use them.
    "person_conf_threshold": 0.25,
    "keypoint_conf_threshold": 0.15,
    "min_detection_keypoints": 5,
    "min_core_keypoints": 3,
    "min_detection_lower_body_keypoints": 1,
    "reject_truncated_person": False,
    "edge_margin_ratio": 0.015,
    "tracker": "bytetrack.yaml",
    "track_ttl_frames": 60,
    "confirm_frames": int(os.getenv("FALL_CONFIRM_FRAMES", RUNTIME_CONFIG.get("confirm_frames", 4))),
    "display_hold_frames": int(os.getenv("FALL_DISPLAY_HOLD_FRAMES", RUNTIME_CONFIG.get("display_hold_frames", 15))),
    "alert_hold_frames": int(os.getenv("FALL_ALERT_HOLD_FRAMES", RUNTIME_CONFIG.get("alert_hold_frames", 0))),
    "video_alert_hold_frames": int(os.getenv("VIDEO_ALERT_HOLD_FRAMES", RUNTIME_CONFIG.get("video_alert_hold_frames", 40))),
    "cooldown_sec": 30,
    "camera_width": 640,
    "camera_height": 480,
    "target_fps": float(os.getenv("TARGET_FPS", RUNTIME_CONFIG.get("target_fps", 10.0))),
    "video_frame_stride": int(os.getenv("VIDEO_FRAME_STRIDE", RUNTIME_CONFIG.get("video_frame_stride", 0))),
    "video_display_fps": float(os.getenv("VIDEO_DISPLAY_FPS", RUNTIME_CONFIG.get("video_display_fps", 30.0))),
    "video_end_hold_ms": int(os.getenv("VIDEO_END_HOLD_MS", RUNTIME_CONFIG.get("video_end_hold_ms", 0))),
    "stable_single_person": config_bool("STABLE_SINGLE_PERSON", RUNTIME_CONFIG.get("stable_single_person", False)),
    "stable_single_person_video": config_bool("STABLE_SINGLE_PERSON_VIDEO", False),
    "enable_track_handoff": config_bool("ENABLE_TRACK_HANDOFF", RUNTIME_CONFIG.get("enable_track_handoff", True)),
    "track_handoff_frames": int(os.getenv("TRACK_HANDOFF_FRAMES", RUNTIME_CONFIG.get("track_handoff_frames", 12))),
    "track_handoff_distance_ratio": float(
        os.getenv("TRACK_HANDOFF_DISTANCE_RATIO", RUNTIME_CONFIG.get("track_handoff_distance_ratio", 0.35))
    ),
    "track_handoff_area_ratio": float(
        os.getenv("TRACK_HANDOFF_AREA_RATIO", RUNTIME_CONFIG.get("track_handoff_area_ratio", 4.0))
    ),
    "display_width": 1280,
    "display_height": 720,
    "show_window": True,
    "overlay_style": "classic",
    "show_demo_debug": config_bool("SHOW_DEMO_DEBUG", False),
    "capture_dir": project_path("captures"),
    "database_path": project_path("fall_events.db"),
    "log_dir": project_path(os.getenv("LOG_DIR", RUNTIME_CONFIG.get("log_dir", "logs"))),
    "yolo_device": 0 if torch.cuda.is_available() else "cpu",
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "") or TELEGRAM_CONFIG.get("bot_token", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "") or str(TELEGRAM_CONFIG.get("chat_id", "")),
}


def setup_logging():
    os.makedirs(CFG["log_dir"], exist_ok=True)
    log_path = os.path.join(CFG["log_dir"], "fall_detection.log")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    LOGGER.info("Logging initialized at %s", log_path)


def normalize_capture_source(source):
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


def is_video_file_source(source):
    if not isinstance(source, str):
        return False
    return os.path.splitext(source)[1].lower() in VIDEO_SOURCE_EXTENSIONS


def open_configured_capture():
    return cv2.VideoCapture(normalize_capture_source(CFG["camera_index"]))


def resolve_frame_stride(cap):
    override = int(CFG.get("video_frame_stride", 0) or 0)
    if override > 0:
        return override

    if not is_video_file_source(CFG["camera_index"]):
        return 1

    target_fps = float(CFG.get("target_fps", 0.0) or 0.0)
    if target_fps <= 0:
        return 1

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps > 60:
        source_fps = 30.0
    if source_fps <= 0:
        return 1

    return max(int(round(source_fps / target_fps)), 1)


def resolve_display_wait_ms(cap):
    if not is_video_file_source(CFG["camera_index"]):
        return 1

    display_fps = float(CFG.get("video_display_fps", 0.0) or 0.0)
    if display_fps <= 0:
        display_fps = float(CFG.get("target_fps", 0.0) or 0.0)
    if display_fps <= 0:
        display_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if display_fps > 60:
        display_fps = 30.0
    if display_fps <= 0:
        return 1

    return max(int(round(1000.0 / display_fps)), 1)


def read_sampled_frame(cap, frame_stride, raw_frame_index):
    while True:
        ret, frame = cap.read()
        if not ret:
            return False, None, raw_frame_index

        raw_frame_index += 1
        if frame_stride <= 1 or raw_frame_index % frame_stride == 0:
            return True, frame, raw_frame_index


class DecisionEngine:
    def __init__(
        self,
        confirm_frames=6,
        cooldown_seconds=30,
        threshold=0.5,
        low_threshold=0.35,
        high_threshold=0.70,
        smooth_window=5,
        pose_ratio_threshold=1.15,
        model_fall_pose_ratio=0.50,
        min_valid_keypoints=6,
        recovery_frames=8,
        recovery_probability=0.55,
        recovery_pose_ratio=1.20,
        motion_probability=0.30,
        fall_motion_threshold=10.0,
        counter_decay=1,
    ):
        self.confirm_frames = confirm_frames
        self.cooldown_seconds = cooldown_seconds
        self.threshold = threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.pose_ratio_threshold = pose_ratio_threshold
        self.model_fall_pose_ratio = model_fall_pose_ratio
        self.min_valid_keypoints = min_valid_keypoints
        self.recovery_frames = recovery_frames
        self.recovery_probability = recovery_probability
        self.recovery_pose_ratio = recovery_pose_ratio
        self.motion_probability = motion_probability
        self.fall_motion_threshold = fall_motion_threshold
        self.counter_decay = max(int(counter_decay), 1)
        self.prob_history = deque(maxlen=max(int(smooth_window), 1))
        self.fall_counter = 0
        self.recovery_counter = 0
        self.last_nose_y = None
        self.last_alert_time = 0
        self.is_alerted = False

    def update(self, fall_probability, pose_meta=None):
        now = time.time()
        fall_probability = float(fall_probability)
        self.prob_history.append(fall_probability)
        smoothed_probability = float(np.mean(self.prob_history))

        bbox_ratio = float(getattr(pose_meta, "bbox_ratio", 0.0) or 0.0)
        valid_keypoints = int(getattr(pose_meta, "valid_keypoints", 0) or 0)
        nose_y = getattr(pose_meta, "nose_y", None)
        enough_keypoints = valid_keypoints >= self.min_valid_keypoints
        pose_cue = bbox_ratio >= self.pose_ratio_threshold
        model_pose_cue = bbox_ratio >= self.model_fall_pose_ratio
        nose_delta = 0.0
        if nose_y is not None and self.last_nose_y is not None:
            nose_delta = float(nose_y) - float(self.last_nose_y)
        if nose_y is not None:
            self.last_nose_y = float(nose_y)

        fall_shape_signal = pose_cue
        strong_model_signal = smoothed_probability >= self.high_threshold and model_pose_cue
        fall_pose_signal = smoothed_probability >= self.threshold and fall_shape_signal
        motion_signal = (
            smoothed_probability >= self.motion_probability
            and nose_delta >= self.fall_motion_threshold
            and model_pose_cue
        )
        continuing_fall_signal = (
            self.fall_counter > 0
            and smoothed_probability >= self.low_threshold
            and (pose_cue or self.is_alerted)
        )

        fall_evidence = enough_keypoints and (
            strong_model_signal
            or fall_pose_signal
            or motion_signal
            or continuing_fall_signal
        )

        recovery_signal = (
            smoothed_probability <= self.recovery_probability
            and bbox_ratio <= self.recovery_pose_ratio
        )
        if self.is_alerted and recovery_signal and not fall_evidence:
            self.recovery_counter += 1
        else:
            self.recovery_counter = 0

        recovered = self.is_alerted and self.recovery_counter >= self.recovery_frames
        recovering = self.is_alerted and self.recovery_counter > 0 and not recovered
        if recovered:
            self.fall_counter = 0
            self.recovery_counter = 0
            self.is_alerted = False
            self.prob_history.clear()
            label = "normal"
        elif fall_evidence:
            self.fall_counter += 1
            label = "fall"
        elif recovering or self.is_alerted:
            label = "fall"
        else:
            self.fall_counter = max(0, self.fall_counter - self.counter_decay)
            label = "fall" if self.fall_counter > 0 else "normal"

        result = {
            "label": label,
            "fall_probability": fall_probability,
            "smoothed_probability": smoothed_probability,
            "bbox_ratio": bbox_ratio,
            "valid_keypoints": valid_keypoints,
            "pose_cue": pose_cue,
            "model_pose_cue": model_pose_cue,
            "fall_shape_signal": fall_shape_signal,
            "motion_signal": motion_signal,
            "nose_delta": nose_delta,
            "recovery_counter": self.recovery_counter,
            "recovering": recovering,
            "recovered": recovered,
            "fall_counter": self.fall_counter,
            "should_alert": False,
            "in_cooldown": False,
        }

        if not enough_keypoints:
            self.fall_counter = max(0, self.fall_counter - self.counter_decay)
            result["label"] = "fall" if self.fall_counter > 0 or self.is_alerted else "normal"
            result["fall_counter"] = self.fall_counter

        if now - self.last_alert_time < self.cooldown_seconds:
            result["in_cooldown"] = True
            return result

        if self.fall_counter >= self.confirm_frames and not self.is_alerted:
            result["should_alert"] = True
            self.is_alerted = True
            self.last_alert_time = now

        return result


class FallDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_schema()

    def connect(self):
        return sqlite3.connect(self.db_path, timeout=15)

    def init_schema(self):
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fall_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fall_counter INTEGER NOT NULL,
                    track_id INTEGER,
                    image_path TEXT,
                    status TEXT NOT NULL DEFAULT 'new',
                    review_note TEXT,
                    reviewed_at TEXT,
                    telegram_sent INTEGER NOT NULL DEFAULT 0,
                    telegram_response TEXT
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(fall_events)")}
            if "track_id" not in columns:
                conn.execute("ALTER TABLE fall_events ADD COLUMN track_id INTEGER")
            if "status" not in columns:
                conn.execute("ALTER TABLE fall_events ADD COLUMN status TEXT NOT NULL DEFAULT 'new'")
            if "review_note" not in columns:
                conn.execute("ALTER TABLE fall_events ADD COLUMN review_note TEXT")
            if "reviewed_at" not in columns:
                conn.execute("ALTER TABLE fall_events ADD COLUMN reviewed_at TEXT")
            conn.commit()

    def insert_event(self, camera_name, label, confidence, fall_counter, image_path, track_id=None, status="new"):
        created_at = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO fall_events (
                    created_at, camera_name, label, confidence,
                    fall_counter, track_id, image_path, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, camera_name, label, confidence, fall_counter, track_id, image_path, status),
            )
            conn.commit()
            return cur.lastrowid

    def update_telegram_result(self, event_id, sent, response_text):
        status = "alert_sent" if sent else "alert_failed"
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE fall_events
                SET telegram_sent = ?, telegram_response = ?, status = ?
                WHERE id = ?
                """,
                (1 if sent else 0, response_text[:2000], status, event_id),
            )
            conn.commit()


class TelegramAlerter:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()

    @property
    def enabled(self):
        return bool(self.bot_token and self.chat_id)

    def _api_url(self, method):
        return f"https://api.telegram.org/bot{self.bot_token}/{method}"

    def send_message(self, text):
        if not self.enabled:
            return False, "Telegram disabled: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"

        payload = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": text,
            }
        ).encode("utf-8")
        request = urllib.request.Request(self._api_url("sendMessage"), data=payload, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8", errors="replace")
                return 200 <= response.status < 300, body
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return False, f"HTTP {exc.code}: {body}"
        except Exception as exc:
            return False, repr(exc)

    def send_photo(self, image_path, caption):
        if not self.enabled:
            return False, "Telegram disabled: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
        if not os.path.exists(image_path):
            return self.send_message(caption)

        boundary = f"----FallDetection{int(time.time() * 1000)}"
        body = bytearray()

        def add_field(name, value):
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        add_field("chat_id", self.chat_id)
        add_field("caption", caption)

        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            photo_bytes = f.read()

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
                "Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(photo_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        request = urllib.request.Request(
            self._api_url("sendPhoto"),
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                return 200 <= response.status < 300, response_body
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return False, f"HTTP {exc.code}: {response_body}"
        except Exception as exc:
            return False, repr(exc)


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _box_confidence(result, person_idx):
    if result.boxes is None or result.boxes.conf is None:
        return 1.0
    conf = _to_numpy(result.boxes.conf)
    if conf is None or person_idx >= len(conf):
        return 1.0
    return float(conf[person_idx])


def _box_xyxy(result, person_idx):
    if result.boxes is None or len(result.boxes.xyxy) <= person_idx:
        return None
    boxes = _to_numpy(result.boxes.xyxy)
    if boxes is None or person_idx >= len(boxes):
        return None
    return boxes[person_idx].astype(np.float32)


def _person_keypoints_data(result, person_idx):
    if result.keypoints is None or len(result.keypoints.data) <= person_idx:
        return None
    keypoints = _to_numpy(result.keypoints.data)
    if keypoints is None or person_idx >= len(keypoints):
        return None
    return keypoints[person_idx]


def _confident_keypoint_mask(result, person_idx):
    points = _person_keypoints_data(result, person_idx)
    if points is None or len(points) == 0:
        return np.zeros(17, dtype=bool)

    xy = points[:, :2]
    if points.shape[1] >= 3:
        conf = points[:, 2]
    else:
        conf = np.ones(len(points), dtype=np.float32)

    mask = (xy[:, 0] > 0) & (xy[:, 1] > 0) & (conf >= CFG["keypoint_conf_threshold"])
    fixed = np.zeros(17, dtype=bool)
    fixed[: min(17, len(mask))] = mask[:17]
    return fixed


def _valid_keypoint_count(result, person_idx):
    return int(np.sum(_confident_keypoint_mask(result, person_idx)))


def _pose_has_human_core(result, person_idx):
    confident = _confident_keypoint_mask(result, person_idx)
    core_indices = [5, 6, 11, 12]
    lower_body_indices = [11, 12, 13, 14, 15, 16]

    core_count = int(np.sum(confident[core_indices]))
    shoulder_count = int(np.sum(confident[[5, 6]]))
    hip_count = int(np.sum(confident[[11, 12]]))
    lower_body_count = int(np.sum(confident[lower_body_indices]))

    return (
        core_count >= CFG["min_core_keypoints"]
        and shoulder_count >= 1
        and hip_count >= 1
        and lower_body_count >= CFG["min_detection_lower_body_keypoints"]
    )


def _point_inside_monitor_roi(x, y, frame_shape):
    roi = CFG.get("monitor_roi")
    if not roi:
        return True 

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi
    return (x1 * w) <= x <= (x2 * w) and (y1 * h) <= y <= (y2 * h)


def is_valid_person_candidate(result, person_idx, frame_shape):
    box = _box_xyxy(result, person_idx)
    if box is None:
        return False

    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    if CFG["reject_truncated_person"]:
        margin_x = CFG["edge_margin_ratio"] * w
        margin_y = CFG["edge_margin_ratio"] * h
        touches_edge = (
            x1 <= margin_x
            or y1 <= margin_y
            or x2 >= (w - margin_x)
            or y2 >= (h - margin_y)
        )
        if touches_edge:
            return False

    center_x = float((x1 + x2) / 2.0)
    center_y = float((y1 + y2) / 2.0)
    if not _point_inside_monitor_roi(center_x, center_y, frame_shape):
        return False

    if _box_confidence(result, person_idx) < CFG["person_conf_threshold"]:
        return False

    if _valid_keypoint_count(result, person_idx) < CFG["min_detection_keypoints"]:
        return False

    if not _pose_has_human_core(result, person_idx):
        return False

    return True


def get_largest_person_as_stable_track(result, frame_shape=None):
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return []

    person_count = len(result.keypoints.xy)
    person_idx = 0
    if result.boxes is not None and len(result.boxes.xyxy) > 0:
        boxes = _to_numpy(result.boxes.xyxy)
        if boxes is not None and len(boxes) > 0:
            limit = min(person_count, len(boxes))
            areas = (boxes[:limit, 2] - boxes[:limit, 0]) * (boxes[:limit, 3] - boxes[:limit, 1])
            person_idx = int(np.argmax(areas))

    return [(person_idx, 0)]


def get_tracked_people(result, frame_shape=None):
    if CFG.get("stable_single_person") or (
        CFG.get("stable_single_person_video") and is_video_file_source(CFG["camera_index"])
    ):
        return get_largest_person_as_stable_track(result, frame_shape)
    return get_track_ids(result)


def get_person_box_metrics(result, person_idx, frame_shape):
    box = _box_xyxy(result, person_idx)
    if box is None:
        return None

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    x1 = max(min(x1, w - 1), 0.0)
    y1 = max(min(y1, h - 1), 0.0)
    x2 = max(min(x2, w - 1), 0.0)
    y2 = max(min(y2, h - 1), 0.0)
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    return {
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
        "bbox_center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        "bbox_area": box_w * box_h,
    }


def resolve_track_handoff(
    raw_track_id,
    person_idx,
    result,
    frame_shape,
    frame_index,
    current_raw_track_ids,
    sequence_buffers,
    engines,
    last_seen_frame,
    last_person_states,
):
    if not CFG.get("enable_track_handoff", True):
        return raw_track_id
    if raw_track_id in sequence_buffers:
        return raw_track_id
    if CFG.get("stable_single_person"):
        return raw_track_id

    metrics = get_person_box_metrics(result, person_idx, frame_shape)
    if metrics is None:
        return raw_track_id

    h, w = frame_shape[:2]
    max_distance = float(CFG["track_handoff_distance_ratio"]) * float((w * w + h * h) ** 0.5)
    max_missing = int(CFG["track_handoff_frames"])
    max_area_ratio = float(CFG["track_handoff_area_ratio"])
    new_center = metrics["bbox_center"]
    new_area = max(float(metrics["bbox_area"]), 1.0)

    best_track_id = None
    best_score = None
    for old_track_id in list(sequence_buffers.keys()):
        if old_track_id == raw_track_id or old_track_id in current_raw_track_ids:
            continue

        missing_frames = frame_index - int(last_seen_frame.get(old_track_id, -10**9))
        if missing_frames <= 0 or missing_frames > max_missing:
            continue

        old_state = last_person_states.get(old_track_id, {})
        old_center = old_state.get("bbox_center")
        old_area = old_state.get("bbox_area")
        if old_center is None or old_area is None:
            continue

        dx = float(new_center[0]) - float(old_center[0])
        dy = float(new_center[1]) - float(old_center[1])
        distance = float((dx * dx + dy * dy) ** 0.5)
        if distance > max_distance:
            continue

        old_area = max(float(old_area), 1.0)
        area_ratio = max(new_area / old_area, old_area / new_area)
        if area_ratio > max_area_ratio:
            continue

        score = distance + missing_frames * 8.0 + abs(area_ratio - 1.0) * 12.0
        if best_score is None or score < best_score:
            best_score = score
            best_track_id = old_track_id

    return best_track_id if best_track_id is not None else raw_track_id


def assign_display_name(track_id, display_numbers):
    if track_id not in display_numbers:
        used_numbers = set(display_numbers.values())
        next_number = 1
        while next_number in used_numbers:
            next_number += 1
        display_numbers[track_id] = next_number

    return f"Person {display_numbers[track_id]}"


def count_raw_people(result):
    return len(get_track_ids(result))


def extract_52_features(result, person_idx=0, return_meta=False):
    features, meta = extract_yolo_pose_features(result, person_idx)
    if return_meta:
        return features, meta
    return features


def predict_lstm_probability(lstm_model, sequence_buffer):
    input_data = np.expand_dims(np.array(sequence_buffer, dtype=np.float32), axis=0)
    return float(lstm_model.predict(input_data, verbose=0)[0][0])


def predict_lstm(lstm_model, sequence_buffer):
    prob = predict_lstm_probability(lstm_model, sequence_buffer)
    if prob >= CFG["conf_threshold"]:
        return "fall", prob

    return "normal", 1.0 - prob


def is_early_fall_candidate(pose_meta):
    if not CFG.get("early_fall_enabled", True):
        return False

    bbox_ratio = float(getattr(pose_meta, "bbox_ratio", 0.0) or 0.0)
    valid_keypoints = int(getattr(pose_meta, "valid_keypoints", 0) or 0)
    lower_body_keypoints = int(getattr(pose_meta, "lower_body_keypoints", 0) or 0)

    return (
        bbox_ratio >= float(CFG["early_fall_pose_ratio"])
        and valid_keypoints >= int(CFG["early_fall_min_keypoints"])
        and lower_body_keypoints >= int(CFG["early_fall_min_lower_body_keypoints"])
    )


def draw_overlay(frame, label, confidence, fall_counter, fps, alert_active):
    h, w = frame.shape[:2]
    color = (0, 0, 220) if alert_active or label == "fall" else (30, 180, 60)
    if fall_counter > 0 and not alert_active:
        color = (0, 140, 255)

    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, 64), (15, 15, 15), -1)
    cv2.addWeighted(top, 0.82, frame, 0.18, 0, frame)

    if alert_active:
        status = "FALL DETECTED"
    elif fall_counter > 0:
        status = f"WARNING {fall_counter}/{CFG['confirm_frames']}"
    else:
        status = "NORMAL"

    cv2.putText(frame, status, (14, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 1)

    bottom = frame.copy()
    cv2.rectangle(bottom, (0, h - 72), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(bottom, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame, f"Action: {label.upper()}", (14, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(frame, f"Conf: {confidence:.1%}", (14, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    if alert_active and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 5)

    return frame


def _status_stage(state):
    if state is None:
        return "no_person"
    if state.get("recovering"):
        return "recovering"
    if state.get("alert_active"):
        return "fall"
    if state.get("fall_counter", 0) > 0:
        return "checking"
    if state.get("label") == "collecting":
        return "checking"
    if state.get("label") == "fall":
        return "checking"
    return "normal"


def _stage_text(stage):
    return {
        "fall": "FALL DETECTED",
        "recovering": "RECOVERING",
        "checking": "CHECKING",
        "normal": "NORMAL",
        "no_person": "NO PERSON",
    }.get(stage, "NORMAL")


def _stage_color(stage):
    return {
        "fall": (0, 0, 220),
        "recovering": (0, 140, 255),
        "checking": (0, 190, 255),
        "normal": (30, 180, 60),
        "no_person": (190, 190, 190),
    }.get(stage, (30, 180, 60))


def _fall_risk_for_display(state):
    if state.get("label") == "collecting":
        return float(state.get("sequence_progress", 0.0) or 0.0)
    return float(state.get("fall_probability", state.get("confidence", 0.0)) or 0.0)


def draw_person_status(
    frame,
    result,
    person_idx,
    track_id,
    label,
    confidence,
    fall_counter,
    alert_active,
    sequence_progress=None,
    recovering=False,
    display_name=None,
):
    if result.boxes is not None and len(result.boxes.xyxy) > person_idx:
        x1, y1, x2, y2 = result.boxes.xyxy[person_idx].cpu().numpy().astype(int)
    elif result.keypoints is not None and len(result.keypoints.xy) > person_idx:
        xy = result.keypoints.xy[person_idx].cpu().numpy()
        valid = xy[(xy[:, 0] > 0) & (xy[:, 1] > 0)]
        if len(valid) == 0:
            return frame
        x1, y1 = valid.min(axis=0).astype(int)
        x2, y2 = valid.max(axis=0).astype(int)
    else:
        return frame

    h, w = frame.shape[:2]
    pad = 8
    x1 = max(int(x1) - pad, 0)
    y1 = max(int(y1) - pad, 0)
    x2 = min(int(x2) + pad, w - 1)
    y2 = min(int(y2) + pad, h - 1)

    state = {
        "label": label,
        "confidence": confidence,
        "fall_probability": confidence,
        "fall_counter": fall_counter,
        "alert_active": alert_active,
        "sequence_progress": sequence_progress,
        "recovering": recovering,
    }
    stage = _status_stage(state)
    color = _stage_color(stage)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    title = f"{display_name or f'ID {track_id}'} - {_stage_text(stage)}"
    if CFG.get("show_demo_debug"):
        if display_name is not None:
            title += f" | id {track_id}"
        risk = _fall_risk_for_display(state)
        title += f" | fall risk {risk:.0%}"
        if fall_counter > 0:
            title += f" | confirm {fall_counter}/{CFG['confirm_frames']}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(title, font, scale, thickness)
    text_x = min(max(x1, 4), max(w - text_w - 12, 4))
    text_y = y1 - 10
    if text_y - text_h - 8 < 0:
        text_y = min(y2 + text_h + 12, h - 8)

    cv2.rectangle(
        frame,
        (text_x - 4, text_y - text_h - 8),
        (min(text_x + text_w + 8, w - 1), min(text_y + 5, h - 1)),
        color,
        -1,
    )
    cv2.putText(frame, title, (text_x, text_y), font, scale, (255, 255, 255), thickness)
    return frame


def select_primary_state(person_states):
    if not person_states:
        return None

    def sort_key(item):
        _, state = item
        label_rank = 2 if state["label"] == "fall" else 1 if state["label"] == "normal" else 0
        return (
            int(state["alert_active"]),
            label_rank,
            int(state["fall_counter"]),
            float(state["fall_probability"]),
        )

    return max(person_states.items(), key=sort_key)[1]


def draw_research_overlay(frame, person_states, fps, raw_people=None):
    h, w = frame.shape[:2]
    primary = select_primary_state(person_states)

    if primary is None:
        result_text = "NO PERSON"
        probability = 0.0
        fall_counter = 0.0
        result_color = (30, 180, 60)
    else:
        label = primary["label"]
        if label == "fall":
            result_text = "FALL"
            result_color = (0, 0, 255)
        elif label == "normal":
            result_text = "NON FALL"
            result_color = (0, 255, 0)
        else:
            result_text = "COLLECTING"
            result_color = (0, 255, 255)

        probability = float(primary["result_probability"])
        fall_counter = float(primary["fall_counter"])

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = max(min(w / 900.0, 1.55), 0.95)
    info_scale = max(min(w / 1250.0, 1.15), 0.72)
    title_thickness = max(int(round(title_scale * 3)), 2)
    info_thickness = max(int(round(info_scale * 3)), 2)

    x = 28
    y = int(44 * title_scale)
    line_gap = int(42 * info_scale)
    info_color = (255, 255, 0)

    cv2.putText(frame, f"RESULT: {result_text}", (x, y), font, title_scale, result_color, title_thickness)
    cv2.putText(frame, f"Prob: {probability:.2f}", (x, y + line_gap), font, info_scale, info_color, info_thickness)
    cv2.putText(
        frame,
        f"Fall counter: {fall_counter:.2f}",
        (x, y + line_gap * 2),
        font,
        info_scale,
        info_color,
        info_thickness,
    )

    if CFG.get("show_detection_debug") and raw_people is not None:
        people_text = f"Raw: {raw_people} | Valid: {len(person_states)} | FPS: {fps:.0f}"
        text_x = max(w - 390, 20)
    else:
        people_text = f"People: {len(person_states)} | FPS: {fps:.0f}"
        text_x = max(w - 285, 20)

    cv2.putText(frame, people_text, (text_x, 36), font, 0.65, (230, 230, 230), 1)

    if primary is not None and primary["alert_active"] and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 5)

    return frame


COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def draw_monitor_roi(frame):
    if not CFG.get("show_monitor_roi") or not CFG.get("monitor_roi"):
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = CFG["monitor_roi"]
    p1 = (int(x1 * w), int(y1 * h))
    p2 = (int(x2 * w), int(y2 * h))
    cv2.rectangle(frame, p1, p2, (0, 255, 255), 1)
    cv2.putText(
        frame,
        "monitor ROI",
        (p1[0], max(p1[1] - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    return frame


def draw_filtered_pose(frame, result, person_states):
    draw_monitor_roi(frame)

    if result.keypoints is None or len(person_states) == 0:
        return frame

    for track_id, state in person_states.items():
        person_idx = state["person_idx"]
        keypoints = _person_keypoints_data(result, person_idx)
        if keypoints is None:
            continue

        stage = _status_stage(state)
        if stage == "fall":
            line_color = (0, 0, 220)
            point_color = (0, 0, 255)
        elif stage == "recovering" or stage == "checking":
            line_color = (0, 160, 255)
            point_color = (0, 210, 255)
        else:
            line_color = (70, 190, 90)
            point_color = (80, 230, 120)

        points = keypoints[:, :2]
        confident = _confident_keypoint_mask(result, person_idx)
        for start, end in COCO_SKELETON:
            if start >= len(points) or end >= len(points):
                continue
            if not confident[start] or not confident[end]:
                continue
            x_start, y_start = points[start]
            x_end, y_end = points[end]
            if x_start <= 0 or y_start <= 0 or x_end <= 0 or y_end <= 0:
                continue
            cv2.line(
                frame,
                (int(x_start), int(y_start)),
                (int(x_end), int(y_end)),
                line_color,
                2,
            )

        for idx, (x, y) in enumerate(points):
            if idx >= len(confident) or not confident[idx]:
                continue
            if x <= 0 or y <= 0:
                continue
            cv2.circle(frame, (int(x), int(y)), 5, point_color, -1)

    return frame


def draw_multi_person_overlay(frame, person_states, fps):
    h, w = frame.shape[:2]
    stages = [_status_stage(state) for state in person_states.values()]

    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(top, 0.82, frame, 0.18, 0, frame)

    if "fall" in stages:
        stage = "fall"
    elif "recovering" in stages:
        stage = "recovering"
    elif "checking" in stages:
        stage = "checking"
    elif stages:
        stage = "normal"
    else:
        stage = "no_person"

    status = _stage_text(stage)
    color = _stage_color(stage)
    cv2.putText(frame, status, (14, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    if CFG.get("show_demo_debug"):
        cv2.putText(
            frame,
            f"People: {len(person_states)} | FPS: {fps:.0f}",
            (max(w - 245, 14), 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
        )

    if stage == "fall" and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 5)
    elif stage == "recovering":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

    return frame


def resize_for_display(frame, max_width=None, max_height=None):
    max_width = max_width or CFG["display_width"]
    max_height = max_height or CFG["display_height"]
    h, w = frame.shape[:2]

    scale = min(max_width / w, max_height / h)
    if scale <= 0:
        return frame

    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def send_alert_async(frame, confidence, fall_counter, database, telegram, track_id=None):
    os.makedirs(CFG["capture_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    track_suffix = f"_id{track_id}" if track_id is not None else ""
    image_path = os.path.join(CFG["capture_dir"], f"fall_{timestamp}{track_suffix}.jpg")
    cv2.imwrite(image_path, frame)

    event_id = database.insert_event(
        camera_name=CFG["camera_name"],
        label="fall",
        confidence=confidence,
        fall_counter=fall_counter,
        image_path=image_path,
        track_id=track_id,
    )
    LOGGER.info("Fall event inserted event_id=%s camera=%s track_id=%s confidence=%.4f", event_id, CFG["camera_name"], track_id, confidence)

    caption = (
        "FALL DETECTION ALERT\n"
        f"Camera: {CFG['camera_name']}\n"
        f"Person ID: {track_id if track_id is not None else 'N/A'}\n"
        f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Event ID: {event_id}"
    )

    sent, response_text = telegram.send_photo(image_path, caption)
    database.update_telegram_result(event_id, sent, response_text)
    LOGGER.info("Telegram result event_id=%s sent=%s response=%s", event_id, sent, response_text[:300])


def load_models():
    print("[INFO] Loading YOLO pose model...")
    yolo_model = YOLO(CFG["yolo_model"])
    if CFG["yolo_device"] != "cpu":
        yolo_model.to(f"cuda:{CFG['yolo_device']}")

    print("[INFO] Loading LSTM model...")
    if not os.path.exists(CFG["lstm_model"]):
        raise FileNotFoundError(f"Cannot find LSTM model: {CFG['lstm_model']}")
    print("[INFO] LSTM model:", CFG["lstm_model"])
    lstm_model = load_lstm_model(CFG["lstm_model"])
    model_input_shape = lstm_model.input_shape
    if len(model_input_shape) >= 3 and model_input_shape[1] is not None:
        model_sequence_len = int(model_input_shape[1])
        if CFG["sequence_len"] != model_sequence_len:
            print(
                f"[INFO] Updating sequence_len from {CFG['sequence_len']} "
                f"to model input sequence_len {model_sequence_len}."
            )
            CFG["sequence_len"] = model_sequence_len

    print("[INFO] YOLO device:", CFG["yolo_device"])
    print("[INFO] LSTM input shape:", lstm_model.input_shape)
    return yolo_model, lstm_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fall-detection demo.")
    parser.add_argument(
        "--camera",
        default=None,
        help="Camera index or video path. Default comes from CFG/CAMERA_INDEX.",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Camera name shown in Telegram alerts and database.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the LSTM .h5 model.",
    )
    parser.add_argument(
        "--sequence-len",
        type=int,
        default=None,
        help="Sequence length. If omitted, the loaded model input shape is used.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Approximate FPS used for model sampling on video files.",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=None,
        help="Playback FPS for video files. Use 30 for MCFD videos.",
    )
    parser.add_argument(
        "--video-alert-hold-frames",
        type=int,
        default=None,
        help="Frames to keep FALL DETECTED visible after a video alert.",
    )
    parser.add_argument(
        "--overlay",
        choices=("classic", "research"),
        default=None,
        help="Overlay style. classic matches the realtime camera demo.",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Show people/FPS/fall-risk counters on top of the clean demo overlay.",
    )
    parser.add_argument(
        "--keep-tracker-ids",
        action="store_true",
        help="Use YOLO tracker IDs instead of stable single-person mode.",
    )
    parser.add_argument(
        "--single-person",
        action="store_true",
        help="Always use the largest detected person as stable ID 0 for one-person demos.",
    )
    parser.add_argument(
        "--single-person-video",
        action="store_true",
        help="For video files, only follow the largest person. Not recommended for MCFD multi-person scenes.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force YOLO to run on CPU.",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram sending but keep local display/database.",
    )
    return parser.parse_args()


def apply_cli_overrides(args):
    if args.camera is not None:
        CFG["camera_index"] = args.camera
    if args.camera_name is not None:
        CFG["camera_name"] = args.camera_name
    elif args.camera is not None and is_video_file_source(str(args.camera)):
        CFG["camera_name"] = os.path.basename(str(args.camera))

    if args.model is not None:
        model_path = args.model
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_DIR, model_path)
        CFG["lstm_model"] = model_path
    if args.sequence_len is not None:
        CFG["sequence_len"] = int(args.sequence_len)
    if args.target_fps is not None:
        CFG["target_fps"] = float(args.target_fps)
    if args.display_fps is not None:
        CFG["video_display_fps"] = float(args.display_fps)
    if args.video_alert_hold_frames is not None:
        CFG["video_alert_hold_frames"] = int(args.video_alert_hold_frames)
    if args.overlay is not None:
        CFG["overlay_style"] = args.overlay
    if args.debug_overlay:
        CFG["show_demo_debug"] = True
    if args.keep_tracker_ids:
        CFG["stable_single_person"] = False
        CFG["stable_single_person_video"] = False
    if args.single_person:
        CFG["stable_single_person"] = True
    if args.single_person_video:
        CFG["stable_single_person_video"] = True
    if args.cpu:
        CFG["yolo_device"] = "cpu"
    if args.no_telegram:
        CFG["telegram_bot_token"] = ""
        CFG["telegram_chat_id"] = ""


def run():
    setup_logging()
    os.makedirs(CFG["capture_dir"], exist_ok=True)
    database = FallDatabase(CFG["database_path"])
    telegram = TelegramAlerter(CFG["telegram_bot_token"], CFG["telegram_chat_id"])

    if not telegram.enabled:
        LOGGER.warning("Telegram disabled. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to send phone alerts.")

    yolo_model, lstm_model = load_models()

    cap = open_configured_capture()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["camera_height"])

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try changing CFG['camera_index'].")
    frame_stride = resolve_frame_stride(cap)
    is_video_source = is_video_file_source(CFG["camera_index"])
    display_wait_ms = resolve_display_wait_ms(cap)
    if is_video_source and frame_stride > 1:
        display_wait_ms *= frame_stride
    last_display_frame = None

    sequence_buffers = {}
    engines = {}
    last_seen_frame = {}
    last_person_states = {}
    alert_hold_until = {}
    alert_hold_confidence = {}
    display_numbers = {}
    person_states = {}
    fps = 0.0
    prev_time = time.time()
    frame_index = 0
    raw_frame_index = 0
    window_name = "Fall Detection Demo - Telegram + Dashboard"

    if CFG["show_window"]:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CFG["display_width"], CFG["display_height"])

    print("[INFO] Running demo. Press Q to quit, S to save snapshot.")
    if frame_stride > 1:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        print(
            f"[INFO] Sampling video source every {frame_stride} frames "
            f"({source_fps:.1f} FPS -> about {CFG['target_fps']:.1f} FPS)."
        )
    if is_video_source and display_wait_ms > 1:
        print(f"[INFO] Displaying video at about {1000.0 / display_wait_ms:.1f} FPS.")
    while True:
        ret, frame, raw_frame_index = read_sampled_frame(cap, frame_stride, raw_frame_index)
        if not ret:
            if is_video_source:
                print("[INFO] End of video file.")
                if (
                    CFG["show_window"]
                    and last_display_frame is not None
                    and int(CFG.get("video_end_hold_ms", 0) or 0) > 0
                ):
                    cv2.imshow(window_name, last_display_frame)
                    cv2.waitKey(int(CFG["video_end_hold_ms"]))
            else:
                print("[WARN] Cannot read frame from camera.")
            break

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        frame_index += 1

        results = yolo_model.track(
            frame,
            persist=True,
            conf=CFG["yolo_conf_threshold"],
            tracker=CFG["tracker"],
            verbose=False,
            device=CFG["yolo_device"],
        )
        result = results[0]
        person_states = {}
        raw_people = count_raw_people(result)

        tracked_people = get_tracked_people(result, frame.shape)
        current_raw_track_ids = {track_id for _, track_id in tracked_people}

        for person_idx, raw_track_id in tracked_people:
            track_id = resolve_track_handoff(
                raw_track_id,
                person_idx,
                result,
                frame.shape,
                frame_index,
                current_raw_track_ids,
                sequence_buffers,
                engines,
                last_seen_frame,
                last_person_states,
            )
            if track_id not in sequence_buffers:
                sequence_buffers[track_id] = deque(maxlen=CFG["sequence_len"])
                engines[track_id] = DecisionEngine(
                    confirm_frames=CFG["confirm_frames"],
                    cooldown_seconds=CFG["cooldown_sec"],
                    threshold=CFG["conf_threshold"],
                    low_threshold=CFG["low_conf_threshold"],
                    high_threshold=CFG["high_conf_threshold"],
                    smooth_window=CFG["prob_smooth_window"],
                    pose_ratio_threshold=CFG["pose_ratio_threshold"],
                    model_fall_pose_ratio=CFG["model_fall_pose_ratio"],
                    min_valid_keypoints=CFG["min_valid_keypoints"],
                    recovery_frames=CFG["recovery_frames"],
                    recovery_probability=CFG["recovery_probability"],
                    recovery_pose_ratio=CFG["recovery_pose_ratio"],
                    motion_probability=CFG["motion_probability"],
                    fall_motion_threshold=CFG["fall_motion_threshold"],
                    counter_decay=CFG["fall_counter_decay"],
                )

            last_seen_frame[track_id] = frame_index
            box_metrics = get_person_box_metrics(result, person_idx, frame.shape) or {}
            display_name = assign_display_name(track_id, display_numbers)
            features, pose_meta = extract_52_features(result, person_idx, return_meta=True)
            sequence_buffers[track_id].append(features)

            label = "collecting"
            confidence = 0.0
            fall_probability = 0.0
            result_probability = 0.0
            engine = engines[track_id]
            decision = {
                "should_alert": False,
                "fall_counter": engine.fall_counter,
                "in_cooldown": False,
            }

            if len(sequence_buffers[track_id]) == CFG["sequence_len"]:
                fall_probability = predict_lstm_probability(lstm_model, sequence_buffers[track_id])
                decision = engine.update(fall_probability, pose_meta)
                label = decision["label"]
                confidence = decision["smoothed_probability"]
                result_probability = confidence if label == "fall" else 1.0 - confidence

                if decision["should_alert"]:
                    threading.Thread(
                        target=send_alert_async,
                        args=(frame.copy(), confidence, engine.fall_counter, database, telegram, track_id),
                        daemon=True,
                    ).start()
            elif is_early_fall_candidate(pose_meta):
                fall_probability = float(CFG["early_fall_probability"])
                decision = engine.update(fall_probability, pose_meta)
                label = decision["label"]
                confidence = decision["smoothed_probability"]
                result_probability = confidence if label == "fall" else 1.0 - confidence

                if decision["should_alert"]:
                    threading.Thread(
                        target=send_alert_async,
                        args=(frame.copy(), confidence, engine.fall_counter, database, telegram, track_id),
                        daemon=True,
                    ).start()
            elif engine.fall_counter > 0:
                decision = engine.update(0.0, pose_meta)
                label = decision["label"]
                confidence = decision["smoothed_probability"]
                result_probability = confidence if label == "fall" else 1.0 - confidence

            hold_frames = int(
                CFG["video_alert_hold_frames"] if is_video_source else CFG["alert_hold_frames"]
            )
            base_alert_active = label == "fall" and engine.fall_counter >= CFG["confirm_frames"]
            if base_alert_active or decision["should_alert"]:
                if hold_frames > 0:
                    alert_hold_until[track_id] = frame_index + hold_frames
                    alert_hold_confidence[track_id] = max(
                        float(alert_hold_confidence.get(track_id, 0.0)),
                        float(confidence),
                    )

            held_alert_active = int(alert_hold_until.get(track_id, 0) or 0) >= frame_index
            alert_active = base_alert_active or held_alert_active
            if held_alert_active and not base_alert_active:
                label = "fall"
                confidence = max(float(confidence), float(alert_hold_confidence.get(track_id, 0.0)))
                engine.fall_counter = max(engine.fall_counter, CFG["confirm_frames"])

            sequence_progress = min(len(sequence_buffers[track_id]) / float(CFG["sequence_len"]), 1.0)
            person_states[track_id] = {
                "person_idx": person_idx,
                "label": label,
                "confidence": confidence,
                "fall_probability": fall_probability,
                "result_probability": result_probability,
                "fall_counter": engine.fall_counter,
                "alert_active": alert_active,
                "sequence_progress": sequence_progress,
                "recovering": bool(decision.get("recovering", False)),
                "recovery_counter": int(decision.get("recovery_counter", 0) or 0),
                "raw_track_id": raw_track_id,
                "display_name": display_name,
                "bbox_center": box_metrics.get("bbox_center"),
                "bbox_area": box_metrics.get("bbox_area"),
            }
            last_person_states[track_id] = {
                **person_states[track_id],
                "last_seen_frame": frame_index,
            }

        stale_track_ids = [
            track_id
            for track_id, seen_at in last_seen_frame.items()
            if frame_index - seen_at > CFG["track_ttl_frames"]
        ]
        for track_id in stale_track_ids:
            sequence_buffers.pop(track_id, None)
            engines.pop(track_id, None)
            last_seen_frame.pop(track_id, None)
            last_person_states.pop(track_id, None)
            alert_hold_until.pop(track_id, None)
            alert_hold_confidence.pop(track_id, None)
            display_numbers.pop(track_id, None)

        if CFG["show_window"]:
            display_states = dict(person_states)
            for track_id, state in list(last_person_states.items()):
                if track_id in display_states:
                    continue
                missing_frames = frame_index - int(state.get("last_seen_frame", frame_index))
                if missing_frames <= CFG["display_hold_frames"] and (
                    state.get("alert_active") or state.get("fall_counter", 0) > 0
                ):
                    display_states[track_id] = state

            if CFG.get("overlay_style") == "research":
                annotated = draw_filtered_pose(frame.copy(), result, person_states)
                frame = draw_research_overlay(annotated, display_states, fps, raw_people)
            else:
                annotated = draw_filtered_pose(frame.copy(), result, person_states)
                frame = draw_multi_person_overlay(annotated, display_states, fps)
                for track_id, state in person_states.items():
                    frame = draw_person_status(
                        frame,
                        result,
                        state["person_idx"],
                        track_id,
                        state["label"],
                        state["confidence"],
                        state["fall_counter"],
                        state["alert_active"],
                        state.get("sequence_progress"),
                        state.get("recovering", False),
                        state.get("display_name"),
                    )
            display_frame = resize_for_display(frame)
            last_display_frame = display_frame.copy()
            cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(display_wait_ms if CFG["show_window"] else 1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snap_path = os.path.join(CFG["capture_dir"], f"snap_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[SNAP] Saved: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    apply_cli_overrides(parse_args())
    run()
