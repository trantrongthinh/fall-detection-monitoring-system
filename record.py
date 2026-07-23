import argparse
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2

from main_telegram_database import (
    CFG,
    DecisionEngine,
    FallDatabase,
    TelegramAlerter,
    assign_display_name,
    draw_filtered_pose,
    draw_research_overlay,
    count_raw_people,
    draw_multi_person_overlay,
    draw_person_status,
    extract_52_features,
    get_person_box_metrics,
    get_tracked_people,
    is_early_fall_candidate,
    is_video_file_source,
    load_models,
    open_configured_capture,
    predict_lstm_probability,
    read_sampled_frame,
    resolve_track_handoff,
    resolve_frame_stride,
    resolve_display_wait_ms,
    resize_for_display,
    send_alert_async,
    setup_logging,
    LOGGER,
)


def build_output_path(output_path):
    if output_path:
        return output_path

    os.makedirs("recordings", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("recordings", f"fall_demo_{timestamp}.mp4")


def open_writer(output_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    return writer


def draw_recording_countdown(frame, seconds_left):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 86), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    title = "GET READY"
    subtitle = f"Recording starts in {max(seconds_left, 0.0):.1f}s"
    cv2.putText(frame, title, (18, 38), cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 190, 255), 2)
    cv2.putText(frame, subtitle, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 190, 255), 2)
    return frame


def wait_for_recording_start(
    cap,
    frame,
    raw_frame_index,
    frame_stride,
    display_wait_ms,
    show_window,
    window_name,
    delay_seconds,
):
    delay_seconds = max(float(delay_seconds or 0.0), 0.0)
    if delay_seconds <= 0:
        return frame, raw_frame_index, True

    started_at = time.time()
    latest_frame = frame

    while True:
        remaining = delay_seconds - (time.time() - started_at)
        if remaining <= 0:
            break

        ret, next_frame, raw_frame_index = read_sampled_frame(cap, frame_stride, raw_frame_index)
        if ret:
            latest_frame = next_frame

        preview = draw_recording_countdown(latest_frame.copy(), remaining)
        if show_window:
            cv2.imshow(window_name, resize_for_display(preview))
            key = cv2.waitKey(max(display_wait_ms, 1)) & 0xFF
            if key == ord("q"):
                return latest_frame, raw_frame_index, False
        else:
            time.sleep(min(max(display_wait_ms / 1000.0, 0.01), remaining))

    return latest_frame, raw_frame_index, True


def write_realtime_frames(writer, frame, start_time, record_fps, written_frames, max_elapsed=None):
    elapsed = max(time.time() - start_time, 0.0)
    if max_elapsed is not None:
        elapsed = min(elapsed, max_elapsed)

    target_frames = max(1, int(elapsed * record_fps))
    while written_frames < target_frames:
        writer.write(frame)
        written_frames += 1

    return written_frames


def record_demo(
    output_path=None,
    duration_seconds=120,
    record_fps=20.0,
    send_alerts=True,
    show_window=True,
    start_delay_seconds=3.0,
    debug_overlay=False,
    stable_single_person=True,
):
    setup_logging()
    if debug_overlay:
        CFG["show_demo_debug"] = True
    CFG["stable_single_person"] = bool(stable_single_person)

    output_path = build_output_path(output_path)
    os.makedirs(CFG["capture_dir"], exist_ok=True)

    database = FallDatabase(CFG["database_path"])
    telegram = TelegramAlerter(CFG["telegram_bot_token"], CFG["telegram_chat_id"])

    if send_alerts and not telegram.enabled:
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

    ret, frame, raw_frame_index = read_sampled_frame(cap, frame_stride, 0)
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame from camera.")

    sequence_buffers = {}
    engines = {}
    last_seen_frame = {}
    last_person_states = {}
    alert_hold_until = {}
    alert_hold_confidence = {}
    display_numbers = {}
    person_states = {}
    frame_index = 0
    written_frames = 0
    last_output_frame = None
    writer = None
    window_name = "Recording Fall Detection Demo"

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CFG["display_width"], CFG["display_height"])

    print(f"[REC] Recording to: {output_path}")
    print(f"[REC] Duration: {duration_seconds}s | Press Q to stop early | Press S to save snapshot")
    if CFG["stable_single_person"]:
        print("[REC] Single-person mode: using the largest detected person as stable ID 0.")
    else:
        print("[REC] Multi-person mode: using tracker IDs with handoff and Person labels.")
    if start_delay_seconds > 0:
        print(f"[REC] Preview delay: {start_delay_seconds:.1f}s. Recording starts after the countdown.")
    if frame_stride > 1:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        print(
            f"[REC] Sampling video source every {frame_stride} frames "
            f"({source_fps:.1f} FPS -> about {CFG['target_fps']:.1f} FPS)."
        )

    frame, raw_frame_index, should_record = wait_for_recording_start(
        cap,
        frame,
        raw_frame_index,
        frame_stride,
        display_wait_ms,
        show_window,
        window_name,
        start_delay_seconds,
    )
    if not should_record:
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE] Recording cancelled before start.")
        return None

    height, width = frame.shape[:2]
    writer = open_writer(output_path, width, height, record_fps)
    prev_time = time.time()
    start_time = time.time()

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                break

            if frame is None:
                ret, frame, raw_frame_index = read_sampled_frame(cap, frame_stride, raw_frame_index)
                if not ret:
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
                    "recovering": False,
                    "recovery_counter": 0,
                }

                if len(sequence_buffers[track_id]) == CFG["sequence_len"]:
                    fall_probability = predict_lstm_probability(lstm_model, sequence_buffers[track_id])
                    decision = engine.update(fall_probability, pose_meta)
                    label = decision["label"]
                    confidence = decision["smoothed_probability"]
                    result_probability = confidence if label == "fall" else 1.0 - confidence

                    if send_alerts and decision["should_alert"]:
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

                    if send_alerts and decision["should_alert"]:
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
                display_frame = draw_research_overlay(annotated, display_states, fps, raw_people)
            else:
                annotated = draw_filtered_pose(frame.copy(), result, person_states)
                display_frame = draw_multi_person_overlay(annotated, display_states, fps)
                for track_id, state in person_states.items():
                    display_frame = draw_person_status(
                        display_frame,
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

            written_frames = write_realtime_frames(
                writer,
                display_frame,
                start_time,
                record_fps,
                written_frames,
                duration_seconds,
            )
            last_output_frame = display_frame

            if show_window:
                cv2.imshow(window_name, resize_for_display(display_frame))

            key = cv2.waitKey(display_wait_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snap_path = os.path.join(CFG["capture_dir"], f"record_snap_{int(time.time())}.jpg")
                cv2.imwrite(snap_path, display_frame)
                print(f"[SNAP] Saved: {snap_path}")

            ret, frame, raw_frame_index = read_sampled_frame(cap, frame_stride, raw_frame_index)
            if not ret:
                frame = None

    finally:
        if writer is not None and last_output_frame is not None:
            final_elapsed = min(time.time() - start_time, duration_seconds)
            write_realtime_frames(
                writer,
                last_output_frame,
                start_time,
                record_fps,
                written_frames,
                final_elapsed,
            )
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"[DONE] Video saved: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Record a fall detection demo video.")
    parser.add_argument("--output", help="Output .mp4 path. Default: recordings/fall_demo_TIMESTAMP.mp4")
    parser.add_argument("--duration", type=int, default=120, help="Recording duration in seconds.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Preview countdown before writing the video. Use 0 to record immediately.",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Show people/FPS/fall-risk counters on top of the clean demo overlay.",
    )
    parser.add_argument(
        "--keep-tracker-ids",
        action="store_true",
        help="Use multi-person tracker IDs instead of stable single-person mode.",
    )
    parser.add_argument(
        "--single-person",
        action="store_true",
        help="Only follow the largest detected person as stable ID 0. This is the default.",
    )
    parser.add_argument("--no-alerts", action="store_true", help="Do not send Telegram alerts while recording.")
    parser.add_argument("--no-window", action="store_true", help="Do not show OpenCV preview window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_demo(
        output_path=args.output,
        duration_seconds=args.duration,
        record_fps=args.fps,
        send_alerts=not args.no_alerts,
        show_window=not args.no_window,
        start_delay_seconds=args.start_delay,
        debug_overlay=args.debug_overlay,
        stable_single_person=not args.keep_tracker_ids,
    )
