import os
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "fall_events.db"
RUNTIME_CONFIG_PATH = PROJECT_DIR / "models_lstm" / "lstm_runtime_config.json"
SEQUENCE_COMPARISON_PATH = PROJECT_DIR / "models_4_clean" / "sequence_comparison_30_35_40.csv"
LSTM_RESULTS_PATH = PROJECT_DIR / "models_lstm" / "lstm_results.json"
EVENT_STATUS_OPTIONS = ["new", "alert_sent", "alert_failed", "confirmed", "false_alarm", "resolved"]
EVENT_STATUS_LABELS = {
    "new": "New",
    "alert_sent": "Alert Sent",
    "alert_failed": "Alert Failed",
    "confirmed": "Confirmed",
    "false_alarm": "False Alarm",
    "resolved": "Resolved",
}


def ensure_schema():
    with sqlite3.connect(DB_PATH) as conn:
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


def load_events(limit):
    ensure_schema()
    with sqlite3.connect(DB_PATH) as conn:
        events = pd.read_sql_query(
            """
            SELECT
                id,
                created_at,
                camera_name,
                label,
                confidence,
                fall_counter,
                track_id,
                image_path,
                status,
                review_note,
                reviewed_at,
                telegram_sent,
                telegram_response
            FROM fall_events
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )

    if events.empty:
        return events

    events["created_at"] = pd.to_datetime(events["created_at"], errors="coerce")
    events["confidence"] = pd.to_numeric(events["confidence"], errors="coerce").fillna(0.0)
    events["fall_counter"] = pd.to_numeric(events["fall_counter"], errors="coerce").fillna(0).astype(int)
    events["telegram_sent"] = pd.to_numeric(events["telegram_sent"], errors="coerce").fillna(0).astype(int)
    events["track_id"] = pd.to_numeric(events["track_id"], errors="coerce")
    events["status"] = events["status"].fillna("new").astype(str)
    events["review_note"] = events["review_note"].fillna("").astype(str)
    events["reviewed_at"] = pd.to_datetime(events["reviewed_at"], errors="coerce")
    return events


def update_event_status(event_id, status, review_note):
    reviewed_at = pd.Timestamp.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE fall_events
            SET status = ?, review_note = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (status, review_note.strip(), reviewed_at, int(event_id)),
        )
        conn.commit()


def update_event_statuses(event_ids, status, review_note):
    if not event_ids:
        return 0

    reviewed_at = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = [(status, review_note.strip(), reviewed_at, int(event_id)) for event_id in event_ids]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            UPDATE fall_events
            SET status = ?, review_note = ?, reviewed_at = ?
            WHERE id = ?
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def format_status(status):
    return EVENT_STATUS_LABELS.get(str(status), str(status).replace("_", " ").title())


def status_badge_class(status):
    if status in {"alert_sent", "confirmed", "resolved"}:
        return "sent"
    if status in {"alert_failed", "false_alarm"}:
        return "pending"
    return "new"


def prepare_export(events):
    if events.empty:
        return ""

    export = events.copy()
    export["created_at"] = export["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export["reviewed_at"] = export["reviewed_at"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    export["telegram_sent"] = export["telegram_sent"].map(lambda value: "yes" if int(value) == 1 else "no")
    export["status"] = export["status"].map(format_status)
    return export.to_csv(index=False).encode("utf-8-sig")


def fmt_percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def existing_image_path(value):
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    fallback = Path.cwd() / path
    if fallback.exists():
        return fallback
    return None


def load_json_file(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_runtime_config():
    return load_json_file(RUNTIME_CONFIG_PATH) or {}


def load_sequence_comparison_rows():
    if not SEQUENCE_COMPARISON_PATH.exists():
        return []

    try:
        summary = pd.read_csv(SEQUENCE_COMPARISON_PATH)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return []

    if summary.empty:
        return []

    rows = []
    for _, row in summary.iterrows():
        seq_len = int(row.get("seq_len", 0))
        rows.append(
            {
                "model": f"LSTM seq{seq_len}",
                "model_path": row.get("best_h5_path"),
                "source": str(SEQUENCE_COMPARISON_PATH),
                "threshold": safe_float(row.get("best_threshold_by_val_f1")),
                "accuracy": safe_float(row.get("test_accuracy_05")),
                "precision": safe_float(row.get("test_precision_05")),
                "recall": safe_float(row.get("test_recall_05")),
                "f1": safe_float(row.get("test_f1_05")),
            }
        )
    return rows


def apply_filters(events, date_range, camera, person_id, event_status, telegram_status, min_confidence):
    filtered = events.copy()

    if filtered.empty:
        return filtered

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filtered = filtered[(filtered["created_at"] >= start_ts) & (filtered["created_at"] < end_ts)]

    if camera != "All":
        filtered = filtered[filtered["camera_name"] == camera]

    if person_id != "All":
        filtered = filtered[filtered["track_id"].fillna(-1).astype(int).astype(str) == person_id]

    if event_status != "All":
        filtered = filtered[filtered["status"] == event_status]

    if telegram_status == "Sent":
        filtered = filtered[filtered["telegram_sent"] == 1]
    elif telegram_status == "Not sent":
        filtered = filtered[filtered["telegram_sent"] == 0]

    filtered = filtered[filtered["confidence"] >= min_confidence]
    return filtered


def render_styles():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background: #0f1218;
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .hero {
            background: linear-gradient(120deg, #111827 0%, #1f2937 55%, #0f766e 100%);
            border-radius: 8px;
            padding: 24px 28px;
            color: #ffffff;
            margin-bottom: 18px;
            border: 1px solid #334155;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 760;
            letter-spacing: 0;
        }
        .hero p {
            margin: 8px 0 0 0;
            color: #d9e2ec;
            font-size: 0.98rem;
        }
        div[data-testid="stMetric"] {
            background: #171b23;
            border: 1px solid #2b3442;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stMetricLabel"] {
            color: #a8b3c2;
            font-size: 0.82rem;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-weight: 740;
        }
        div[data-testid="stMetricDelta"] {
            color: #86efac;
        }
        .panel {
            background: #171b23;
            border: 1px solid #2b3442;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }
        .status-badge {
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .sent {
            background: rgba(34, 197, 94, 0.16);
            color: #86efac;
        }
        .pending {
            background: rgba(248, 113, 113, 0.16);
            color: #fca5a5;
        }
        .new {
            background: rgba(96, 165, 250, 0.16);
            color: #93c5fd;
        }
        .muted {
            color: #a8b3c2;
            font-size: 0.9rem;
        }
        section[data-testid="stSidebar"] {
            background: #0b0f15;
            border-right: 1px solid #202938;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: inherit;
        }
        div[data-testid="stTabs"] button p {
            color: #cbd5e1;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: #ffffff;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #2dd4bf;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #2b3442;
            border-radius: 8px;
        }
        div[data-testid="stAlert"] {
            background: #111827;
            border: 1px solid #2b3442;
            color: #e5e7eb;
        }
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"],
        .stDateInput input,
        .stNumberInput input,
        .stTextInput input {
            background-color: #111827;
            color: #f8fafc;
            border-color: #334155;
        }
        .stButton button {
            background: #263142;
            color: #f8fafc;
            border: 1px solid #3b4658;
        }
        .stButton button:hover {
            background: #334155;
            color: #ffffff;
            border-color: #5b687a;
        }
        .stCodeBlock, pre {
            background: #0b0f15 !important;
            border: 1px solid #2b3442;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_delta(current, previous):
    if previous == 0:
        return None
    change = current - previous
    return f"{change:+d}"


def render_metrics(events):
    now = pd.Timestamp.now()
    today_start = now.normalize()
    yesterday_start = today_start - pd.Timedelta(days=1)

    total = len(events)
    today = len(events[events["created_at"] >= today_start]) if total else 0
    yesterday = len(events[(events["created_at"] >= yesterday_start) & (events["created_at"] < today_start)]) if total else 0
    sent = int(events["telegram_sent"].sum()) if total else 0
    failed = int((events["status"] == "alert_failed").sum()) if total else 0
    pending_review = int(events["status"].isin(["new", "alert_sent", "alert_failed"]).sum()) if total else 0
    avg_confidence = events["confidence"].mean() if total else 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Filtered Events", total)
    col2.metric("Today", today, delta=metric_delta(today, yesterday))
    col3.metric("Telegram Sent", sent, delta=f"{sent / total:.0%}" if total else "0%")
    col4.metric("Alert Failed", failed)
    col5.metric("Pending Review", pending_review)
    col6.metric("Avg Confidence", fmt_percent(avg_confidence))


def render_latest_event(events):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Latest Event")

    if events.empty:
        st.info("No events match the current filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    latest = events.sort_values("id", ascending=False).iloc[0]
    status = str(latest.get("status") or "new")
    status_class = status_badge_class(status)
    status_text = format_status(status)

    st.markdown(
        f"""
        <div class="muted">Event #{int(latest["id"])} | {latest["created_at"].strftime("%Y-%m-%d %H:%M:%S")}</div>
        <div style="height: 8px"></div>
        <span class="status-badge {status_class}">{status_text}</span>
        """,
        unsafe_allow_html=True,
    )

    detail_cols = st.columns(3)
    detail_cols[0].metric("Camera", str(latest["camera_name"]))
    person_text = "N/A" if pd.isna(latest["track_id"]) else str(int(latest["track_id"]))
    detail_cols[1].metric("Person ID", person_text)
    detail_cols[2].metric("Confidence", fmt_percent(latest["confidence"]))

    if str(latest.get("review_note") or "").strip():
        st.caption(f"Review note: {latest['review_note']}")

    image_path = existing_image_path(latest.get("image_path"))
    if image_path:
        st.image(str(image_path), caption=str(image_path), use_container_width=True)
    else:
        st.warning(f"Image not found: {latest.get('image_path')}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_timeline(events):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Event Timeline")

    if events.empty:
        st.info("No timeline data.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    timeline = (
        events.dropna(subset=["created_at"])
        .set_index("created_at")
        .resample("1H")
        .size()
        .rename("events")
        .reset_index()
    )
    st.bar_chart(timeline, x="created_at", y="events", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_events_table(events):
    st.subheader("Event History")

    if events.empty:
        st.info("No events match the current filters.")
        return

    display = events.copy()
    display["created_at"] = display["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display["telegram"] = display["telegram_sent"].map(lambda value: "Sent" if int(value) == 1 else "Not sent")
    display["status"] = display["status"].map(format_status)
    display["reviewed_at"] = display["reviewed_at"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    display["track_id"] = display["track_id"].apply(lambda value: "N/A" if pd.isna(value) else str(int(value)))
    display = display[
        [
            "id",
            "created_at",
            "camera_name",
            "track_id",
            "label",
            "status",
            "confidence",
            "fall_counter",
            "telegram",
            "reviewed_at",
            "review_note",
            "image_path",
        ]
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "created_at": st.column_config.TextColumn("Time", width="medium"),
            "camera_name": st.column_config.TextColumn("Camera", width="small"),
            "track_id": st.column_config.TextColumn("Person", width="small"),
            "label": st.column_config.TextColumn("Label", width="small"),
            "status": st.column_config.TextColumn("Status", width="small"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "fall_counter": st.column_config.NumberColumn("Frames", width="small"),
            "telegram": st.column_config.TextColumn("Telegram", width="small"),
            "reviewed_at": st.column_config.TextColumn("Reviewed At", width="medium"),
            "review_note": st.column_config.TextColumn("Review Note", width="large"),
            "image_path": st.column_config.TextColumn("Capture", width="large"),
        },
    )


def render_event_management(events):
    st.subheader("Event Management")

    if events.empty:
        st.info("No events match the current filters.")
        return

    st.download_button(
        "Export filtered events CSV",
        data=prepare_export(events),
        file_name=f"fall_events_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    sorted_events = events.sort_values("id", ascending=False)
    event_options = {
        int(row["id"]): (
            f"#{int(row['id'])} | {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
            f" | {format_status(row['status'])} | {fmt_percent(row['confidence'])}"
        )
        for _, row in sorted_events.iterrows()
    }

    selected_event_id = st.selectbox(
        "Select event",
        options=list(event_options.keys()),
        format_func=lambda value: event_options[value],
    )
    selected = sorted_events[sorted_events["id"] == selected_event_id].iloc[0]

    left, right = st.columns([0.9, 1.1])
    with left:
        image_path = existing_image_path(selected.get("image_path"))
        if image_path:
            st.image(str(image_path), caption=str(image_path), use_container_width=True)
        else:
            st.warning(f"Image not found: {selected.get('image_path')}")

    with right:
        person_text = "N/A" if pd.isna(selected["track_id"]) else str(int(selected["track_id"]))
        st.write(f"Camera: `{selected['camera_name']}`")
        st.write(f"Person ID: `{person_text}`")
        st.write(f"Confidence: `{fmt_percent(selected['confidence'])}`")
        st.write(f"Current status: `{format_status(selected['status'])}`")

        current_status = str(selected["status"])
        status_index = EVENT_STATUS_OPTIONS.index(current_status) if current_status in EVENT_STATUS_OPTIONS else 0
        with st.form("event_status_form"):
            next_status = st.selectbox(
                "Update status",
                EVENT_STATUS_OPTIONS,
                index=status_index,
                format_func=format_status,
            )
            review_note = st.text_area(
                "Review note",
                value=str(selected.get("review_note") or ""),
                placeholder="Example: Confirmed by operator after checking snapshot.",
                height=110,
            )
            submitted = st.form_submit_button("Save event review", use_container_width=True)

        if submitted:
            update_event_status(selected_event_id, next_status, review_note)
            st.success(f"Event #{selected_event_id} updated to {format_status(next_status)}.")
            time.sleep(0.3)
            st.rerun()

    st.divider()
    st.subheader("Batch Review")

    batch_options = {
        int(row["id"]): (
            f"#{int(row['id'])} | {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
            f" | {format_status(row['status'])} | {row['camera_name']}"
        )
        for _, row in sorted_events.iterrows()
    }
    default_batch_ids = sorted_events[
        sorted_events["status"].isin(["new", "alert_sent", "alert_failed"])
    ]["id"].astype(int).head(10).tolist()

    with st.form("batch_event_status_form"):
        selected_ids = st.multiselect(
            "Events to update",
            options=list(batch_options.keys()),
            default=default_batch_ids,
            format_func=lambda value: batch_options[value],
        )
        batch_status = st.selectbox(
            "Batch status",
            EVENT_STATUS_OPTIONS,
            index=EVENT_STATUS_OPTIONS.index("resolved"),
            format_func=format_status,
        )
        batch_note = st.text_area(
            "Batch review note",
            placeholder="Example: Reviewed together after checking the exported evidence.",
            height=90,
        )
        batch_submitted = st.form_submit_button("Update selected events", use_container_width=True)

    if batch_submitted:
        updated_count = update_event_statuses(selected_ids, batch_status, batch_note)
        st.success(f"Updated {updated_count} events to {format_status(batch_status)}.")
        time.sleep(0.3)
        st.rerun()


def render_reports(events):
    st.subheader("Reports")

    if events.empty:
        st.info("No report data for the current filters.")
        return

    report_data = events.copy()
    report_data["date"] = report_data["created_at"].dt.date
    report_data["status_label"] = report_data["status"].map(format_status)

    daily_counts = report_data.groupby("date").size().rename("events").reset_index()
    status_counts = report_data.groupby("status_label").size().rename("events").reset_index()
    camera_summary = (
        report_data.groupby("camera_name")
        .agg(
            events=("id", "count"),
            avg_confidence=("confidence", "mean"),
            telegram_sent=("telegram_sent", "sum"),
            pending_review=("status", lambda values: values.isin(["new", "alert_sent", "alert_failed"]).sum()),
        )
        .reset_index()
        .sort_values("events", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("Events by day")
        st.bar_chart(daily_counts, x="date", y="events", use_container_width=True)
    with col2:
        st.write("Events by status")
        st.bar_chart(status_counts, x="status_label", y="events", use_container_width=True)

    st.write("Camera summary")
    st.dataframe(
        camera_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "camera_name": st.column_config.TextColumn("Camera", width="large"),
            "events": st.column_config.NumberColumn("Events", width="small"),
            "avg_confidence": st.column_config.ProgressColumn(
                "Avg Confidence",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "telegram_sent": st.column_config.NumberColumn("Telegram Sent", width="small"),
            "pending_review": st.column_config.NumberColumn("Pending Review", width="small"),
        },
    )


def render_capture_grid(events):
    st.subheader("Captured Images")

    if events.empty:
        st.info("No captures match the current filters.")
        return

    capture_rows = []
    for _, row in events.iterrows():
        image_path = existing_image_path(row.get("image_path"))
        if image_path:
            capture_rows.append((row, image_path))

    if not capture_rows:
        st.info("Events exist, but no image files were found.")
        return

    cols = st.columns(3)
    for index, (row, image_path) in enumerate(capture_rows[:18]):
        with cols[index % 3]:
            person_text = "N/A" if pd.isna(row["track_id"]) else str(int(row["track_id"]))
            caption = (
                f"#{int(row['id'])} | {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
                f" | ID {person_text} | {fmt_percent(row['confidence'])}"
            )
            st.image(str(image_path), caption=caption, use_container_width=True)


def render_system_panel(raw_events):
    st.subheader("System")

    db_exists = DB_PATH.exists()
    db_size = DB_PATH.stat().st_size if db_exists else 0
    image_count = 0
    capture_dir = PROJECT_DIR / "captures"
    if capture_dir.exists():
        image_count = len(list(capture_dir.glob("*.jpg")))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Database", str(DB_PATH))
    col2.metric("DB Size", f"{db_size / 1024:.1f} KB")
    col3.metric("Loaded Rows", len(raw_events))
    col4.metric("Capture Files", image_count)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    runtime_config = load_runtime_config()
    active_model = os.getenv("LSTM_MODEL_PATH") or runtime_config.get("model_path", "models_4_clean/seq30/lstm_best_seq30.h5")
    active_sequence_len = os.getenv("SEQUENCE_LEN") or runtime_config.get("sequence_len", 30)

    st.write("Active model")
    st.code(
        f"LSTM_MODEL_PATH={active_model}\n"
        f"SEQUENCE_LEN={active_sequence_len}",
        language="text",
    )

    st.write("Active runtime files")
    st.code(
        "python main_telegram_database.py\n"
        "streamlit run dashboard_streamlit.py\n"
        "python record.py --duration 120 --no-alerts\n"
        "python extract_frame_feature_cache.py --datasets urfd,leifall,gmdcsa24,mcfd --output-dir frame_feature_cache_v3 --skip-existing",
        language="powershell",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_comparison():
    st.subheader("LSTM Model Metrics")

    rows = []
    rows.extend(load_sequence_comparison_rows())

    lstm_results = load_json_file(LSTM_RESULTS_PATH)
    if not rows and lstm_results and lstm_results.get("test"):
        rows.append(
            {
                "model": "LSTM",
                "model_path": lstm_results.get("runtime_config", {}).get("model_path"),
                "source": str(LSTM_RESULTS_PATH),
                "threshold": lstm_results.get("test", {}).get("threshold")
                or lstm_results.get("selected_threshold", {}).get("threshold"),
                "accuracy": lstm_results["test"].get("accuracy"),
                "precision": lstm_results["test"].get("precision"),
                "recall": lstm_results["test"].get("recall"),
                "f1": lstm_results["test"].get("f1"),
            }
        )

    if not rows:
        st.info("No LSTM results found. Add `models_4_clean/sequence_comparison_30_35_40.csv` to populate this view.")
        return

    metrics = pd.DataFrame(rows)
    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
        column_config={
            "model": st.column_config.TextColumn("Model", width="small"),
            "model_path": st.column_config.TextColumn("Model Path", width="large"),
            "source": st.column_config.TextColumn("Source", width="large"),
            "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
            "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.0, max_value=1.0, format="%.2f"),
            "precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.2f"),
            "recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.2f"),
            "f1": st.column_config.ProgressColumn("F1", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    chart_data = metrics.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    st.bar_chart(chart_data, use_container_width=True)


st.set_page_config(
    page_title="Fall Detection Dashboard",
    page_icon="!",
    layout="wide",
)

render_styles()
ensure_schema()

st.markdown(
    """
    <div class="hero">
        <h1>Fall Detection Dashboard</h1>
        <p>Realtime event monitoring, operator review, person tracking, Telegram delivery status, and captured evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dashboard")
    auto_refresh = st.toggle("Auto refresh", value=True)
    refresh_seconds = st.slider("Refresh interval", 2, 30, 5)
    row_limit = st.slider("Rows loaded", 50, 1000, 300, 50)
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

raw_events = load_events(row_limit)

with st.sidebar:
    st.divider()
    st.header("Filters")

    if raw_events.empty:
        date_range = None
        camera = "All"
        person_id = "All"
        telegram_status = "All"
        event_status = "All"
        min_confidence = 0.0
        st.caption("No data available.")
    else:
        min_date = raw_events["created_at"].min().date()
        max_date = raw_events["created_at"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date))

        cameras = ["All"] + sorted(raw_events["camera_name"].dropna().astype(str).unique().tolist())
        camera = st.selectbox("Camera", cameras)

        people = (
            raw_events["track_id"]
            .dropna()
            .astype(int)
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        person_id = st.selectbox("Person ID", ["All"] + people)
        statuses = ["All"] + EVENT_STATUS_OPTIONS
        event_status = st.selectbox(
            "Event status",
            statuses,
            format_func=lambda value: "All" if value == "All" else format_status(value),
        )
        telegram_status = st.segmented_control("Telegram", ["All", "Sent", "Not sent"], default="All")
        min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

filtered_events = apply_filters(
    raw_events,
    date_range=date_range,
    camera=camera,
    person_id=person_id,
    event_status=event_status,
    telegram_status=telegram_status,
    min_confidence=min_confidence,
)

render_metrics(filtered_events)

overview_tab, events_tab, manage_tab, reports_tab, captures_tab, models_tab, system_tab = st.tabs(
    ["Overview", "Events", "Manage", "Reports", "Captures", "Models", "System"]
)

with overview_tab:
    left, right = st.columns([1.15, 0.85])
    with left:
        render_timeline(filtered_events)
    with right:
        render_latest_event(filtered_events)

with events_tab:
    render_events_table(filtered_events)

with manage_tab:
    render_event_management(filtered_events)

with reports_tab:
    render_reports(filtered_events)

with captures_tab:
    render_capture_grid(filtered_events)

with models_tab:
    render_model_comparison()

with system_tab:
    render_system_panel(raw_events)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
