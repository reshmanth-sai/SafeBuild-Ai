from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
import time
from collections import deque

HF_REPO_ID = "Hexmon/vyra-yolo-ppe-detection"
HF_FILENAME = "best.pt"
CONFIDENCE_THRESHOLD = 0.25
TRACKER_CONFIG = "custom_bytetrack.yaml"
TRACK_BUFFER_FRAMES = 90.0

try:
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = YOLO(weights_path)
    print(f"Loaded PPE model: {HF_REPO_ID}")
    print(f"Classes this model knows: {model.names}")
except Exception as e:
    print(f"Could not load '{HF_REPO_ID}': {e}")
    print("Fallback: using generic pretrained model (yolo11n.pt).")
    model = YOLO("yolo11n.pt")

# Map class names to IDs for fast lookup
name_to_id = {name: cls_id for cls_id, name in model.names.items()}
PERSON_CLS_ID = name_to_id.get("Person", 11)
NO_HARDHAT_CLS_ID = name_to_id.get("NO-Hardhat", 8)
NO_VEST_CLS_ID = name_to_id.get("NO-Safety Vest", 10)
FALL_DETECTED_CLS_ID = name_to_id.get("Fall-Detected", 0)
HARDHAT_CLS_ID = name_to_id.get("Hardhat", 3)
VEST_CLS_ID = name_to_id.get("Safety Vest", 13)

WORKER_CLASS_IDS = {
    PERSON_CLS_ID,
    NO_VEST_CLS_ID,
    NO_HARDHAT_CLS_ID,
    HARDHAT_CLS_ID,
    VEST_CLS_ID,
    FALL_DETECTED_CLS_ID,
}

cap = None

def init_webcam():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam — check camera permissions/index.")
        print(f"Webcam feed started (Tracker: {TRACKER_CONFIG}, Conf: {CONFIDENCE_THRESHOLD}). Press 'q' to quit.")
    return cap

# ---------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def is_item_associated_with_person(item_box, person_box, padding_pct=0.25):
    """
    Check if the center point of item_box lies within person_box (with 25% padding).
    Padded person_box accounts for head/torso crop alignment.
    """
    ix1, iy1, ix2, iy2 = item_box
    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    ph = py2 - py1
    epx1 = px1 - pw * padding_pct
    epy1 = py1 - ph * padding_pct
    epx2 = px2 + pw * padding_pct
    epy2 = py2 + ph * padding_pct
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0
    return (epx1 <= cx <= epx2) and (epy1 <= cy <= epy2)

import requests
from datetime import datetime, timezone

# Backend HTTP integration URL & Demo Worker ID
BACKEND_URL = "http://localhost:8000/events/safety"
DEMO_WORKER_ID = "W001"

EVENT_TYPE_MAP = {
    "MISSING HELMET": ("MISSING_HELMET", "MEDIUM"),
    "MISSING VEST": ("MISSING_VEST", "MEDIUM"),
    "ZONE INTRUSION": ("ZONE_INTRUSION", "HIGH"),
    "FALL DETECTED": ("FALL_DETECTED", "HIGH"),
}

def send_safety_event_to_backend(violation_name, track_id, conf, zone_name=None):
    """
    Send safety violation event payload to FastAPI backend over HTTP POST.
    Fails gracefully with warning log if backend server is unreachable.
    """
    mapped = EVENT_TYPE_MAP.get(violation_name)
    if not mapped:
        return
    backend_event_type, severity = mapped
    zone_val = "Restricted Danger Zone" if violation_name == "ZONE INTRUSION" else zone_name

    payload = {
        "worker_id": DEMO_WORKER_ID,
        "event_type": backend_event_type,
        "track_id": int(track_id),
        "severity": severity,
        "confidence": round(float(conf), 2),
        "zone": zone_val,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=0.5)
        if resp.status_code == 201:
            created_event = resp.json()
            print(f"   └─> [POST BACKEND SUCCESS] Event #{created_event.get('id')} ({backend_event_type}) stored in DB")
        else:
            print(f"   └─> [WARNING] Backend HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"   └─> [WARNING] Could not send event to backend: {e}")

# ---------------------------------------------------------------------------
# 4. STATE ENGINE & DEBOUNCE DICTIONARIES
# ---------------------------------------------------------------------------
seen_track_ids = set()
highest_track_id = 0
script_start_time = time.time()

# Per-track active violation state: {track_id: set(["MISSING HELMET", ...])}
track_active_violations = {}
# Per-track last seen timestamp: {track_id: timestamp_seconds}
track_last_seen = {}

# Per-track zone hysteresis state (5-frame rolling window)
track_zone_history = {}   # {track_id: deque(maxlen=5)}
track_zone_smoothed = {}  # {track_id: bool}

# Rolling FPS calculation over last 30 frames for dynamic track expiration calculation
frame_timestamps = deque(maxlen=30)
current_fps = 10.0  # default initial estimate until frames accumulate

# ---------------------------------------------------------------------------
# 5. LIVE LOOP WITH BYTETRACK & DEBOUNCED RULE ENGINE
# ---------------------------------------------------------------------------
cap = init_webcam()
while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from webcam.")
        break

    h, w, _ = frame.shape
    current_time = time.time()
    elapsed_sec = current_time - script_start_time

    # Update dynamic FPS calculation
    frame_timestamps.append(current_time)
    if len(frame_timestamps) > 1:
        fps_calc = (len(frame_timestamps) - 1) / (frame_timestamps[-1] - frame_timestamps[0])
        if fps_calc > 0.5:
            current_fps = fps_calc

    # Dynamically match Python track expiration window to ByteTrack's 90-frame buffer
    track_expiration_sec = TRACK_BUFFER_FRAMES / current_fps

    # Define hardcoded Danger Zone polygon (bottom-left quadrant of feed)
    zone_polygon = np.array([
        [int(w * 0.05), int(h * 0.45)],
        [int(w * 0.45), int(h * 0.45)],
        [int(w * 0.45), int(h * 0.95)],
        [int(w * 0.05), int(h * 0.95)]
    ], np.int32)

    # Multi-Object Tracking with custom_bytetrack.yaml
    results = model.track(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        tracker=TRACKER_CONFIG,
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Draw Danger Zone on screen (Red polygon)
    cv2.polylines(annotated_frame, [zone_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(
        annotated_frame,
        "RESTRICTED DANGER ZONE",
        (zone_polygon[0][0] + 5, zone_polygon[0][1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        # Separate Person/Worker detections and Target items
        worker_detections = []
        no_hardhat_boxes = []
        no_vest_boxes = []
        fall_detected_boxes = []

        cls_list = boxes.cls.cpu().numpy()
        xyxy_list = boxes.xyxy.cpu().numpy()
        conf_list = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(boxes)
        id_list = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)

        for i in range(len(boxes)):
            cls_id = int(cls_list[i])
            box = xyxy_list[i]
            conf = float(conf_list[i])
            track_id = int(id_list[i]) if id_list[i] is not None else None

            # Collect violation item boxes for spatial association
            if cls_id == NO_HARDHAT_CLS_ID:
                no_hardhat_boxes.append(box)
            elif cls_id == NO_VEST_CLS_ID:
                no_vest_boxes.append(box)
            elif cls_id == FALL_DETECTED_CLS_ID:
                fall_detected_boxes.append(box)

            # REQUIRE BOTH valid track_id AND cls_id in WORKER_CLASS_IDS
            if track_id is not None and cls_id in WORKER_CLASS_IDS:
                worker_detections.append((track_id, cls_id, box, conf))
                track_last_seen[track_id] = current_time
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    cls_name = model.names.get(cls_id, f"Class_{cls_id}")
                    print(f"[{elapsed_sec:06.2f}s] [NEW TRACK DETECTED] Track ID: {track_id} (label: {cls_name}, conf: {conf:.2f})")
                if track_id > highest_track_id:
                    highest_track_id = track_id

        # Check per-track PPE violations, Fall detection, & Zone intrusion
        for track_id, p_cls_id, p_box, p_conf in worker_detections:
            if track_id is None:
                continue

            track_str = f"Track {track_id}"
            current_violations = set()

            # 1. PPE Check: MISSING VEST
            if p_cls_id == NO_VEST_CLS_ID:
                current_violations.add("MISSING VEST")
            else:
                for nv_box in no_vest_boxes:
                    if is_item_associated_with_person(nv_box, p_box):
                        current_violations.add("MISSING VEST")
                        break

            # 2. PPE Check: MISSING HELMET
            if p_cls_id == NO_HARDHAT_CLS_ID:
                current_violations.add("MISSING HELMET")
            else:
                for nh_box in no_hardhat_boxes:
                    if is_item_associated_with_person(nh_box, p_box):
                        current_violations.add("MISSING HELMET")
                        break

            # 3. Fall Detection Check (Class 0: Fall-Detected)
            if p_cls_id == FALL_DETECTED_CLS_ID:
                current_violations.add("FALL DETECTED")
            else:
                for fd_box in fall_detected_boxes:
                    if is_item_associated_with_person(fd_box, p_box):
                        current_violations.add("FALL DETECTED")
                        break

            # 4. Zone Intrusion Check with Hysteresis (5-frame majority filter)
            foot_x = (p_box[0] + p_box[2]) / 2.0
            foot_y = p_box[3]
            raw_in_zone = cv2.pointPolygonTest(zone_polygon, (foot_x, foot_y), False) >= 0

            if track_id not in track_zone_history:
                track_zone_history[track_id] = deque(maxlen=5)
                track_zone_smoothed[track_id] = False

            history = track_zone_history[track_id]
            history.append(raw_in_zone)

            prev_smoothed = track_zone_smoothed[track_id]
            true_count = sum(history)
            false_count = len(history) - true_count

            # Require majority (at least 4 out of 5 frames) to flip state
            if not prev_smoothed and true_count >= 4:
                track_zone_smoothed[track_id] = True
            elif prev_smoothed and false_count >= 4:
                track_zone_smoothed[track_id] = False

            if track_zone_smoothed[track_id]:
                current_violations.add("ZONE INTRUSION")

            # State Transition Debouncing
            previous_violations = track_active_violations.get(track_id, set())

            # Detect newly started violations (Prints ONCE on onset & POSTs to backend)
            newly_started = current_violations - previous_violations
            for v in sorted(newly_started):
                print(f"[{elapsed_sec:06.2f}s] [ALERT STARTED] {track_str}: {v}")
                send_safety_event_to_backend(v, track_id, p_conf)

            # Detect resolved violations (Prints ONCE on resolution)
            newly_resolved = previous_violations - current_violations
            for v in sorted(newly_resolved):
                print(f"[{elapsed_sec:06.2f}s] [ALERT RESOLVED] {track_str}: {v}")

            # Update active violation state for this track
            track_active_violations[track_id] = current_violations

            # Draw visual alert badge above person head on screen
            if current_violations:
                alert_text = f"{track_str}: " + ", ".join(sorted(current_violations))
                px1, py1 = int(p_box[0]), int(p_box[1])
                cv2.putText(
                    annotated_frame,
                    alert_text,
                    (px1, max(20, py1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    # ---------------------------------------------------------------------------
    # 6. MEMORY PRUNING FOR EXPIRED TRACKS
    # ---------------------------------------------------------------------------
    expired_track_ids = [
        tid for tid, last_t in track_last_seen.items()
        if (current_time - last_t) > track_expiration_sec
    ]
    for tid in expired_track_ids:
        # Resolve any active violations before purging
        active_v = track_active_violations.get(tid, set())
        for v in sorted(active_v):
            print(f"[{elapsed_sec:06.2f}s] [ALERT RESOLVED - TRACK EXPIRED] Track {tid}: {v}")
        
        track_active_violations.pop(tid, None)
        track_last_seen.pop(tid, None)
        track_zone_history.pop(tid, None)
        track_zone_smoothed.pop(tid, None)

    cv2.imshow("SafeBuild AI — Day 1: PPE Detection & Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()








