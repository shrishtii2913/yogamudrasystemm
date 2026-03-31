"""
=============================================================================
pose_detector.py — Yoga Mudra Detection System (Python / MediaPipe Component)
=============================================================================
ACADEMIC PROJECT: Hybrid C++ & Python Yoga Mudra Detection System
Component Role : Python handles real-time hand landmark detection via MediaPipe
                 and writes structured JSON results to a shared file so that
                 the C++ visualizer can read them without network overhead.

Libraries Used:
  - OpenCV   : Camera capture and basic frame display on the Python side
  - MediaPipe: Google's hand-landmark model (21 key-points per hand)
  - json     : IPC bridge — writes output.json consumed by the C++ process
  - logging  : Structured session logging to logs/session.log
=============================================================================
"""

import cv2
import mediapipe as mp
import math
import json
import time
import logging
import os
import sys

# ─────────────────────────────────────────────
# Logging setup — writes to shared logs/ folder
# ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/session.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MediaPipe Hands initialisation
#   max_num_hands=1    : single-hand detection for clarity
#   min_detection_confidence: threshold before landmarks are reported
#   min_tracking_confidence : threshold to keep tracking between frames
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.60,
)

# ──────────────────────────────────────────────────────────────────────────────
# MUDRA DEFINITIONS
# Each mudra is a named pose with a detector function that returns (bool, float)
#   bool  → mudra matches
#   float → raw confidence in [0, 1]
#
# Landmark indices (MediaPipe 21-point hand model):
#   0  = WRIST
#   4  = THUMB_TIP
#   8  = INDEX_TIP       5  = INDEX_MCP
#   12 = MIDDLE_TIP      9  = MIDDLE_MCP
#   16 = RING_TIP       13  = RING_MCP
#   20 = PINKY_TIP      17  = PINKY_MCP
# ──────────────────────────────────────────────────────────────────────────────

def dist(p1, p2) -> float:
    """Euclidean distance between two normalised MediaPipe landmarks."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def finger_extended(tip, mcp) -> bool:
    """
    A finger is considered 'extended' when its tip is higher on screen
    (smaller y value in normalised coords) than its MCP knuckle.
    """
    return tip.y < mcp.y


def detect_gyan_mudra(lm) -> tuple[bool, float]:
    """
    Gyan Mudra (Chin Mudra):
    Thumb tip touches Index finger tip; remaining fingers are extended.
    Confidence scales inversely with thumb-index distance.
    """
    d = dist(lm[4], lm[8])
    touch_threshold = 0.06  # normalised distance
    middle_ext = finger_extended(lm[12], lm[9])
    ring_ext   = finger_extended(lm[16], lm[13])
    pinky_ext  = finger_extended(lm[20], lm[17])

    if d < touch_threshold and middle_ext and ring_ext and pinky_ext:
        confidence = round(1.0 - (d / touch_threshold), 3)
        return True, confidence
    return False, 0.0


def detect_chin_mudra(lm) -> tuple[bool, float]:
    """
    Chin Mudra (palms-up variant):
    Same finger topology as Gyan but we additionally check that the wrist
    y-coordinate is below all fingertips (palm facing upward heuristic).
    """
    matched, conf = detect_gyan_mudra(lm)
    # Palm-up heuristic: wrist below mid-finger tip
    palm_up = lm[0].y > lm[12].y
    if matched and palm_up:
        return True, min(conf + 0.05, 1.0)
    return False, 0.0


def detect_abhaya_mudra(lm) -> tuple[bool, float]:
    """
    Abhaya Mudra (gesture of fearlessness / open palm):
    All five fingers fully extended and spread — no finger touching another.
    """
    tips  = [lm[4],  lm[8],  lm[12], lm[16], lm[20]]
    mcps  = [lm[2],  lm[5],  lm[9],  lm[13], lm[17]]

    all_extended = all(finger_extended(t, m) for t, m in zip(tips[1:], mcps[1:]))
    # Thumb is extended if tip is to the left of its IP joint (right hand)
    thumb_ext = lm[4].x < lm[3].x

    if all_extended and thumb_ext:
        # Confidence based on average extension magnitude
        magnitudes = [abs(t.y - m.y) for t, m in zip(tips[1:], mcps[1:])]
        conf = round(min(sum(magnitudes) / len(magnitudes) * 5, 1.0), 3)
        return True, conf
    return False, 0.0


def detect_dhyana_mudra(lm) -> tuple[bool, float]:
    """
    Dhyana Mudra (meditation gesture):
    All fingers curled inward (tips below MCPs) — closed fist / cupped hand.
    """
    tips = [lm[8], lm[12], lm[16], lm[20]]
    mcps = [lm[5], lm[9],  lm[13], lm[17]]

    all_curled = all(not finger_extended(t, m) for t, m in zip(tips, mcps))
    if all_curled:
        depths = [abs(t.y - m.y) for t, m in zip(tips, mcps)]
        conf = round(min(sum(depths) / len(depths) * 4, 1.0), 3)
        return True, conf
    return False, 0.0


def detect_shuni_mudra(lm) -> tuple[bool, float]:
    """
    Shuni Mudra:
    Thumb tip touches Middle finger tip; index, ring, pinky extended.
    """
    d = dist(lm[4], lm[12])
    threshold = 0.06
    index_ext = finger_extended(lm[8],  lm[5])
    ring_ext  = finger_extended(lm[16], lm[13])
    pinky_ext = finger_extended(lm[20], lm[17])

    if d < threshold and index_ext and ring_ext and pinky_ext:
        conf = round(1.0 - (d / threshold), 3)
        return True, conf
    return False, 0.0


# Ordered list — first match wins
MUDRA_DETECTORS = [
    ("Gyan Mudra",   detect_gyan_mudra),
    ("Chin Mudra",   detect_chin_mudra),
    ("Abhaya Mudra", detect_abhaya_mudra),
    ("Dhyana Mudra", detect_dhyana_mudra),
    ("Shuni Mudra",  detect_shuni_mudra),
]

# Brief description shown in the C++ feedback overlay
MUDRA_INFO = {
    "Gyan Mudra"  : "Knowledge | Improves concentration & memory",
    "Chin Mudra"  : "Consciousness | Calms the mind",
    "Abhaya Mudra": "Fearlessness | Offers protection & peace",
    "Dhyana Mudra": "Meditation | Deepens focus & serenity",
    "Shuni Mudra" : "Patience | Enhances intuition",
    "No Pose"     : "No recognised mudra detected",
}


# ──────────────────────────────────────────────────────────────────────────────
# JSON IPC BRIDGE
# The shared file path must match the C++ reader path.
# We write atomically: write to a temp file then rename, preventing the C++
# side from reading a half-written file.
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_JSON = "shared/output.json"
TEMP_JSON   = "shared/output.tmp.json"
os.makedirs("shared", exist_ok=True)


def write_output(pose: str, confidence: float, fps: float, hand_detected: bool):
    """
    Writes detection results to output.json for the C++ visualizer.
    Fields:
      pose          : mudra name string
      confidence    : float [0, 1]
      description   : human-readable mudra meaning
      timestamp     : Unix epoch for the C++ logger
      fps           : detection frame rate
      hand_detected : whether a hand was found this frame
    """
    payload = {
        "pose"         : pose,
        "confidence"   : round(confidence, 4),
        "description"  : MUDRA_INFO.get(pose, ""),
        "timestamp"    : round(time.time(), 3),
        "fps"          : round(fps, 1),
        "hand_detected": hand_detected,
    }
    # Atomic write
    with open(TEMP_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(TEMP_JSON, OUTPUT_JSON)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN DETECTION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Yoga Mudra Detection System — Python Component ===")
    log.info("Press ESC in the Python window to exit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera. Check device permissions.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # FPS tracking
    prev_time = time.time()
    frame_count = 0
    session_stats: dict[str, int] = {}  # pose → detection count

    log.info("Camera opened successfully. Starting detection loop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame capture failed — retrying.")
            continue

        # ── MediaPipe processing ──────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # minor speed optimisation
        result = hands.process(rgb)
        rgb.flags.writeable = True

        # ── FPS calculation ───────────────────────────────────────────────
        frame_count += 1
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── Mudra detection ───────────────────────────────────────────────
        pose, confidence, hand_detected = "No Pose", 0.0, False

        if result.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark  # list of 21 NormalizedLandmark

                # Draw skeleton on Python preview
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Try each mudra detector in priority order
                for name, detector in MUDRA_DETECTORS:
                    matched, conf = detector(lm)
                    if matched:
                        pose, confidence = name, conf
                        break  # first matching mudra wins

        # ── Write IPC JSON ────────────────────────────────────────────────
        write_output(pose, confidence, fps, hand_detected)

        # ── Update session stats ──────────────────────────────────────────
        session_stats[pose] = session_stats.get(pose, 0) + 1

        # ── Python preview overlay ────────────────────────────────────────
        status_color = (0, 220, 100) if hand_detected else (60, 60, 200)
        cv2.putText(frame, f"Pose: {pose}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}  FPS: {fps:.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 50), 2)
        cv2.putText(frame, "Python / MediaPipe", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Yoga Mudra — Python Detector", frame)

        # ── Periodic logging ──────────────────────────────────────────────
        if frame_count % 60 == 0:
            log.info(f"Frame {frame_count} | Pose: {pose} | Conf: {confidence:.3f} | FPS: {fps:.1f}")

        if cv2.waitKey(1) == 27:  # ESC
            break

    # ── Teardown ──────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    log.info("=== Session complete. Detection summary ===")
    for p, count in sorted(session_stats.items(), key=lambda x: -x[1]):
        log.info(f"  {p:<20}: {count:>5} frames")
    log.info("Python component shut down.")


if __name__ == "__main__":
    main()
