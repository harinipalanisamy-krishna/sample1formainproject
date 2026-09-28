"""
Streamlit app: Edge AI Pedestrian & Animal Crossing Intention Prediction (ADAS)
Upload a video -> YOLOv8 detect+track -> MediaPipe pose -> features -> rule-based intent.
"""
import math
import os
import subprocess
import tempfile
from collections import defaultdict, deque

import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="ADAS Crossing Intention", layout="wide")

PERSON = 0
ANIMALS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}  # COCO animal classes
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 11, 12, 23, 24, 27, 28
HIST = 10
MAX_W = 640  # resize for speed on cloud


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def features(pts):
    sh = mid(pts[L_SH], pts[R_SH])
    hip = mid(pts[L_HIP], pts[R_HIP])
    lean = math.degrees(math.atan2(sh[0] - hip[0], -(sh[1] - hip[1])))
    sw = abs(pts[L_SH][0] - pts[R_SH][0]) + 1e-6
    head = (pts[NOSE][0] - sh[0]) / sw
    return lean, head, mid(pts[L_ANK], pts[R_ANK])


def process(path, conf, road_frac, lean_t, head_t, speed_t, max_frames, bar):
    cap = cv2.VideoCapture(path)
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_W / W0)
    W, H = int(W0 * scale), int(H0 * scale)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames, max_frames)

    raw_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    model = YOLO("yolov8n.pt")  # new instance per run resets the tracker
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                  min_detection_confidence=0.5)
    conn = mp.solutions.pose.POSE_CONNECTIONS

    ank_hist = defaultdict(lambda: deque(maxlen=HIST))
    cx_hist = defaultdict(lambda: deque(maxlen=HIST))
    rows, samples, alerts, n = [], [], 0, 0
    road_px = int(road_frac * W) if road_frac > 0 else None

    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        frame = cv2.resize(frame, (W, H))
        if road_px is not None:
            cv2.line(frame, (road_px, 0), (road_px, H), (0, 255, 255), 2)
            cv2.putText(frame, "ROAD", (road_px + 6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        res = model.track(frame, persist=True, conf=conf,
                          classes=[PERSON] + sorted(ANIMALS), verbose=False)[0]
        warn = False
        for i, box in enumerate(res.boxes):
            cls = int(box.cls[0])
            c = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id[0]) if box.id is not None else i

            if cls in ANIMALS:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(frame, f"{res.names[cls]} {c:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                rows.append({"frame": n, "id": tid, "class": res.names[cls], "conf": round(c, 3)})
                continue

            pad = 10
            cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            r = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            color, label, score = (0, 200, 0), "pedestrian", 0
            row = {"frame": n, "id": tid, "class": "person", "conf": round(c, 3)}

            if r.pose_landmarks:
                cw, ch = cx2 - cx1, cy2 - cy1
                pts = [(cx1 + l.x * cw, cy1 + l.y * ch, l.visibility) for l in r.pose_landmarks.landmark]
                for a, b in conn:
                    cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                             (int(pts[b][0]), int(pts[b][1])), (255, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                lean, head, ank = features(pts)
                ank_hist[tid].append(ank)
                cx_hist[tid].append((x1 + x2) / 2)
                bh = max(y2 - y1, 1)
                speed = 0.0
                if len(ank_hist[tid]) >= 3:
                    a0, a1 = ank_hist[tid][0], ank_hist[tid][-1]
                    speed = math.hypot(a1[0] - a0[0], a1[1] - a0[1]) / (len(ank_hist[tid]) - 1) / bh

                score += abs(lean) > lean_t
                score += abs(head) > head_t
                score += speed > speed_t
                if road_px is not None and len(cx_hist[tid]) >= 3:
                    dx = cx_hist[tid][-1] - cx_hist[tid][0]
                    toward = abs(cx_hist[tid][-1] - road_px) < abs(cx_hist[tid][0] - road_px)
                    score += toward and abs(dx) > 2
                if score >= 2:
                    color, label, warn = (0, 0, 255), "WILL CROSS?", True

                row.update({"lean_deg": round(lean, 2), "head_offset": round(head, 3),
                            "leg_speed": round(speed, 4), "intent_score": int(score)})

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID{tid} {label} {c:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            rows.append(row)

        if warn:
            alerts += 1
            cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 255), -1)
            cv2.putText(frame, "WARNING: CROSSING INTENT DETECTED", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if alerts in (1, 25, 60) and len(samples) < 6:
                samples.append((f"Alert frame {n}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        elif n % 40 == 1 and len(samples) < 6:
            samples.append((f"Frame {n}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        writer.write(frame)
        bar.progress(min(n / max(total, 1), 1.0), text=f"Processing frame {n}/{total}")

    cap.release()
    writer.release()

    # re-encode to H.264 so the browser can play it
    play_path = raw_path
    try:
        import imageio_ffmpeg
        out = tempfile.mktemp(suffix=".mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw_path,
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p", out],
                       check=True, capture_output=True)
        play_path = out
    except Exception:
        pass
    return play_path, pd.DataFrame(rows), samples, n, alerts


# ---------------- UI ----------------
st.title("Edge AI Based Pedestrian and Animal Crossing Intention Prediction for ADAS")
st.caption("Zeroth Review demo: YOLOv8 detection + tracking, MediaPipe pose, feature extraction, "
           "rule-based intent (placeholder for the LSTM stage).")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detection confidence", 0.1, 0.9, 0.4, 0.05)
    road = st.slider("Road edge position (0 = off)", 0.0, 1.0, 0.0, 0.05)
    lean_t = st.slider("Lean threshold (deg)", 2.0, 30.0, 8.0)
    head_t = st.slider("Head turn threshold", 0.1, 1.0, 0.35)
    speed_t = st.slider("Leg speed threshold", 0.005, 0.05, 0.015, 0.005)
    max_frames = st.slider("Max frames to process", 50, 600, 250, 50)

up = st.file_uploader("Upload a short road/pedestrian video", type=["mp4", "avi", "mov", "mkv"])

if up is not None and st.button("Run pipeline", type="primary"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
    tmp.write(up.read())
    tmp.close()
    bar = st.progress(0.0, text="Starting...")
    video, df, samples, n, alerts = process(tmp.name, conf, road, lean_t, head_t,
                                            speed_t, max_frames, bar)
    bar.empty()
    st.success(f"Processed {n} frames, {alerts} frames raised a crossing-intent warning.")

    st.subheader("1. Annotated output (detection + pose + warning)")
    st.video(video)
    with open(video, "rb") as f:
        st.download_button("Download annotated video", f, "annotated.mp4")

    if samples:
        st.subheader("2. Sample frames (use these in your PPT)")
        cols = st.columns(3)
        for k, (cap_txt, img) in enumerate(samples):
            cols[k % 3].image(img, caption=cap_txt)

    persons = df[df["class"] == "person"] if not df.empty else df
    if not persons.empty and "lean_deg" in persons.columns:
        st.subheader("3. Extracted features (features.csv)")
        st.dataframe(persons.head(200), use_container_width=True)
        st.download_button("Download features.csv", persons.to_csv(index=False), "features.csv")

        st.subheader("4. Feature trend for one pedestrian")
        pid = st.selectbox("Pedestrian ID", sorted(persons["id"].unique()))
        one = persons[persons["id"] == pid].set_index("frame")
        c1, c2 = st.columns(2)
        c1.caption("Body lean (deg)")
        c1.line_chart(one["lean_deg"])
        c2.caption("Leg speed (per frame / body height)")
        c2.line_chart(one["leg_speed"])
    else:
        st.info("No person with a usable pose was found. Try a clearer video or lower the confidence.")
