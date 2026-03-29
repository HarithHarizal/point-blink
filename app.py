"""
Eye Tracker Mouse Control v6 (Program by Miguel meshed with Web app program)
----------------------------
Features:
- Smooth cursor movement via Iris tracking.
- Left Click: Blink Left Eye.
- Right Click: Blink Right Eye.
- Pause/Resume: Press 'p' or double blink to toggle control (allows using physical mouse).
- Exit: Press 'q' or close eyes 3 seconds.

    • Helps people who have limited use of their hands
	• It can also help people in multitasking where they can use their eyes as an extra pair of limbs.
	• Allows people regardless of ability to use more hands-free technology

Utilizes: OpenCV | PyAutoGUI | Streamlit |  MediaPipe Ver 10.14
"""
# --- imports ---
# Streamlit web application integration
import streamlit as st
# OpenCV to access Webcam
import cv2
# Mediapipe to access FaceMesh feature
import mediapipe as mp
# Pyautogui to affect cursor
import pyautogui
# Time for measuring time eyes were closed
import time
# Increase speeds in numpy library operations
import numpy as np
# Saving the last values for smoothing cursor glide
from collections import deque
# --- LOAD EXTERNAL CSS ---
# This is for website styling
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"SYSTEM ERROR: {file_name} not found in root directory.")

st.set_page_config(page_title="NEURAL_LINK_v6", layout="wide")

# Inject the separate CSS file
local_css("style.css")

# --- HEADER SECTION ---
st.image("https://placehold.co/1200x200/0d0221/00ffff?text=NEURAL+INTERFACE+BOOTING...+ACCESS+GRANTED",
         use_container_width=True)

st.title("👁️ NEURAL_LINK // EYE TRACKER v6.0")
st.markdown("""Features:
- Smooth cursor movement via Iris tracking.

- Pause/Resume: Press 'p' or double blink to toggle control (allows using physical mouse).
- To Exit: Press 'q' or close eyes 3 seconds.

- Helps people who have limited use of their hands
- It can also help people in multitasking where they can use their eyes as an extra pair of limbs.
- Allows people regardless of ability to use more hands-free technology
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://i.imgur.com/NRo3tye.png")
    st.header("⚙️ SYSTEM_PARAMS")
    smoothing_window = st.slider("Signal Smoothing", 1, 20, 8)
    click_threshold = st.slider("Ocular Sensitivity", 0.001, 0.02, 0.007, format="%.3f")
    cam_id = st.selectbox("Hardware Source", ["WEBCAM_0", "WEBCAM_1"], index=0)
    cam_id_int = 0 if cam_id == "WEBCAM_0" else 1

# --- INIT SESSION WITH IF STATEMENTS ---
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False
if 'run' not in st.session_state:
    st.session_state.run = False

smooth_x = deque(maxlen=smoothing_window)
smooth_y = deque(maxlen=smoothing_window)

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# --- MAIN INTERFACE ---
col_main, col_stats = st.columns([3, 1])

with col_main:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("INITIALIZE LINK"):
            st.session_state.run = True
    with c2:
        if st.button("TERMINATE"):
            st.session_state.run = False

    frame_placeholder = st.empty()

with col_stats:
    st.subheader("📡 STATUS_LOG")
    status_box = st.empty()
    st.write("**Left-Wink:** LEFT_CLICK")
    st.write("**Right-Wink:** RIGHT_CLICK")
    st.write("**Pause-Unpause:** DOUBLE_BLINK")
    st.write("**Long-Close:** EMERGENCY_STOP")

    # --- DYNAMIC STATUS IMAGE SWITCH ---
    if st.session_state.run:
        st.image("https://i.imgur.com/pL8SgGY.png", use_container_width=True)  # Online
    else:
        st.image("https://i.imgur.com/SQcBPB8.png", use_container_width=True)  # Offline

# --- TRACKING LOGIC ---
if st.session_state.run:
    cam = cv2.VideoCapture(cam_id_int)
    last_blink_time = 0
    eyes_closed_start = None
    DOUBLE_BLINK_GAP = 0.5
    EXIT_CLOSE_DURATION = 3.0

    while cam.isOpened() and st.session_state.run:
        success, frame = cam.read()
        if not success:
            st.error("HARDWARE_FAILURE: CAMERA NOT FOUND")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. MEASURE EYES
            left_eye_dist = landmarks[145].y - landmarks[159].y
            right_eye_dist = landmarks[374].y - landmarks[386].y
            both_closed = left_eye_dist < click_threshold and right_eye_dist < click_threshold

            # 2. EXIT/PAUSE LOGIC
            if both_closed:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                elif time.time() - eyes_closed_start > EXIT_CLOSE_DURATION:
                    st.session_state.run = False
                    break
            else:
                if eyes_closed_start is not None:
                    duration = time.time() - eyes_closed_start
                    if duration < 0.3:
                        if (time.time() - last_blink_time) < DOUBLE_BLINK_GAP:
                            st.session_state.is_paused = not st.session_state.is_paused
                        last_blink_time = time.time()
                    eyes_closed_start = None

            # 3. MOUSE CONTROL
            if not st.session_state.is_paused:
                iris_center = landmarks[468]
                input_x = (iris_center.x - 0.4) / 0.2
                input_y = (iris_center.y - 0.4) / 0.2

                smooth_x.append(input_x * screen_w)
                smooth_y.append(input_y * screen_h)

                if len(smooth_x) > 0:
                    target_x = sum(smooth_x) / len(smooth_x)
                    target_y = sum(smooth_y) / len(smooth_y)
                    pyautogui.moveTo(target_x, target_y, _pause=False)

                if left_eye_dist < click_threshold and not both_closed:
                    pyautogui.click(button='left')
                    time.sleep(0.1)
                elif right_eye_dist < click_threshold and not both_closed:
                    pyautogui.click(button='right')
                    time.sleep(0.1)

        # UI Overlay - Cyan for active, Neon Pink for paused
        status_text = ">> SYSTEM_PAUSED <<" if st.session_state.is_paused else ">> LINK_ACTIVE <<"
        status_color = (255, 0, 255) if st.session_state.is_paused else (255, 255, 0)  # BGR

        # Draw tech-lines on frame
        cv2.rectangle(frame, (0, 0), (w, h), status_color, 2)
        cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, status_color, 2)

        status_box.info(status_text)
        frame_placeholder.image(frame, channels="BGR")

    cam.release()
else:
    st.info("System Standby... Waiting for Neural Link initialization...")
