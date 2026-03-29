import cv2
import numpy as np
import time
from collections import deque

# ----------------------------
# Settings
# ----------------------------
GAZE_HOLD_TIME = 0.7        # Seconds to hold gaze before recording
COOLDOWN_TIME = 0.6         # Minimum time between two gaze recordings
HISTORY_LENGTH = 5           # Frames used to smooth gaze
MAX_ATTEMPTS = 3
LOCKOUT_DURATION = 30

# ----------------------------
# Load detectors
# ----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------------------------
# State variables
# ----------------------------
gaze_sequence = []
password_pattern = None
current_mode = "IDLE"

last_gaze = None
gaze_start_time = 0
last_record_time = 0
failed_attempts = 0
lockout_until = 0

gaze_history = deque(maxlen=HISTORY_LENGTH)  # For smoothing gaze

# ----------------------------
# Helper functions
# ----------------------------
def get_pupil_position(eye_img):
    if eye_img.size == 0:
        return None
    try:
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 10:
            return None
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)
    except:
        return None

def get_gaze_direction(cx, eye_width):
    ratio = cx / eye_width
    if ratio < 0.4: return "LEFT"
    elif ratio > 0.6: return "RIGHT"
    return "CENTER"

def compress_sequence(seq):
    compressed = []
    for s in seq:
        if not compressed or compressed[-1] != s:
            compressed.append(s)
    return compressed

def draw_gaze_trail(frame, sequence):
    if not sequence: return
    arrow_map = {"LEFT": "<", "RIGHT": ">", "CENTER": "O"}
    colors = [(0,255,255), (0,210,255), (0,165,255), (0,120,255), (0,75,255)]
    cv2.putText(frame, "Trail:", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
    last_moves = sequence[-5:]
    for i, gaze in enumerate(last_moves):
        symbol = arrow_map.get(gaze, gaze)
        cv2.putText(frame, symbol, (100 + i*40, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i % len(colors)], 2)

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # --- LOCKOUT HANDLING ---
    if current_time < lockout_until:
        remaining = int(lockout_until - current_time)
        frame[:] = (0,0,60)
        cv2.putText(frame, "LOCKED OUT", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),3)
        cv2.putText(frame, f"Try again in {remaining}s", (160,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255),2)
        cv2.imshow("Gaze Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # --- FACE & EYE DETECTION ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    current_gaze = None

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,200,0),1)
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,8,minSize=(20,20))
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,100),1)
            margin = 4
            eye_img = roi_color[ey+margin:ey+eh-margin, ex+margin:ex+ew-margin]
            pupil = get_pupil_position(eye_img)
            if pupil:
                cx, cy = pupil
                gaze_history.append(get_gaze_direction(cx, eye_img.shape[1]))
                # Majority vote for smoothing
                current_gaze = max(set(gaze_history), key=gaze_history.count)
                cv2.circle(eye_img,(cx,cy),4,(0,255,0),-1)

    # --- STABLE RECORDING ---
    if current_gaze:
        if current_gaze == last_gaze:
            held_time = current_time - gaze_start_time
            bar_len = int(min((held_time / GAZE_HOLD_TIME)*150, 150))
            cv2.rectangle(frame,(20,175),(20+bar_len,195),(0,255,150),-1)
            cv2.rectangle(frame,(20,175),(170,195),(100,100,100),1)

            if held_time > GAZE_HOLD_TIME and (current_time - last_record_time) > COOLDOWN_TIME:
                gaze_sequence.append(current_gaze)
                last_record_time = current_time
                print("Recorded:", current_gaze)
        else:
            last_gaze = current_gaze
            gaze_start_time = current_time

    # --- DRAW UI ---
    mode_color = (0,255,0) if "GRANTED" in current_mode else (0,0,255) if "DENIED" in current_mode or "LOCKED" in current_mode else (255,255,0)
    cv2.putText(frame, f"Mode: {current_mode}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.9,mode_color,2)
    cv2.putText(frame, f"Gaze: {current_gaze if current_gaze else '--'}", (20,80), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0) if current_gaze else (80,80,80),2)
    cv2.putText(frame, f"Attempts: {failed_attempts}/{MAX_ATTEMPTS}", (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,180,255),2)
    cv2.putText(frame, f"Recorded: {len(gaze_sequence)} moves", (20,150), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
    cv2.putText(frame, "S=Save  V=Verify  R=Reset  Q=Quit", (20,frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1)
    draw_gaze_trail(frame,gaze_sequence)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        password_pattern = compress_sequence(gaze_sequence)
        gaze_sequence=[]
        failed_attempts=0
        current_mode="PATTERN SAVED"
        print("Saved Pattern:", password_pattern)
    elif key == ord('v'):
        input_pattern = compress_sequence(gaze_sequence)
        gaze_sequence=[]
        if password_pattern and input_pattern == password_pattern:
            current_mode = "ACCESS GRANTED"
            failed_attempts = 0
            print("ACCESS GRANTED")
        else:
            failed_attempts += 1
            if failed_attempts >= MAX_ATTEMPTS:
                lockout_until = current_time + LOCKOUT_DURATION
                current_mode = "LOCKED OUT"
            else:
                current_mode = f"DENIED - {MAX_ATTEMPTS - failed_attempts} left"
            print(f"ACCESS DENIED ({failed_attempts}/{MAX_ATTEMPTS})")
    elif key == ord('r'):
        gaze_sequence=[]
        current_mode="RESET"
        print("Reset")
    elif key == ord('q'):
        break

    cv2.imshow("Gaze Authentication", frame)

cap.release()
cv2.destroyAllWindows()