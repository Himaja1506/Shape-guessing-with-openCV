import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# -----------------------------
# Helper Functions
# -----------------------------
def finger_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def recognize_shape(contour):
    if cv2.contourArea(contour) < 1000:
        return None
    approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 0.8 < aspect_ratio < 1.2:
            return "Square"
        else:
            return "Rectangle"
    elif len(approx) > 5:
        # Check if roughly circular
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * radius * radius
        contour_area = cv2.contourArea(contour)
        if 0.7 < contour_area/circle_area < 1.3:
            return "Circle"
        else:
            return "Ellipse"
    else:
        return "Unknown"

# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Camera and Canvas Setup
# -----------------------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Cannot open camera")
    exit()

height, width = frame.shape[:2]
canvas = np.zeros((height, width, 3), dtype=np.uint8)
last_point = None
brush_color = (0, 0, 255)
brush_size = 5
point_buffer = deque(maxlen=5)
shape_label = ""

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_point = None
    detect_shape = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        index = finger_up(landmarks, 8, 6)
        middle = finger_up(landmarks, 12, 10)
        ring = finger_up(landmarks, 16, 14)
        pinky = finger_up(landmarks, 20, 18)
        thumb = landmarks[4].x > landmarks[3].x

        fingers_up = [thumb, index, middle, ring, pinky]
        total_fingers = sum(fingers_up)

        # Clear canvas if 5 fingers
        if total_fingers == 5:
            canvas.fill(0)
            last_point = None
            point_buffer.clear()
            shape_label = ""

        # Draw if only index finger up
        elif total_fingers == 1 and index:
            x = int(landmarks[8].x * width)
            y = int(landmarks[8].y * height)
            point_buffer.append((x, y))
            # Smooth point using weighted average
            avg_x = int(np.mean([p[0] for p in point_buffer]))
            avg_y = int(np.mean([p[1] for p in point_buffer]))
            draw_point = (avg_x, avg_y)

            if last_point is not None:
                cv2.line(canvas, last_point, draw_point, brush_color, brush_size)
            last_point = draw_point
        else:
            # Gesture lifted, now detect shape
            last_point = None
            point_buffer.clear()
            detect_shape = True
    else:
        last_point = None
        point_buffer.clear()
        detect_shape = True

    # -----------------------------
    # Shape recognition after lifting finger
    # -----------------------------
    if detect_shape:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            shape_label = recognize_shape(cnt)
            if shape_label:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.putText(canvas, shape_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # -----------------------------
    # Combine webcam & canvas
    # -----------------------------
    combined = np.hstack((frame, canvas))
    cv2.imshow("Hand Drawing & Shape Detection", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
