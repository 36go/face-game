import cv2
import mediapipe as mp
from PIL import Image, ImageSequence
import numpy as np
import os
import time
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()

# Setup mediapipe with balanced settings for smooth performance
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Optimized for i7 11th gen - balanced quality and performance
face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=False,
    max_num_faces=1
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# Automatically detect the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load static images
static_images = ["neutral", "smile", "sad", "angry", "jew", "nien"]
images = {}
for name in static_images:
    path = os.path.join(BASE_DIR, f"{name}.png")
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    images[name] = img

# Load and prepare GIF frames
thumbs_up_gif_path = os.path.join(BASE_DIR, "thumbs_up.gif")
thumbs_up_gif = Image.open(thumbs_up_gif_path)
gif_frames = []
for img in ImageSequence.Iterator(thumbs_up_gif):
    img_frame = np.array(img.convert("RGB"))
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)
    gif_frames.append(img_frame)

# Load oi.gif
oi_gif_path = os.path.join(BASE_DIR, "oi.gif")
oi_gif_frames = []
try:
    oi_gif = Image.open(oi_gif_path)
    for img in ImageSequence.Iterator(oi_gif):
        img_frame = np.array(img.convert("RGB"))
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)
        oi_gif_frames.append(img_frame)
except:
    print("Warning: oi.gif not found")
    oi_gif_frames = None

# Load sound file
sound_path = os.path.join(BASE_DIR, "sound.mp3")
try:
    sound_effect = pygame.mixer.Sound(sound_path)
except:
    print("Warning: sound.mp3 not found")
    sound_effect = None

# Global variables
clasped_frames = 0
CLASPED_FRAMES_THRESHOLD = 3
last_was_clasped = False
current_gif_frame = 0
gif_start_time = None

# Oi gif variables
oi_gif_active = False
oi_gif_start_time = None
oi_current_frame = 0
OI_GIF_DURATION = 4.0  # Show oi.gif for 4 seconds

# Sound trigger variables
mouth_open_count = 0
last_mouth_state = False
last_count_reset_time = time.time()
SOUND_COOLDOWN = 3.0
MOUTH_COUNT_WINDOW = 2.0  # Time window to detect 3 mouth opens

# Hand helper functions
def is_hand_open(hand_landmarks):
    thumb_x = hand_landmarks.landmark[4].x
    pinky_x = hand_landmarks.landmark[20].x
    return abs(pinky_x - thumb_x) > 0.12

def are_hands_clasped(hand_landmarks_list):
    if len(hand_landmarks_list) < 2:
        return False
    wrist1 = hand_landmarks_list[0].landmark[0]
    wrist2 = hand_landmarks_list[1].landmark[0]
    dist = np.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2)
    return dist < 0.25

# Face expression functions
def is_face_angry(face_landmarks):
    left_eyebrow = face_landmarks.landmark[70]
    right_eyebrow = face_landmarks.landmark[300]
    left_eye = face_landmarks.landmark[159]
    right_eye = face_landmarks.landmark[386]
    
    left_distance = left_eye.y - left_eyebrow.y
    right_distance = right_eye.y - right_eyebrow.y
    brow_distance = (left_distance + right_distance) / 2
    
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    mouth_open = bottom_lip.y - top_lip.y
    
    return brow_distance < 0.028 and mouth_open < 0.035

def is_face_smiling(face_landmarks):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]
    
    mouth_open = bottom_lip.y - top_lip.y
    mouth_center_y = (top_lip.y + bottom_lip.y) / 2
    left_raised = left_corner.y < mouth_center_y + 0.005
    right_raised = right_corner.y < mouth_center_y + 0.005
    
    mouth_slightly_open = 0.012 < mouth_open < 0.05
    
    return mouth_slightly_open and (left_raised or right_raised)

def is_face_sad(face_landmarks):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    mouth_open = bottom_lip.y - top_lip.y
    mouth_wide_open = mouth_open > 0.055
    return mouth_wide_open

def is_mouth_opening(face_landmarks):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    mouth_open = bottom_lip.y - top_lip.y
    # Lower threshold for easier detection
    return mouth_open > 0.025

def detect_expression(face_landmarks, hand_open):
    if not hand_open and is_face_angry(face_landmarks):
        return "angry"
    if is_face_smiling(face_landmarks):
        return "smile"
    if is_face_sad(face_landmarks):
        return "sad"
    return "neutral"

# Gesture detection
def detect_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    middle_tip = hand_landmarks.landmark[12]
    middle_pip = hand_landmarks.landmark[10]
    ring_tip = hand_landmarks.landmark[16]
    ring_pip = hand_landmarks.landmark[14]
    pinky_tip = hand_landmarks.landmark[20]
    pinky_pip = hand_landmarks.landmark[18]
    
    thumb_up = thumb_tip.y < thumb_ip.y - 0.02 and thumb_ip.y < thumb_mcp.y
    
    fingers_folded = (
        index_tip.y > index_pip.y - 0.015 and
        middle_tip.y > middle_pip.y - 0.015 and
        ring_tip.y > ring_pip.y - 0.015 and
        pinky_tip.y > pinky_pip.y - 0.015
    )
    
    thumb_highest = (
        thumb_tip.y < index_tip.y and
        thumb_tip.y < middle_tip.y and
        thumb_tip.y < ring_tip.y and
        thumb_tip.y < pinky_tip.y
    )
    
    return thumb_up and fingers_folded and thumb_highest

def detect_nazi_salute(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    index_pip = hand_landmarks.landmark[6]
    middle_pip = hand_landmarks.landmark[10]
    ring_pip = hand_landmarks.landmark[14]
    pinky_pip = hand_landmarks.landmark[18]
    
    all_fingers_extended = (
        index_tip.y < index_pip.y and
        middle_tip.y < middle_pip.y and
        ring_tip.y < ring_pip.y and
        pinky_tip.y < pinky_pip.y
    )
    
    fingers_together = (
        abs(index_tip.x - middle_tip.x) < 0.08 and
        abs(middle_tip.x - ring_tip.x) < 0.08 and
        abs(ring_tip.x - pinky_tip.x) < 0.08
    )
    
    hand_raised = middle_tip.y < wrist.y - 0.15
    vertical_alignment = abs(index_tip.y - pinky_tip.y) < 0.1
    
    return all_fingers_extended and fingers_together and hand_raised and vertical_alignment

def hand_gesture(hand_landmarks):
    if detect_nazi_salute(hand_landmarks):
        return "nazi_salute"
    elif detect_thumbs_up(hand_landmarks):
        return "thumbs_up"
    elif is_hand_open(hand_landmarks):
        return "open"
    else:
        return "closed"

# GIF display functions
def get_gif_frame():
    global current_gif_frame, gif_start_time
    
    if gif_start_time is None:
        gif_start_time = time.time()
    
    elapsed = time.time() - gif_start_time
    frame_duration = 0.033
    current_gif_frame = int(elapsed / frame_duration) % len(gif_frames)
    
    return gif_frames[current_gif_frame]

def get_oi_gif_frame():
    global oi_current_frame, oi_gif_start_time
    
    if oi_gif_start_time is None:
        oi_gif_start_time = time.time()
    
    elapsed = time.time() - oi_gif_start_time
    frame_duration = 0.033
    oi_current_frame = int(elapsed / frame_duration) % len(oi_gif_frames)
    
    return oi_gif_frames[oi_current_frame]

# Camera setup with optimal resolution for smooth playback
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

current_gesture = None
gesture_start_time = None
GESTURE_HOLD_TIME = 0.5

# No frame skipping for smooth video
frame_skip = 0
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for smoothness

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face
    face_results = face_mesh.process(frame_rgb)
    expression = "neutral"
    hand_open_flag = False
    
    # Detect hands
    hand_results = hands.process(frame_rgb)
    hands_open = 0
    hand_landmarks_list = []
    detected_gesture = None
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            gesture = hand_gesture(hand_landmarks)
            
            if gesture == "nazi_salute":
                detected_gesture = "nazi_salute"
            elif gesture == "thumbs_up":
                detected_gesture = "thumbs_up"
            elif gesture == "open":
                hands_open += 1
            
            hand_landmarks_list.append(hand_landmarks)
    
    # Check for "Oi oi oi" mouth pattern with simplified detection
    if face_results.multi_face_landmarks and sound_effect and oi_gif_frames:
        face_landmarks = face_results.multi_face_landmarks[0]
        current_time = time.time()
        
        mouth_currently_open = is_mouth_opening(face_landmarks)
        
        # Reset counter if too much time has passed
        if current_time - last_count_reset_time > MOUTH_COUNT_WINDOW:
            mouth_open_count = 0
        
        # Detect mouth opening transition (closed to open)
        if mouth_currently_open and not last_mouth_state:
            mouth_open_count += 1
            last_count_reset_time = current_time
            print(f"Mouth open detected! Count: {mouth_open_count}/3")  # Debug
            
            # If 3 mouth opens detected in window, play sound and trigger gif
            if mouth_open_count >= 3:
                print("Playing sound and showing gif!")  # Debug
                sound_effect.play()
                mouth_open_count = 0
                # Trigger oi.gif
                oi_gif_active = True
                oi_gif_start_time = None
        
        last_mouth_state = mouth_currently_open
    
    # Check if oi.gif should stop
    if oi_gif_active and oi_gif_start_time is not None:
        if time.time() - oi_gif_start_time > OI_GIF_DURATION:
            oi_gif_active = False
            oi_gif_start_time = None
            oi_current_frame = 0
    
    # Display oi.gif if active
    if oi_gif_active:
        display_img = get_oi_gif_frame()
    else:
        # Confirm gesture
        if detected_gesture:
            if current_gesture != detected_gesture:
                current_gesture = detected_gesture
                gesture_start_time = time.time()
            elif time.time() - gesture_start_time >= GESTURE_HOLD_TIME:
                if current_gesture == "nazi_salute":
                    display_img = images["nien"]
                elif current_gesture == "thumbs_up":
                    display_img = get_gif_frame()
        else:
            current_gesture = None
            gesture_start_time = None
            gif_start_time = None
            current_gif_frame = 0
        
        # If no special gesture
        if not detected_gesture or (gesture_start_time and time.time() - gesture_start_time < GESTURE_HOLD_TIME):
            hand_open_flag = hands_open > 0
            
            hands_clasped = are_hands_clasped(hand_landmarks_list)
            hands_single_closed = len(hand_landmarks_list) == 1 and all(not is_hand_open(h) for h in hand_landmarks_list)
            
            if hands_clasped or (hands_single_closed and last_was_clasped):
                clasped_frames += 1
            else:
                clasped_frames = 0
            
            last_was_clasped = hands_clasped or hands_single_closed
            
            if clasped_frames >= CLASPED_FRAMES_THRESHOLD:
                display_img = images["jew"]
            elif face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                expression = detect_expression(face_landmarks, hand_open_flag)
                display_img = images.get(expression, images["neutral"])
            else:
                display_img = images["neutral"]
    
    # Resize image to fit window
    h, w = display_img.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    scale = min(frame_h/h, frame_w/w)
    new_size = (int(w*scale), int(h*scale))
    display_img_resized = cv2.resize(display_img, new_size)
    
    # Display windows
    cv2.imshow("Camera", frame)
    cv2.imshow("Expression", display_img_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()