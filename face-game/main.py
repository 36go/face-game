import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
import pygame
from PIL import Image, ImageSequence

from model_assets import (
    FACE_LANDMARKER_MODEL,
    HAND_LANDMARKER_MODEL,
    ModelAssetError,
    ensure_model_asset,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_IMAGE_NAMES = ("neutral", "smile", "sad", "angry", "jew", "nien")
CLASPED_FRAMES_THRESHOLD = 3
GESTURE_HOLD_TIME = 0.5
OI_GIF_DURATION = 4.0
SOUND_COOLDOWN = 3.0
MOUTH_COUNT_WINDOW = 2.0


@dataclass
class MouthTriggerState:
    mouth_open_times: list[float] = field(default_factory=list)
    last_mouth_state: bool = False
    last_sound_time: float = float("-inf")

    def observe(self, mouth_is_open: bool, current_time: float) -> bool:
        self.mouth_open_times = [
            opened_at
            for opened_at in self.mouth_open_times
            if current_time - opened_at <= MOUTH_COUNT_WINDOW
        ]
        triggered = False

        if mouth_is_open and not self.last_mouth_state:
            self.mouth_open_times.append(current_time)
            print(f"Mouth open detected! Count: {len(self.mouth_open_times)}/3")
            if len(self.mouth_open_times) >= 3:
                if current_time - self.last_sound_time >= SOUND_COOLDOWN:
                    self.last_sound_time = current_time
                    triggered = True
                self.mouth_open_times.clear()

        self.last_mouth_state = mouth_is_open
        return triggered

    def reset_tracking(self) -> None:
        self.mouth_open_times.clear()
        self.last_mouth_state = False


def is_hand_open(hand_landmarks):
    return abs(hand_landmarks[20].x - hand_landmarks[4].x) > 0.12


def are_hands_clasped(hand_landmarks_list):
    if len(hand_landmarks_list) < 2:
        return False
    wrist1 = hand_landmarks_list[0][0]
    wrist2 = hand_landmarks_list[1][0]
    distance = np.sqrt((wrist1.x - wrist2.x) ** 2 + (wrist1.y - wrist2.y) ** 2)
    return distance < 0.25


def is_face_angry(face_landmarks):
    left_eyebrow = face_landmarks[70]
    right_eyebrow = face_landmarks[300]
    left_eye = face_landmarks[159]
    right_eye = face_landmarks[386]
    brow_distance = (
        (left_eye.y - left_eyebrow.y) + (right_eye.y - right_eyebrow.y)
    ) / 2
    mouth_open = face_landmarks[14].y - face_landmarks[13].y
    return brow_distance < 0.028 and mouth_open < 0.035


def is_face_smiling(face_landmarks):
    top_lip = face_landmarks[13]
    bottom_lip = face_landmarks[14]
    left_corner = face_landmarks[61]
    right_corner = face_landmarks[291]
    mouth_open = bottom_lip.y - top_lip.y
    mouth_center_y = (top_lip.y + bottom_lip.y) / 2
    left_raised = left_corner.y < mouth_center_y + 0.005
    right_raised = right_corner.y < mouth_center_y + 0.005
    return 0.012 < mouth_open < 0.05 and (left_raised or right_raised)


def is_face_sad(face_landmarks):
    return face_landmarks[14].y - face_landmarks[13].y > 0.055


def is_mouth_opening(face_landmarks):
    return face_landmarks[14].y - face_landmarks[13].y > 0.025


def detect_expression(face_landmarks, hand_open):
    if not hand_open and is_face_angry(face_landmarks):
        return "angry"
    if is_face_smiling(face_landmarks):
        return "smile"
    if is_face_sad(face_landmarks):
        return "sad"
    return "neutral"


def detect_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_mcp = hand_landmarks[2]
    index_tip = hand_landmarks[8]
    index_pip = hand_landmarks[6]
    middle_tip = hand_landmarks[12]
    middle_pip = hand_landmarks[10]
    ring_tip = hand_landmarks[16]
    ring_pip = hand_landmarks[14]
    pinky_tip = hand_landmarks[20]
    pinky_pip = hand_landmarks[18]

    thumb_up = thumb_tip.y < thumb_ip.y - 0.02 and thumb_ip.y < thumb_mcp.y
    fingers_folded = (
        index_tip.y > index_pip.y - 0.015
        and middle_tip.y > middle_pip.y - 0.015
        and ring_tip.y > ring_pip.y - 0.015
        and pinky_tip.y > pinky_pip.y - 0.015
    )
    thumb_highest = (
        thumb_tip.y < index_tip.y
        and thumb_tip.y < middle_tip.y
        and thumb_tip.y < ring_tip.y
        and thumb_tip.y < pinky_tip.y
    )
    return thumb_up and fingers_folded and thumb_highest


def detect_nazi_salute(hand_landmarks):
    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]
    index_pip = hand_landmarks[6]
    middle_pip = hand_landmarks[10]
    ring_pip = hand_landmarks[14]
    pinky_pip = hand_landmarks[18]

    all_fingers_extended = (
        index_tip.y < index_pip.y
        and middle_tip.y < middle_pip.y
        and ring_tip.y < ring_pip.y
        and pinky_tip.y < pinky_pip.y
    )
    fingers_together = (
        abs(index_tip.x - middle_tip.x) < 0.08
        and abs(middle_tip.x - ring_tip.x) < 0.08
        and abs(ring_tip.x - pinky_tip.x) < 0.08
    )
    hand_raised = middle_tip.y < wrist.y - 0.15
    vertical_alignment = abs(index_tip.y - pinky_tip.y) < 0.1
    return all_fingers_extended and fingers_together and hand_raised and vertical_alignment


def hand_gesture(hand_landmarks):
    if detect_nazi_salute(hand_landmarks):
        return "nazi_salute"
    if detect_thumbs_up(hand_landmarks):
        return "thumbs_up"
    if is_hand_open(hand_landmarks):
        return "open"
    return "closed"


def load_static_images(base_dir: Path = BASE_DIR) -> dict[str, np.ndarray]:
    images = {}
    for name in STATIC_IMAGE_NAMES:
        image = cv2.imread(str(base_dir / f"{name}.png"))
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        images[name] = image
    return images


def load_gif_frames(path: Path) -> list[np.ndarray]:
    frames = []
    try:
        with Image.open(path) as animation:
            for image in ImageSequence.Iterator(animation):
                rgb_frame = np.array(image.convert("RGB"))
                frames.append(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    except (OSError, ValueError) as error:
        print(f"Warning: could not load {path.name} ({error})")
    return frames


def initialize_audio() -> bool:
    try:
        pygame.mixer.init()
        return pygame.mixer.get_init() is not None
    except pygame.error as error:
        print(f"Warning: audio is unavailable ({error})")
        return False


def load_sound(path: Path):
    try:
        if pygame.mixer.get_init() is None:
            return None
        return pygame.mixer.Sound(os.fspath(path))
    except (FileNotFoundError, pygame.error) as error:
        print(f"Warning: could not load {path.name} ({error})")
        return None


def _safe_cleanup(name: str, callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception as error:
        print(f"Warning: could not clean up {name} ({error})")


def create_trackers(model_dir: Path | None = None, offline: bool | None = None):
    face_model = ensure_model_asset(FACE_LANDMARKER_MODEL, model_dir, offline)
    hand_model = ensure_model_asset(HAND_LANDMARKER_MODEL, model_dir, offline)
    vision = mp.tasks.vision
    face_tracker = None
    hand_tracker = None
    try:
        face_tracker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=os.fspath(face_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
        hand_tracker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=os.fspath(hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
    except Exception:
        if hand_tracker is not None:
            _safe_cleanup("hand tracker", lambda: hand_tracker.close())
        if face_tracker is not None:
            _safe_cleanup("face tracker", lambda: face_tracker.close())
        raise

    return face_tracker, hand_tracker, vision


def animated_frame(
    frames: list[np.ndarray],
    started_at: float | None,
    current_time: float,
) -> tuple[np.ndarray, float]:
    if not frames:
        raise ValueError("Animation has no frames")
    if started_at is None:
        started_at = current_time
    frame_index = int((current_time - started_at) / 0.033) % len(frames)
    return frames[frame_index], started_at


def cleanup_runtime(cap, face_tracker, hand_tracker, audio_initialized: bool) -> None:
    if cap is not None:
        _safe_cleanup("camera", lambda: cap.release())
    _safe_cleanup("OpenCV windows", cv2.destroyAllWindows)
    if face_tracker is not None:
        _safe_cleanup("face tracker", lambda: face_tracker.close())
    if hand_tracker is not None:
        _safe_cleanup("hand tracker", lambda: hand_tracker.close())
    if audio_initialized:
        _safe_cleanup("audio", pygame.mixer.quit)


def run_game(camera_index: int = 0, capture_factory=None) -> None:
    cap = None
    face_tracker = None
    hand_tracker = None
    audio_initialized = False

    try:
        audio_initialized = initialize_audio()
        images = load_static_images()
        gif_frames = load_gif_frames(BASE_DIR / "thumbs_up.gif")
        oi_gif_frames = load_gif_frames(BASE_DIR / "oi.gif")
        sound_effect = load_sound(BASE_DIR / "sound.mp3") if audio_initialized else None
        face_tracker, hand_tracker, vision = create_trackers()

        factory = capture_factory or cv2.VideoCapture
        cap = factory(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError("Could not open the default webcam")

        clasped_frames = 0
        last_was_clasped = False
        current_gesture = None
        gesture_start_time = None
        gif_start_time = None
        oi_gif_active = False
        oi_gif_start_time = None
        mouth_trigger = MouthTriggerState()
        last_timestamp_ms = -1
        hand_landmark_style = vision.drawing_styles.get_default_hand_landmarks_style()
        hand_connection_style = vision.drawing_styles.get_default_hand_connections_style()

        while True:
            frame_available, frame = cap.read()
            if not frame_available:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            face_results = face_tracker.detect_for_video(media_image, timestamp_ms)
            hand_results = hand_tracker.detect_for_video(media_image, timestamp_ms)
            display_img = images["neutral"]
            hands_open = 0
            hand_landmarks_list = []
            detected_gesture = None

            for hand_landmarks in hand_results.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    hand_landmark_style,
                    hand_connection_style,
                )
                gesture = hand_gesture(hand_landmarks)
                if gesture == "nazi_salute":
                    detected_gesture = "nazi_salute"
                elif gesture == "thumbs_up":
                    detected_gesture = "thumbs_up"
                elif gesture == "open":
                    hands_open += 1
                hand_landmarks_list.append(hand_landmarks)

            face_landmarks_list = face_results.face_landmarks
            if face_landmarks_list and sound_effect is not None and oi_gif_frames:
                mouth_is_open = is_mouth_opening(face_landmarks_list[0])
                if mouth_trigger.observe(mouth_is_open, time.time()):
                    try:
                        sound_effect.play()
                    except (FileNotFoundError, pygame.error) as error:
                        print(f"Warning: could not play sound.mp3 ({error})")
                    else:
                        print("Playing sound and showing gif!")
                        oi_gif_active = True
                        oi_gif_start_time = None
            elif not face_landmarks_list:
                mouth_trigger.reset_tracking()
            else:
                mouth_trigger.reset_tracking()

            current_time = time.time()
            if (
                oi_gif_active
                and oi_gif_start_time is not None
                and current_time - oi_gif_start_time > OI_GIF_DURATION
            ):
                oi_gif_active = False
                oi_gif_start_time = None

            if oi_gif_active:
                display_img, oi_gif_start_time = animated_frame(
                    oi_gif_frames,
                    oi_gif_start_time,
                    current_time,
                )
            else:
                if detected_gesture:
                    if current_gesture != detected_gesture:
                        current_gesture = detected_gesture
                        gesture_start_time = current_time
                    elif current_time - gesture_start_time >= GESTURE_HOLD_TIME:
                        if current_gesture == "nazi_salute":
                            display_img = images["nien"]
                        elif current_gesture == "thumbs_up" and gif_frames:
                            display_img, gif_start_time = animated_frame(
                                gif_frames,
                                gif_start_time,
                                current_time,
                            )
                else:
                    current_gesture = None
                    gesture_start_time = None
                    gif_start_time = None

                gesture_is_held = (
                    detected_gesture
                    and gesture_start_time is not None
                    and current_time - gesture_start_time >= GESTURE_HOLD_TIME
                )
                if not gesture_is_held:
                    hand_open_flag = hands_open > 0
                    hands_clasped = are_hands_clasped(hand_landmarks_list)
                    single_closed_hand = len(hand_landmarks_list) == 1 and all(
                        not is_hand_open(hand) for hand in hand_landmarks_list
                    )
                    if hands_clasped or (single_closed_hand and last_was_clasped):
                        clasped_frames += 1
                    else:
                        clasped_frames = 0
                    last_was_clasped = hands_clasped or single_closed_hand

                    if clasped_frames >= CLASPED_FRAMES_THRESHOLD:
                        display_img = images["jew"]
                    elif face_landmarks_list:
                        expression = detect_expression(face_landmarks_list[0], hand_open_flag)
                        display_img = images.get(expression, images["neutral"])

            height, width = display_img.shape[:2]
            frame_height, frame_width = frame.shape[:2]
            scale = min(frame_height / height, frame_width / width)
            new_size = (int(width * scale), int(height * scale))
            display_img_resized = cv2.resize(display_img, new_size)

            cv2.imshow("Camera", frame)
            cv2.imshow("Expression", display_img_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cleanup_runtime(cap, face_tracker, hand_tracker, audio_initialized)


def main() -> None:
    try:
        run_game()
    except ModelAssetError as error:
        raise SystemExit(f"Model setup error: {error}") from error


if __name__ == "__main__":
    main()
