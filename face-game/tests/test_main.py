import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pygame
import pytest


APP_DIR = Path(__file__).resolve().parents[1]


def load_main_module(name: str):
    spec = importlib.util.spec_from_file_location(name, APP_DIR / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


main = load_main_module("face_game_main")


def test_pinned_mediapipe_exposes_tasks_api() -> None:
    assert main.mp.tasks.vision.FaceLandmarker is not None
    assert main.mp.tasks.vision.HandLandmarker is not None
    assert main.mp.tasks.vision.RunningMode.VIDEO is not None


def test_trackers_start_and_close_without_camera() -> None:
    model_dir = os.environ.get("FACE_GAME_TEST_MODEL_DIR")
    if not model_dir:
        pytest.skip("FACE_GAME_TEST_MODEL_DIR is required for the model integration test")
    face_tracker, hand_tracker, _vision_api = main.create_trackers(
        Path(model_dir),
        offline=True,
    )
    try:
        image = main.mp.Image(
            image_format=main.mp.ImageFormat.SRGB,
            data=np.zeros((32, 32, 3), dtype=np.uint8),
        )
        assert face_tracker.detect_for_video(image, 0).face_landmarks == []
        assert hand_tracker.detect_for_video(image, 0).hand_landmarks == []
    finally:
        main.cleanup_runtime(None, face_tracker, hand_tracker, audio_initialized=False)


def test_import_does_not_initialize_audio_or_open_camera(monkeypatch) -> None:
    camera_calls = []
    audio_calls = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera_calls.append(args))
    monkeypatch.setattr(pygame.mixer, "init", lambda: audio_calls.append(True))

    load_main_module("face_game_import_check")

    assert camera_calls == []
    assert audio_calls == []


def test_missing_sound_is_optional(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: (44100, -16, 2))

    def missing_sound(_path: str):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(pygame.mixer, "Sound", missing_sound)

    assert main.load_sound(tmp_path / "sound.mp3") is None


def test_model_setup_error_is_reported_without_starting_camera(monkeypatch) -> None:
    def fail_model_setup() -> None:
        raise main.ModelAssetError("offline model cache is missing")

    monkeypatch.setattr(main, "run_game", fail_model_setup)

    with pytest.raises(SystemExit, match="Model setup error: offline model cache is missing"):
        main.main()


def test_failed_camera_is_released_and_all_cleanup_continues(monkeypatch) -> None:
    cleanup_calls = []

    class FailedCamera:
        def set(self, _property, _value) -> None:
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            cleanup_calls.append("camera")
            raise RuntimeError("release failed")

    class Tracker:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            cleanup_calls.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    face_tracker = Tracker("face", fail=True)
    hand_tracker = Tracker("hands")
    monkeypatch.setattr(main, "initialize_audio", lambda: True)
    monkeypatch.setattr(
        main,
        "load_static_images",
        lambda: {name: np.zeros((2, 2, 3), dtype=np.uint8) for name in main.STATIC_IMAGE_NAMES},
    )
    monkeypatch.setattr(main, "load_gif_frames", lambda _path: [])
    monkeypatch.setattr(main, "load_sound", lambda _path: None)
    monkeypatch.setattr(
        main,
        "create_trackers",
        lambda: (
            face_tracker,
            hand_tracker,
            SimpleNamespace(),
        ),
    )

    def destroy_windows() -> None:
        cleanup_calls.append("windows")
        raise RuntimeError("window cleanup failed")

    monkeypatch.setattr(main.cv2, "destroyAllWindows", destroy_windows)
    monkeypatch.setattr(main.pygame.mixer, "quit", lambda: cleanup_calls.append("audio"))

    with pytest.raises(RuntimeError, match="Could not open"):
        main.run_game(capture_factory=lambda _index: FailedCamera())

    assert cleanup_calls == ["camera", "windows", "face", "hands", "audio"]


def test_tracking_loss_clears_partial_mouth_history() -> None:
    state = main.MouthTriggerState()
    assert state.observe(True, 0.0) is False
    assert state.observe(False, 0.1) is False
    assert state.observe(True, 0.2) is False

    state.reset_tracking()

    assert state.mouth_open_times == []
    assert state.last_mouth_state is False
    assert state.observe(True, 0.3) is False
    assert state.mouth_open_times == [0.3]
