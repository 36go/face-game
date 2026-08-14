import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


MODEL_LICENSE_URL = "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE"


class ModelAssetError(RuntimeError):
    """Raised when a required model cannot be loaded with verified contents."""


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    url: str
    sha256: str
    size: int


FACE_LANDMARKER_MODEL = ModelAsset(
    name="Face Landmarker",
    filename="face_landmarker-float16-v1.task",
    url=(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    size=3_758_596,
)

HAND_LANDMARKER_MODEL = ModelAsset(
    name="Hand Landmarker",
    filename="hand_landmarker-float16-v1.task",
    url=(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    size=7_819_105,
)


def default_model_dir() -> Path:
    configured = os.environ.get("FACE_GAME_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    if os.name == "nt":
        cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "face-game" / "models"


def _offline_requested() -> bool:
    return os.environ.get("FACE_GAME_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _file_matches(path: Path, asset: ModelAsset) -> bool:
    try:
        if path.stat().st_size != asset.size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == asset.sha256
    except OSError:
        return False


def _asset_error(asset: ModelAsset, target: Path, reason: str) -> ModelAssetError:
    return ModelAssetError(
        f"{asset.name} is unavailable: {reason}. Expected verified model at {target}. "
        f"Official source: {asset.url}. SHA-256: {asset.sha256}. "
        "Connect once to populate the cache, or place the exact model in "
        "FACE_GAME_MODEL_DIR before running with FACE_GAME_OFFLINE=1."
    )


def ensure_model_asset(
    asset: ModelAsset,
    model_dir: Path | None = None,
    offline: bool | None = None,
    opener=None,
) -> Path:
    cache_dir = Path(model_dir) if model_dir is not None else default_model_dir()
    target = cache_dir / asset.filename
    if _file_matches(target, asset):
        return target

    offline_mode = _offline_requested() if offline is None else offline
    if offline_mode:
        raise _asset_error(asset, target, "offline mode is enabled and the cache is missing or invalid")

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _asset_error(asset, target, f"the cache directory could not be created ({error})") from error

    open_url = opener or urlopen
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{asset.filename}.",
            suffix=".tmp",
            dir=cache_dir,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            digest = hashlib.sha256()
            downloaded_size = 0
            with open_url(asset.url, timeout=30) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary_file.write(chunk)
                    digest.update(chunk)
                    downloaded_size += len(chunk)

        if downloaded_size != asset.size or digest.hexdigest() != asset.sha256:
            raise _asset_error(asset, target, "the downloaded file failed size or SHA-256 verification")

        os.replace(temporary_path, target)
        temporary_path = None
        return target
    except ModelAssetError:
        raise
    except (OSError, TimeoutError, URLError) as error:
        raise _asset_error(asset, target, f"the verified download failed ({error})") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
