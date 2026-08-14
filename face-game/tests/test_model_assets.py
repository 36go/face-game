import hashlib
from pathlib import Path

import pytest

from model_assets import ModelAsset, ModelAssetError, ensure_model_asset


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        pass

    def read(self, size: int) -> bytes:
        chunk = self.content[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def model_asset(content: bytes) -> ModelAsset:
    return ModelAsset(
        name="Test model",
        filename="test.task",
        url="https://models.example.invalid/test.task",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )


def test_verified_download_is_cached_and_reused(tmp_path: Path) -> None:
    content = b"verified model content"
    asset = model_asset(content)
    requests = []

    def opener(url: str, timeout: int):
        requests.append((url, timeout))
        return FakeResponse(content)

    cached_path = ensure_model_asset(asset, tmp_path, opener=opener)
    reused_path = ensure_model_asset(
        asset,
        tmp_path,
        opener=lambda *_args, **_kwargs: pytest.fail("valid cache must not download"),
    )

    assert cached_path.read_bytes() == content
    assert reused_path == cached_path
    assert requests == [(asset.url, 30)]
    assert list(tmp_path.glob("*.tmp")) == []


def test_hash_mismatch_is_rejected_without_replacing_cache(tmp_path: Path) -> None:
    asset = model_asset(b"expected")

    with pytest.raises(ModelAssetError, match="failed size or SHA-256 verification"):
        ensure_model_asset(
            asset,
            tmp_path,
            opener=lambda _url, timeout: FakeResponse(b"tampered"),
        )

    assert not (tmp_path / asset.filename).exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_offline_missing_model_has_actionable_error(tmp_path: Path) -> None:
    asset = model_asset(b"expected")

    with pytest.raises(ModelAssetError) as raised:
        ensure_model_asset(asset, tmp_path, offline=True)

    message = str(raised.value)
    assert "offline mode is enabled" in message
    assert asset.url in message
    assert asset.sha256 in message
    assert "FACE_GAME_MODEL_DIR" in message
