"""Tests for sstv.decode_from_mp3()."""

from pathlib import Path

import lameenc  # ty: ignore[unresolved-import]  # binary-only module without stubs
import numpy as np
import pytest

import sstv
from tests.helpers import SAMPLE_RATE, assert_matches


def mp3_bytes(samples: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """The samples as a 128 kbit/s MP3; multi-channel samples interleaved."""
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)
    encoder.silence()
    return bytes(encoder.encode(samples.tobytes()) + encoder.flush())


@pytest.fixture(scope="module")
def mp3(samples: np.ndarray) -> bytes:
    return mp3_bytes(samples, SAMPLE_RATE)


def test_from_bytes(mp3, image):
    decoded = sstv.decode_from_mp3(mp3)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_from_path(tmp_path: Path, mp3, image):
    path = tmp_path / "transmission.mp3"
    path.write_bytes(mp3)
    decoded = sstv.decode_from_mp3(path)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_uses_first_channel_of_stereo(samples, image):
    stereo = np.stack([samples, np.zeros_like(samples)], axis=1)
    decoded = sstv.decode_from_mp3(mp3_bytes(stereo, SAMPLE_RATE, channels=2))
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_garbage_raises():
    with pytest.raises(ValueError, match="MP3"):
        sstv.decode_from_mp3(b"this is not an mp3 file")


def test_invalid_type_raises():
    with pytest.raises(TypeError, match="path, bytes, or a binary file-like"):
        sstv.decode_from_mp3(42)  # ty: ignore[invalid-argument-type]
