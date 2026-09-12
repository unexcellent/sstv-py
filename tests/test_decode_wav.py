"""Tests for sstv.decode_wav()."""

import io
import wave
from pathlib import Path

import numpy as np
import pysstv.color
import pytest

import sstv
from tests.helpers import SAMPLE_RATE, assert_matches, header_sample_count


def wav_bytes(samples: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """The samples as a 16-bit PCM WAV; multi-channel samples interleaved."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.fixture(scope="module")
def wav(samples: np.ndarray) -> bytes:
    return wav_bytes(samples, SAMPLE_RATE)


def test_from_bytes(wav, image):
    decoded = sstv.decode_wav(wav)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_from_path(tmp_path: Path, wav, image):
    path = tmp_path / "transmission.wav"
    path.write_bytes(wav)
    for argument in (path, str(path)):
        decoded = sstv.decode_wav(argument)
        assert len(decoded) == 1
        assert_matches(decoded[0], image)


def test_from_file_object(wav, image):
    decoded = sstv.decode_wav(io.BytesIO(wav))
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_uses_first_channel_of_stereo(samples, image):
    stereo = np.stack([samples, np.zeros_like(samples)], axis=1)
    decoded = sstv.decode_wav(wav_bytes(stereo, SAMPLE_RATE, channels=2))
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_with_explicit_mode(wav, image):
    decoded = sstv.decode_wav(wav, mode=sstv.Mode.ROBOT_36)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_without_header(samples, image):
    header = header_sample_count(pysstv.color.Robot36, SAMPLE_RATE)
    wav = wav_bytes(samples[header:], SAMPLE_RATE)
    decoded = sstv.decode_wav(wav, mode=sstv.Mode.ROBOT_36, header=False)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_malformed_raises():
    with pytest.raises(ValueError, match="WAV"):
        sstv.decode_wav(b"this is not a wav file")


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(OSError):
        sstv.decode_wav(tmp_path / "does-not-exist.wav")


def test_invalid_type_raises():
    with pytest.raises(TypeError, match="path, bytes, or a binary file-like"):
        sstv.decode_wav(42)  # ty: ignore[invalid-argument-type]


def test_text_file_object_raises(tmp_path: Path, wav):
    path = tmp_path / "transmission.wav"
    path.write_bytes(wav)
    with open(path, encoding="latin-1") as handle:
        with pytest.raises(TypeError, match="binary mode"):
            sstv.decode_wav(handle)  # ty: ignore[invalid-argument-type]
