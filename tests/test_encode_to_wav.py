"""Tests for sstv.encode_to_wav() and sstv.encode_to_wav_file()."""

import io
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sstv
from tests.helpers import assert_matches


def test_round_trip(image):
    data = sstv.encode_to_wav(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    with wave.open(io.BytesIO(data)) as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 22_050
    decoded = sstv.decode_from_wav(data)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_matches_raw_samples(image):
    data = sstv.encode_to_wav(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    with wave.open(io.BytesIO(data)) as reader:
        frames = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
    samples = sstv.encode(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    assert np.array_equal(frames, samples)


def test_wrong_size_raises(image):
    small = Image.fromarray(image).resize((100, 100))
    with pytest.raises(ValueError, match="resize"):
        sstv.encode_to_wav(small, sstv.Mode.ROBOT_36)


def test_file_round_trip(tmp_path: Path, image):
    path = tmp_path / "transmission.wav"
    sstv.encode_to_wav_file(image, path, sstv.Mode.ROBOT_36, sample_rate=22_050)
    decoded = sstv.decode_from_wav(path)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_file_matches_bytes(tmp_path: Path, image):
    path = tmp_path / "transmission.wav"
    sstv.encode_to_wav_file(image, str(path), sstv.Mode.ROBOT_36, sample_rate=22_050)
    assert path.read_bytes() == sstv.encode_to_wav(
        image, sstv.Mode.ROBOT_36, sample_rate=22_050
    )


def test_file_unwritable_path_raises(tmp_path: Path, image):
    with pytest.raises(OSError):
        sstv.encode_to_wav_file(
            image, tmp_path / "missing-dir" / "out.wav", sstv.Mode.ROBOT_36
        )
