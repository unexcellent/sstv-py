"""Tests for the encoding functions.

Encoding is tested by round-tripping through the decoder, which is itself
validated against PySSTV in the decoding tests.
"""

import io
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sstv
from tests.helpers import HEIGHT, WIDTH, assert_matches, make_image


@pytest.mark.parametrize(
    "mode",
    [sstv.Mode.ROBOT_36, sstv.Mode.MARTIN_1, sstv.Mode.SCOTTIE_1, sstv.Mode.PD_90],
)
def test_round_trip(mode):
    image = make_image(mode.image_width, mode.image_height)
    samples = sstv.encode(image, mode, sample_rate=22_050)
    # No mode passed: detecting it exercises the encoded VIS header.
    decoded = sstv.decode(samples, 22_050)
    assert len(decoded) == 1
    assert_matches(decoded[0], image, mode)


def test_default_sample_rate_is_48khz(image):
    samples = sstv.encode(image, sstv.Mode.ROBOT_36)
    decoded = sstv.decode(samples, 48_000)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_returns_int16_samples(image):
    samples = sstv.encode(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    assert samples.dtype == np.int16
    assert samples.ndim == 1


def test_accepts_pil_and_numpy_equally(image):
    from_numpy = sstv.encode(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    from_pil = sstv.encode(
        Image.fromarray(image), sstv.Mode.ROBOT_36, sample_rate=22_050
    )
    assert np.array_equal(from_numpy, from_pil)


def test_to_wav_round_trip(image):
    data = sstv.encode_to_wav(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    with wave.open(io.BytesIO(data)) as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 22_050
    decoded = sstv.decode_from_wav(data)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_to_wav_matches_raw_samples(image):
    data = sstv.encode_to_wav(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    with wave.open(io.BytesIO(data)) as reader:
        frames = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
    samples = sstv.encode(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    assert np.array_equal(frames, samples)


def test_to_wav_wrong_size_raises(image):
    small = Image.fromarray(image).resize((100, 100))
    with pytest.raises(ValueError, match="resize"):
        sstv.encode_to_wav(small, sstv.Mode.ROBOT_36)


def test_to_wav_file_round_trip(tmp_path: Path, image):
    path = tmp_path / "transmission.wav"
    sstv.encode_to_wav_file(image, path, sstv.Mode.ROBOT_36, sample_rate=22_050)
    decoded = sstv.decode_from_wav(path)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_to_wav_file_matches_to_wav(tmp_path: Path, image):
    path = tmp_path / "transmission.wav"
    sstv.encode_to_wav_file(image, str(path), sstv.Mode.ROBOT_36, sample_rate=22_050)
    assert path.read_bytes() == sstv.encode_to_wav(
        image, sstv.Mode.ROBOT_36, sample_rate=22_050
    )


def test_to_wav_file_unwritable_path_raises(tmp_path: Path, image):
    with pytest.raises(OSError):
        sstv.encode_to_wav_file(
            image, tmp_path / "missing-dir" / "out.wav", sstv.Mode.ROBOT_36
        )


def test_to_mp3_round_trip(image):
    data = sstv.encode_to_mp3(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    decoded = sstv.decode_from_mp3(data)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_to_mp3_unsupported_sample_rate_raises(image):
    with pytest.raises(ValueError, match="sample_rate"):
        sstv.encode_to_mp3(image, sstv.Mode.ROBOT_36, sample_rate=12_345)


def test_wrong_pil_size_raises(image):
    small = Image.fromarray(image).resize((100, 100))
    with pytest.raises(ValueError, match="resize"):
        sstv.encode(small, sstv.Mode.ROBOT_36)


def test_wrong_numpy_shape_raises():
    with pytest.raises(ValueError, match="resolution"):
        sstv.encode(np.zeros((100, 100, 3), dtype=np.uint8), sstv.Mode.ROBOT_36)


def test_wrong_numpy_dtype_raises():
    with pytest.raises(TypeError, match="uint8"):
        sstv.encode(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64), sstv.Mode.ROBOT_36)  # ty: ignore[invalid-argument-type]


def test_invalid_type_raises():
    with pytest.raises(TypeError, match="PIL.Image"):
        sstv.encode("not an image", sstv.Mode.ROBOT_36)  # ty: ignore[invalid-argument-type]


def test_zero_sample_rate_raises(image):
    with pytest.raises(ValueError, match="sample_rate"):
        sstv.encode(image, sstv.Mode.ROBOT_36, sample_rate=0)
