"""Tests for sstv.decode()."""

import numpy as np
import pysstv.color
import pytest
from PIL import Image

import sstv
from tests.helpers import (
    HEIGHT,
    SAMPLE_RATE,
    WIDTH,
    assert_matches,
    header_sample_count,
    make_image,
    mean_abs_error,
    pysstv_encode,
)


@pytest.fixture(scope="module")
def decoded(samples: np.ndarray) -> Image.Image:
    return sstv.decode(samples, SAMPLE_RATE)[0]


def test_round_trip_with_explicit_mode(samples, image):
    decoded = sstv.decode(samples, SAMPLE_RATE, mode=sstv.Mode.ROBOT_36)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_auto_mode_detects_from_header(samples, image):
    decoded = sstv.decode(samples, SAMPLE_RATE)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


@pytest.mark.parametrize(
    ("mode_cls", "mode"),
    [
        (pysstv.color.MartinM2, sstv.Mode.MARTIN_2),
        (pysstv.color.ScottieS2, sstv.Mode.SCOTTIE_2),
        (pysstv.color.PD90, sstv.Mode.PD_90),
    ],
)
def test_round_trip_other_modes(mode_cls, mode):
    # PySSTV transmits Martin 2 / Scottie 2 at 160 pixels per line while the
    # decoder renders the mode's full 320; encode at PySSTV's native size and
    # upscale the reference to match.
    image = make_image(mode_cls.WIDTH, mode_cls.HEIGHT)
    samples = pysstv_encode(image, mode_cls, 22_050)
    decoded = sstv.decode(samples, 22_050)
    assert len(decoded) == 1
    expected = np.repeat(image, mode.image_width // mode_cls.WIDTH, axis=1)
    assert_matches(decoded[0], expected, mode)


def test_two_images_with_gap(samples, image):
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.int16)
    stream = np.concatenate([samples, silence, samples])
    decoded = sstv.decode(stream, SAMPLE_RATE)
    assert len(decoded) == 2
    for img in decoded:
        assert_matches(img, image)


def test_without_header(samples, image):
    header = header_sample_count(pysstv.color.Robot36, SAMPLE_RATE)
    decoded = sstv.decode(
        samples[header:], SAMPLE_RATE, mode=sstv.Mode.ROBOT_36, header=False
    )
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_truncated_signal_yields_incomplete_image(samples, image):
    decoded = sstv.decode(samples[: len(samples) // 2], SAMPLE_RATE)
    assert len(decoded) == 1
    img = decoded[0]
    assert not img.info["sstv_complete"]
    assert img.size == (WIDTH, HEIGHT)
    pixels = np.asarray(img)
    # The decoded upper half should match; the missing rows are black.
    assert mean_abs_error(pixels[: HEIGHT // 3], image[: HEIGHT // 3]) < 12.0
    assert np.all(pixels[-1] == 0)


def test_silence_yields_no_images():
    silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
    assert sstv.decode(silence, SAMPLE_RATE) == []


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_samples(samples, image, dtype):
    scaled = (samples.astype(dtype) / np.iinfo(np.int16).max).astype(dtype)
    decoded = sstv.decode(scaled, SAMPLE_RATE)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_int_sequence():
    assert sstv.decode([0] * SAMPLE_RATE, SAMPLE_RATE) == []


def test_non_contiguous_array(samples, image):
    padded = np.zeros(len(samples) * 2, dtype=np.int16)
    padded[::2] = samples
    decoded = sstv.decode(padded[::2], SAMPLE_RATE)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_unsupported_dtype_raises():
    with pytest.raises(TypeError, match="dtype"):
        sstv.decode(np.zeros(100, dtype=np.int32), SAMPLE_RATE)  # ty: ignore[invalid-argument-type]


def test_multidimensional_array_raises():
    with pytest.raises(TypeError, match="one-dimensional"):
        sstv.decode(np.zeros((100, 2), dtype=np.int16), SAMPLE_RATE)


def test_out_of_range_sequence_raises():
    with pytest.raises(TypeError):
        sstv.decode([0, 100_000], SAMPLE_RATE)


def test_zero_sample_rate_raises():
    with pytest.raises(ValueError, match="sample_rate"):
        sstv.decode(np.zeros(100, dtype=np.int16), 0)


def test_returns_pil_images(decoded):
    assert isinstance(decoded, Image.Image)
    assert decoded.size == (WIDTH, HEIGHT)
    assert decoded.mode == "RGB"
    assert np.asarray(decoded).shape == (HEIGHT, WIDTH, 3)


def test_metadata(decoded):
    assert decoded.info["sstv_mode"] == sstv.Mode.ROBOT_36
    assert decoded.info["sstv_complete"] is True
