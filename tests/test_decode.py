"""Tests for sstv.decode().

Test signals are generated with the pure-Python PySSTV package
(https://github.com/dnet/pySSTV), so decoding is checked against an
independent SSTV implementation rather than the Rust crate's own encoder.
"""

import io
import wave
from itertools import islice
from pathlib import Path

import lameenc  # ty: ignore[unresolved-import]  # binary-only module without stubs
import numpy as np
import pysstv.color
import pytest
from PIL import Image

import sstv

SAMPLE_RATE = 48_000
WIDTH = 320
HEIGHT = 240


def make_image(width: int, height: int) -> np.ndarray:
    """A test image with variation in all three channels."""
    y, x = np.mgrid[0:height, 0:width]
    red = x * 255 // (width - 1)
    green = y * 255 // (height - 1)
    blue = (x + y) * 255 // (width - 1 + height - 1)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def encode(image: np.ndarray, mode_cls, sample_rate: int) -> np.ndarray:
    """The image as an SSTV transmission (header included), via PySSTV."""
    transmission = mode_cls(Image.fromarray(image), sample_rate, bits=16)
    return np.fromiter(transmission.gen_samples(), dtype=np.int16)


def header_sample_count(mode_cls, sample_rate: int) -> int:
    """The number of samples PySSTV's header occupies before the image data.

    The first 13 (frequency, msec) tuples of gen_freq_bits are the header:
    leader, break, leader, VIS start bit, 7 data bits, parity, stop bit.
    """
    transmission = mode_cls(Image.new("RGB", (1, 1)), sample_rate, bits=16)
    header_ms = sum(ms for _, ms in islice(transmission.gen_freq_bits(), 13))
    return round(header_ms * sample_rate / 1000)


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return make_image(WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def samples(image: np.ndarray) -> np.ndarray:
    """The test image as a Robot 36 transmission."""
    return encode(image, pysstv.color.Robot36, SAMPLE_RATE)


@pytest.fixture(scope="module")
def decoded(samples: np.ndarray) -> Image.Image:
    return sstv.decode(samples, SAMPLE_RATE)[0]


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute per-channel error between two images of equal shape."""
    assert a.shape == b.shape
    return float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32))))


def assert_matches(
    decoded: Image.Image, image: np.ndarray, mode: sstv.Mode = sstv.Mode.ROBOT_36
) -> None:
    assert decoded.info["sstv_complete"], "image should decode completely"
    assert decoded.info["sstv_mode"] == mode
    assert decoded.mode == "RGB"
    assert (decoded.height, decoded.width) == image.shape[:2]
    error = mean_abs_error(np.asarray(decoded), image)
    assert error < 12.0, f"mean abs error {error} too high"


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
    samples = encode(image, mode_cls, 22_050)
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


def test_decode_wav_from_bytes(wav, image):
    decoded = sstv.decode_wav(wav)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_wav_from_path(tmp_path: Path, wav, image):
    path = tmp_path / "transmission.wav"
    path.write_bytes(wav)
    for argument in (path, str(path)):
        decoded = sstv.decode_wav(argument)
        assert len(decoded) == 1
        assert_matches(decoded[0], image)


def test_decode_wav_from_file_object(wav, image):
    decoded = sstv.decode_wav(io.BytesIO(wav))
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_wav_uses_first_channel_of_stereo(samples, image):
    stereo = np.stack([samples, np.zeros_like(samples)], axis=1)
    decoded = sstv.decode_wav(wav_bytes(stereo, SAMPLE_RATE, channels=2))
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_wav_with_explicit_mode(wav, image):
    decoded = sstv.decode_wav(wav, mode=sstv.Mode.ROBOT_36)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_wav_without_header(samples, image):
    header = header_sample_count(pysstv.color.Robot36, SAMPLE_RATE)
    wav = wav_bytes(samples[header:], SAMPLE_RATE)
    decoded = sstv.decode_wav(wav, mode=sstv.Mode.ROBOT_36, header=False)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_wav_malformed_raises():
    with pytest.raises(ValueError, match="WAV"):
        sstv.decode_wav(b"this is not a wav file")


def test_decode_wav_missing_file_raises(tmp_path: Path):
    with pytest.raises(OSError):
        sstv.decode_wav(tmp_path / "does-not-exist.wav")


def test_decode_wav_invalid_type_raises():
    with pytest.raises(TypeError, match="path, bytes, or a binary file-like"):
        sstv.decode_wav(42)  # ty: ignore[invalid-argument-type]


def test_decode_wav_text_file_object_raises(tmp_path: Path, wav):
    path = tmp_path / "transmission.wav"
    path.write_bytes(wav)
    with open(path, encoding="latin-1") as handle:
        with pytest.raises(TypeError, match="binary mode"):
            sstv.decode_wav(handle)  # ty: ignore[invalid-argument-type]


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


def test_decode_mp3_from_bytes(mp3, image):
    decoded = sstv.decode_mp3(mp3)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_mp3_from_path(tmp_path: Path, mp3, image):
    path = tmp_path / "transmission.mp3"
    path.write_bytes(mp3)
    decoded = sstv.decode_mp3(path)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_mp3_uses_first_channel_of_stereo(samples, image):
    stereo = np.stack([samples, np.zeros_like(samples)], axis=1)
    decoded = sstv.decode_mp3(mp3_bytes(stereo, SAMPLE_RATE, channels=2))
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_decode_mp3_garbage_raises():
    with pytest.raises(ValueError, match="MP3"):
        sstv.decode_mp3(b"this is not an mp3 file")


def test_decode_mp3_invalid_type_raises():
    with pytest.raises(TypeError, match="path, bytes, or a binary file-like"):
        sstv.decode_mp3(42)  # ty: ignore[invalid-argument-type]


def test_mode_dimensions():
    assert sstv.Mode.ROBOT_36.image_width == 320
    assert sstv.Mode.ROBOT_36.image_height == 240
    assert sstv.Mode.PD_290.image_width == 800
    assert sstv.Mode.PD_290.image_height == 616


def test_auto_mode_has_no_dimensions():
    with pytest.raises(ValueError, match="AUTO"):
        sstv.Mode.AUTO.image_width
    with pytest.raises(ValueError, match="AUTO"):
        sstv.Mode.AUTO.image_height


def test_mode_equality_and_hash():
    assert sstv.Mode.ROBOT_36 == sstv.Mode.ROBOT_36
    assert sstv.Mode.ROBOT_36 != sstv.Mode.ROBOT_72
    assert len({sstv.Mode.ROBOT_36, sstv.Mode.ROBOT_36, sstv.Mode.AUTO}) == 2
