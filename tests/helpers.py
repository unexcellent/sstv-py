"""Shared helpers for the test suite.

Decoding test signals are generated with the pure-Python PySSTV package
(https://github.com/dnet/pySSTV), so decoding is checked against an
independent SSTV implementation rather than the Rust crate's own encoder.
"""

from itertools import islice

import numpy as np
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


def pysstv_encode(image: np.ndarray, mode_cls, sample_rate: int) -> np.ndarray:
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
