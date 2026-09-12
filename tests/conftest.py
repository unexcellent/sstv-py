import numpy as np
import pysstv.color
import pytest

from tests.helpers import HEIGHT, SAMPLE_RATE, WIDTH, make_image, pysstv_encode


@pytest.fixture(scope="session")
def image() -> np.ndarray:
    return make_image(WIDTH, HEIGHT)


@pytest.fixture(scope="session")
def samples(image: np.ndarray) -> np.ndarray:
    """The test image as a Robot 36 transmission."""
    return pysstv_encode(image, pysstv.color.Robot36, SAMPLE_RATE)
