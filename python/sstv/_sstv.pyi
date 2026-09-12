import os
from typing import BinaryIO, Sequence, Union

import numpy as np
import numpy.typing as npt
import PIL.Image

_Samples = Union[
    npt.NDArray[np.int16],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
    Sequence[int],
]

_Recording = Union[str, os.PathLike[str], bytes, bytearray, BinaryIO]

_Image = Union[PIL.Image.Image, npt.NDArray[np.uint8]]

class Mode:
    """An SSTV transmission mode."""

    SCOTTIE_1: Mode
    SCOTTIE_2: Mode
    SCOTTIE_DX: Mode
    MARTIN_1: Mode
    MARTIN_2: Mode
    ROBOT_36: Mode
    ROBOT_72: Mode
    WRASSE_SC2_180: Mode
    PASOKON_P3: Mode
    PASOKON_P5: Mode
    PASOKON_P7: Mode
    PD_50: Mode
    PD_90: Mode
    PD_120: Mode
    PD_160: Mode
    PD_180: Mode
    PD_240: Mode
    PD_290: Mode

    @property
    def image_width(self) -> int:
        """The horizontal resolution in pixels."""

    @property
    def image_height(self) -> int:
        """The vertical resolution in pixels."""

def decode(
    samples: _Samples,
    sample_rate: int,
    *,
    mode: Mode | None = None,
    header: bool = True,
) -> list[PIL.Image.Image]:
    """Decode every SSTV image contained in a stream of audio samples.

    Decode metadata is stored in each image's ``info`` dict under the
    ``sstv_mode`` (a ``Mode``) and ``sstv_complete`` (a ``bool``) keys.
    """

def decode_from_wav(
    wav: _Recording,
    *,
    mode: Mode | None = None,
    header: bool = True,
) -> list[PIL.Image.Image]:
    """Decode every SSTV image contained in a WAV recording.

    Accepts a path, in-memory WAV data, or a binary file-like object. Decode
    metadata is stored in each image's ``info`` dict under the ``sstv_mode``
    (a ``Mode``) and ``sstv_complete`` (a ``bool``) keys.
    """

def decode_from_mp3(
    mp3: _Recording,
    *,
    mode: Mode | None = None,
    header: bool = True,
) -> list[PIL.Image.Image]:
    """Decode every SSTV image contained in an MP3 recording.

    Accepts a path, in-memory MP3 data, or a binary file-like object. Decode
    metadata is stored in each image's ``info`` dict under the ``sstv_mode``
    (a ``Mode``) and ``sstv_complete`` (a ``bool``) keys.
    """

def encode(
    image: _Image,
    mode: Mode,
    sample_rate: int = 48000,
) -> npt.NDArray[np.int16]:
    """Encode an image into the raw audio samples of an SSTV transmission.

    The image dimensions must match the mode's resolution exactly.
    """

def encode_to_wav(
    image: _Image,
    mode: Mode,
    sample_rate: int = 48000,
) -> bytes:
    """Encode an image into a complete WAV file of an SSTV transmission.

    The image dimensions must match the mode's resolution exactly.
    """

def encode_to_mp3(
    image: _Image,
    mode: Mode,
    sample_rate: int = 48000,
) -> bytes:
    """Encode an image into a complete MP3 file of an SSTV transmission.

    The image dimensions must match the mode's resolution exactly, and the
    sample rate must be one supported by MP3.
    """

def encode_to_mp3_file(
    image: _Image,
    path: str | os.PathLike[str],
    mode: Mode,
    sample_rate: int = 48000,
) -> None:
    """Encode an image into an SSTV transmission and write it to an MP3 file.

    The image dimensions must match the mode's resolution exactly, and the
    sample rate must be one supported by MP3.
    """

def encode_to_wav_file(
    image: _Image,
    path: str | os.PathLike[str],
    mode: Mode,
    sample_rate: int = 48000,
) -> None:
    """Encode an image into an SSTV transmission and write it to a WAV file.

    The image dimensions must match the mode's resolution exactly.
    """
