"""Encode and decode slow-scan television (SSTV) transmissions.

This package wraps the Rust `sstv <https://crates.io/crates/sstv>`_ crate.
:func:`decode_wav` and :func:`decode_mp3` take a recording and return every
SSTV image found in it as a Pillow image; :func:`decode` does the same for
raw audio samples. :func:`encode` is the inverse, turning an image into the
samples of a transmission::

    import sstv
    from PIL import Image

    for image in sstv.decode_wav("recording.wav"):
        image.save("decoded.png")

    photo = Image.open("photo.png").resize((320, 240))
    samples = sstv.encode(photo, sstv.Mode.ROBOT_36)

Decode metadata is stored in each image's ``info`` dict: ``sstv_mode`` holds
the :class:`Mode` the image was transmitted in, and ``sstv_complete`` is
``False`` if the signal cut off before the image's last scanline.

Supported modes: Scottie 1/2/DX, Martin 1/2, Robot 36/72, Wrasse SC2-180,
Pasokon P3/P5/P7, and PD 50-290 (see :class:`Mode`). By default the mode is
detected automatically from each transmission's VIS header when decoding.
"""

from sstv._sstv import Mode, decode, decode_mp3, decode_wav, encode

__all__ = ["Mode", "decode", "decode_mp3", "decode_wav", "encode"]
