"""Decode slow-scan television (SSTV) transmissions.

This package wraps the Rust `sstv <https://crates.io/crates/sstv>`_ crate.
The entry point is :func:`decode`, which takes raw audio samples and returns
every SSTV image found in them as a Pillow image::

    import sstv
    import soundfile

    samples, sample_rate = soundfile.read("recording.wav")
    for image in sstv.decode(samples, sample_rate):
        image.save("decoded.png")

Decode metadata is stored in each image's ``info`` dict: ``sstv_mode`` holds
the :class:`Mode` the image was transmitted in, and ``sstv_complete`` is
``False`` if the signal cut off before the image's last scanline.

Supported modes: Scottie 1/2/DX, Martin 1/2, Robot 36/72, Wrasse SC2-180,
Pasokon P3/P5/P7, and PD 50-290 (see :class:`Mode`). By default the mode is
detected automatically from each transmission's VIS header.
"""

from sstv._sstv import Mode, decode

__all__ = ["Mode", "decode"]
